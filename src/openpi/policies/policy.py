from collections.abc import Sequence
import enum
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_fast_ricl as _pi0_fast_ricl
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.policies.utils import embed, embed_with_batches, load_dinov2, EMBED_DIM
import os
from autofaiss import build_index
import logging
from datetime import datetime
import json
from PIL import Image
logger = logging.getLogger()
BasePolicy: TypeAlias = _base_policy.BasePolicy


class RetrievalStrategy(str, enum.Enum):
    """Controls how retrieval examples are selected at inference time."""

    KNN = "knn"  # default behaviour: nearest neighbors via FAISS index
    RANDOM = "random"  # random samples from the demo pool
    NONE = "none"  # do not use demos; fill retrieved slots with blanks


def _load_processed_demo(path: str) -> np.lib.npyio.NpzFile:
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "pickled data" not in str(exc):
            raise
        logger.warning("Processed demo %s contains pickled data; reloading with allow_pickle=True", path)
        return np.load(path, allow_pickle=True)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._model = model

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        print(f'outputs: {outputs}')
        final_outputs = self._output_transform(outputs)
        logger.info(f'final_outputs: {final_outputs}')
        return final_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
    

def get_action_chunk_at_inference_time(actions, step_idx, action_horizon):
    num_steps = len(actions)
    action_chunk = []
    for i in range(action_horizon):
        if step_idx+i < num_steps:
            action_chunk.append(actions[step_idx+i])
        else:
            action_chunk.append(np.concatenate([np.zeros(actions.shape[-1]-1, dtype=np.float32), actions[-1, -1:]], axis=0)) # combines 0 joint vels with last gripper pos
    action_chunk = np.stack(action_chunk, axis=0)
    assert action_chunk.shape == (action_horizon, 8), f"{action_chunk.shape=}"
    return action_chunk


class RiclPolicy(BasePolicy):
    def __init__(
        self,
        model: _pi0_fast_ricl.Pi0FASTRicl,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        demos_dir: str | None = None,
        use_action_interpolation: bool | None = None,
        lamda: float | None = None,
        action_horizon: int | None = None,
        retrieval_strategy: str | RetrievalStrategy = RetrievalStrategy.KNN,
        retrieval_seed: int | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._model = model
        self._use_action_interpolation = use_action_interpolation
        self._lamda = lamda
        self._action_horizon = action_horizon
        self._retrieval_strategy = RetrievalStrategy(retrieval_strategy)
        self._retrieval_rng = np.random.default_rng(retrieval_seed)
        # setup demos for retrieval
        print()
        logger.info(f'loading demos from {demos_dir}...')
        demo_folders = [folder for folder in sorted(os.listdir(demos_dir)) if os.path.isdir(f"{demos_dir}/{folder}")]
        self._demos = {
            demo_idx: _load_processed_demo(f"{demos_dir}/{folder}/processed_demo.npz")
            for demo_idx, folder in enumerate(demo_folders)
        }
        if not self._demos:
            raise ValueError(f"No demos found in {demos_dir}")
        self._all_indices = np.array([(ep_idx, step_idx) for ep_idx in list(self._demos.keys()) for step_idx in range(self._demos[ep_idx]["actions"].shape[0])])
        _all_embeddings = np.concatenate([self._demos[ep_idx]["top_image_embeddings"] for ep_idx in list(self._demos.keys())])
        assert _all_embeddings.shape == (len(self._all_indices), EMBED_DIM), f"{_all_embeddings.shape=}"
        self._knn_k = self._model.num_retrieved_observations
        self._action_dim = next(iter(self._demos.values()))["actions"].shape[-1]
        self._blank_prompt = ""
        print()
        logger.info(f'building retrieval index...')
        self._knn_index, knn_index_infos = build_index(embeddings=_all_embeddings, # Note: embeddings have to be float to avoid errors in autofaiss / embedding_reader!
                                            save_on_disk=False,
                                            min_nearest_neighbors_to_retrieve=self._knn_k + 5, # default: 20
                                            max_index_query_time_ms=10, # default: 10
                                            max_index_memory_usage="25G", # default: "16G"
                                            current_memory_available="50G", # default: "32G"
                                            metric_type='l2',
                                            nb_cores=8, # default: None # "The number of cores to use, by default will use all cores" as seen in https://criteo.github.io/autofaiss/getting_started/quantization.html#the-build-index-command
                                            )
        # setup the dinov2 model for embedding only
        logger.info('loading dinov2 for image embedding...')
        self._dinov2 = load_dinov2()
        self._max_dist = json.load(open(f"assets/max_distance.json", 'r'))['distances']['max']
        print(f'self._max_dist: {self._max_dist} [helpful to carefully check this value in case of any issues]')
        # cache shapes for "blank" retrievals
        sample_demo = next(iter(self._demos.values()))
        self._blank_top_image = np.zeros_like(sample_demo["top_image"][0])
        self._blank_right_image = np.zeros_like(sample_demo["right_image"][0])
        self._blank_wrist_image = np.zeros_like(sample_demo["wrist_image"][0])
        self._blank_state = np.zeros_like(sample_demo["state"][0])
        self._blank_actions = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)

    def _fill_blank_slot(self, more_obs: dict, ct: int) -> None:
        """Populate the ct-th retrieval slot with blank observations."""
        more_obs[f"retrieved_{ct}_top_image"] = self._blank_top_image
        more_obs[f"retrieved_{ct}_right_image"] = self._blank_right_image
        more_obs[f"retrieved_{ct}_wrist_image"] = self._blank_wrist_image
        more_obs[f"retrieved_{ct}_state"] = self._blank_state
        more_obs[f"retrieved_{ct}_actions"] = self._blank_actions
        more_obs[f"retrieved_{ct}_prompt"] = self._blank_prompt

    def _select_retrieved_indices(
        self, strategy: RetrievalStrategy, query_embedding: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Select retrieved indices based on the configured strategy.

        Returns:
            retrieved_indices: shape (1, k, 2) or None when strategy is NONE.
            distances: distances to first retrieved/query for interpolation or None.
        """
        if strategy is RetrievalStrategy.NONE:
            return None, None

        if strategy is RetrievalStrategy.RANDOM:
            population = len(self._all_indices)
            replace = population < self._knn_k
            random_choices = self._retrieval_rng.choice(population, size=self._knn_k, replace=replace)
            retrieved_indices = self._all_indices[random_choices][np.newaxis, ...]
            return retrieved_indices, None

        # Default KNN retrieval
        if query_embedding is None:
            raise ValueError("query_embedding is required for knn retrieval")
        topk_distance, topk_indices = self._knn_index.search(query_embedding, self._knn_k)
        return self._all_indices[topk_indices], topk_distance

    def _build_blank_retrieval(self) -> dict:
        """Build empty retrieval slots to simulate 'no context'."""
        more_obs: dict[str, Any] = {"inference_time": True}
        for ct in range(self._knn_k):
            self._fill_blank_slot(more_obs, ct)
        if self._use_action_interpolation:
            exp_lamda = np.zeros((self._knn_k + 1, 1), dtype=np.float32)
            exp_lamda[-1, 0] = 1.0  # let the query dominate when no context is provided
            more_obs["exp_lamda_distances"] = exp_lamda
        return more_obs

    def retrieve(self, obs: dict) -> dict:
        more_obs = {"inference_time": True}
        requested_count = obs.pop("retrieval_count", self._knn_k)
        retrieved_count = self._knn_k if requested_count is None else int(requested_count)
        if not (0 <= retrieved_count <= self._knn_k):
            raise ValueError(f"retrieval_count should be in [0, {self._knn_k}], got {retrieved_count}")
        # Choose retrieval strategy (can be overridden per-request)
        strategy_value = obs.pop("retrieval_strategy", self._retrieval_strategy)
        strategy = (
            strategy_value
            if isinstance(strategy_value, RetrievalStrategy)
            else RetrievalStrategy(strategy_value)
        )
        if strategy is RetrievalStrategy.NONE or retrieved_count == 0:
            return {**obs, **self._build_blank_retrieval()}

        # embed
        query_embedding = embed(obs["query_top_image"], self._dinov2)
        assert query_embedding.shape == (1, EMBED_DIM), f"{query_embedding.shape=}"
        # retrieve indices
        retrieved_indices, _ = self._select_retrieved_indices(strategy, query_embedding)
        assert retrieved_indices is not None, "retrieved_indices should not be None for non-NONE strategy"
        assert retrieved_indices.shape == (1, self._knn_k, 2), f"{retrieved_indices.shape=}"
        # collect retrieved info
        for ct in range(self._knn_k):
            if ct < retrieved_count:
                ep_idx, step_idx = retrieved_indices[0, ct]
                for key in ["state", "wrist_image", "top_image", "right_image"]:
                    more_obs[f"retrieved_{ct}_{key}"] = self._demos[ep_idx][key][step_idx]
                more_obs[f"retrieved_{ct}_actions"] = get_action_chunk_at_inference_time(self._demos[ep_idx]["actions"], step_idx, self._action_horizon)
                more_obs[f"retrieved_{ct}_prompt"] = self._demos[ep_idx]["prompt"].item()
            else:
                self._fill_blank_slot(more_obs, ct)
        # Compute exp_lamda_distances if use_action_interpolation
        if self._use_action_interpolation:
            first_embedding = self._demos[retrieved_indices[0, 0, 0]]["top_image_embeddings"][retrieved_indices[0, 0, 1]]
            distances = np.full((self._knn_k + 1,), self._max_dist, dtype=np.float32)
            distances[0] = 0.0
            for idx in range(1, retrieved_count):
                ep_idx, step_idx = retrieved_indices[0, idx]
                distances[idx] = np.linalg.norm(self._demos[ep_idx]["top_image_embeddings"][step_idx:step_idx+1] - first_embedding)
            distances[-1] = np.linalg.norm(query_embedding - first_embedding)
            distances = np.clip(distances, 0, self._max_dist) / self._max_dist
            print(f'distances: {distances}')
            more_obs["exp_lamda_distances"] = np.exp(-self._lamda * distances).reshape(-1, 1)
            print(f'exp_lamda_distances: {more_obs["exp_lamda_distances"]}')
        return {**obs, **more_obs}
    
    def save_obs(self, obs: dict, date: str, prefix: str):
        fol = f"obs_logs/{date}/{prefix}"
        os.makedirs(fol, exist_ok=True)
        current_datettime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save all images in one png
        big_top_image = []
        big_right_image = []
        big_wrist_image = []    
        for ct in range(self._knn_k):
            big_top_image.append(obs[f"retrieved_{ct}_top_image"])
            big_right_image.append(obs[f"retrieved_{ct}_right_image"])
            big_wrist_image.append(obs[f"retrieved_{ct}_wrist_image"])
        big_top_image.append(obs["query_top_image"])
        big_right_image.append(obs["query_right_image"])
        big_wrist_image.append(obs["query_wrist_image"])
        final_image = np.concatenate((np.concatenate(big_top_image, axis=1), np.concatenate(big_right_image, axis=1), np.concatenate(big_wrist_image, axis=1)), axis=0)
        Image.fromarray(final_image).save(f"{fol}/{current_datettime}.png")
        # save everything else to json
        with open(f"{fol}/{current_datettime}.json", "w") as f:
            everything_else = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items() if "image" not in k}
            everything_else["final_image_shape"] = list(final_image.shape)
            json.dump(everything_else, f, indent=4)
        return current_datettime

    def save_tokenized_inputs(self, inputs: dict, current_datettime: str, date: str, prefix: str):
        fol = f"obs_logs/{date}/{prefix}"
        os.makedirs(fol, exist_ok=True)
        every_tokenized_input = {}
        for k, v in inputs.items():
            if "token" in k:
                if v is None:
                    every_tokenized_input[k] = v
                    continue
                assert isinstance(v, np.ndarray) and v.dtype in [np.bool_, np.int64], f"{k=}, {v.dtype=}"
                every_tokenized_input[k] = v.astype(np.int32).tolist()
        with open(f"{fol}/{current_datettime}_token_inputs.json", "w") as f:
            json.dump(every_tokenized_input, f, indent=4)

    @override
    def infer(self, obs: dict, debug: bool = True) -> dict:  # type: ignore[misc]
        # Remove the prefix from the obs; get date; below for saving folder only
        prefix = obs.pop("prefix", "temp")
        date = datetime.now().strftime("%m%d")
        # Retrieval
        print()
        logger.info(f'retrieving...')
        obs = self.retrieve(obs)
        # for debugging, save everything in obs
        if debug:
            logger.info(f'saving obs...')
            current_datettime = self.save_obs(obs, date, prefix)
        # Make a copy since transformations may modify the inputs in place.
        logger.info(f'transforming...')
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # for debugging, save tokenized inputs
        if debug:
            logger.info(f'saving tokenized inputs...')
            self.save_tokenized_inputs(inputs, current_datettime, date, prefix)
        # Make a batch and convert to jax.Array.
        logger.info(f'batching...')
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        logger.info(f'sampling...')
        outputs = {
            "query_state": inputs["query_state"],
            "query_actions": self._sample_actions(sample_rng, _model.RiclObservation.from_dict(inputs, num_retrieved_observations=self._knn_k), **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        logger.info(f'unbatching...')
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        final_outputs = self._output_transform(outputs)
        print(f'final_outputs: {final_outputs}')
        return final_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def num_retrieved_observations(self) -> int:
        return self._knn_k


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
