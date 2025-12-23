import csv
import dataclasses
import json
import logging
import os
import pathlib
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetPair:
    """One (index, eval) dataset combination to test."""

    name: str
    demos_dir: str
    eval_demos_dir: str


@dataclasses.dataclass
class Args:
    # Model checkpoint / config
    checkpoint_dir: str | None = None
    config: str = "pi0_fast_droid_ricl"

    # Single-pair fallback (used when datasets_json is not provided)
    demos_dir: str = "preprocessing/collected_demos/2025-03-14_move_the_idli_plate_to_the_right"
    eval_demos_dir: str = "preprocessing/collected_demos/2025-03-14-5demos_move_the_idli_plate_to_the_right"

    # Optional JSON file with a list of {"name", "demos_dir", "eval_demos_dir"} entries.
    datasets_json: Optional[str] = None

    retrieval_strategies: Tuple[str, ...] = ("knn",)
    retrieval_seed: Optional[int] = 0

    max_episodes: Optional[int] = 20
    max_steps_per_episode: Optional[int] = 100
    step_stride: int = 1

    csv_in: str | None = None  # If provided, load rows from an existing CSV instead of re-evaluating
    csv_out: Optional[str] = "data/droid/results/retrieval_multi.csv"
    plot_out: str | None = "data/droid/results/retrieval_multi.png"
    per_step_plot_out: str | None = "data/droid/results/retrieval_multi_per_step.png"
    max_plot_points: int = 8000  # downsample per-step plots for readability


def _iter_processed_demos(eval_demos_dir: str) -> Iterable[tuple[str, np.lib.npyio.NpzFile]]:
    """Yield (episode_id, npz) for each processed_demo.npz under eval_demos_dir."""
    for folder in sorted(os.listdir(eval_demos_dir)):
        fol_path = os.path.join(eval_demos_dir, folder)
        if not os.path.isdir(fol_path):
            continue
        npz_path = os.path.join(fol_path, "processed_demo.npz")
        if not os.path.exists(npz_path):
            logger.warning("Skipping %s (missing processed_demo.npz)", fol_path)
            continue
        try:
            data = np.load(npz_path, allow_pickle=False)
        except ValueError as exc:
            if "pickled data" not in str(exc):
                raise
            data = np.load(npz_path, allow_pickle=True)
        yield folder, data


def _prepare_obs(data: np.lib.npyio.NpzFile, step_idx: int, strategy: str) -> dict:
    prompt = data["prompt"]
    if hasattr(prompt, "item"):
        prompt = prompt.item()
    obs = {
        "query_top_image": data["top_image"][step_idx],
        "query_right_image": data["right_image"][step_idx],
        "query_wrist_image": data["wrist_image"][step_idx],
        "query_state": data["state"][step_idx],
        "query_prompt": prompt,
        "retrieval_strategy": strategy,
        "prefix": "eval",
    }
    return obs


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    mse = float(np.mean((pred - gt) ** 2))
    l1 = float(np.mean(np.abs(pred - gt)))
    return {"mse": mse, "l1": l1}


def _load_dataset_pairs(args: Args) -> List[DatasetPair]:
    if args.datasets_json:
        with open(args.datasets_json, "r") as f:
            raw = json.load(f)
        pairs = [DatasetPair(**item) for item in raw]
    else:
        default_name = pathlib.Path(args.eval_demos_dir).name or "default"
        pairs = [
            DatasetPair(
                name=default_name,
                demos_dir=args.demos_dir,
                eval_demos_dir=args.eval_demos_dir,
            )
        ]
    return pairs


def _eval_single_dataset(
    train_cfg: _config.TrainConfig,
    dataset: DatasetPair,
    args: Args,
) -> List[dict]:
    policy = _policy_config.create_trained_ricl_policy(
        train_cfg,
        checkpoint_dir=args.checkpoint_dir,
        demos_dir=dataset.demos_dir,
        retrieval_strategy="knn",
        retrieval_seed=args.retrieval_seed,
    )
    action_horizon = policy._action_horizon  # pylint: disable=protected-access

    rows: list[dict] = []
    for strategy in args.retrieval_strategies:
        logger.info("Evaluating dataset=%s retrieval_strategy=%s", dataset.name, strategy)
        for ep_idx, (ep_id, data) in enumerate(_iter_processed_demos(dataset.eval_demos_dir)):
            if args.max_episodes is not None and ep_idx >= args.max_episodes:
                break
            num_steps = data["actions"].shape[0]
            max_steps = min(num_steps, args.max_steps_per_episode or num_steps)
            for step_idx in range(0, max_steps, args.step_stride):
                obs = _prepare_obs(data, step_idx, strategy)
                pred = policy.infer(obs, debug=False)["query_actions"]
                gt = _policy.get_action_chunk_at_inference_time(data["actions"], step_idx, action_horizon)
                metrics = _compute_metrics(gt, pred)
                rows.append(
                    {
                        "dataset": dataset.name,
                        "strategy": strategy,
                        "episode": ep_id,
                        "step": step_idx,
                        "mse": metrics["mse"],
                        "l1": metrics["l1"],
                    }
                )
    return rows


def _load_rows_from_csv(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "strategy": row["strategy"],
                    "episode": row["episode"],
                    "step": int(row["step"]),
                    "mse": float(row["mse"]),
                    "l1": float(row["l1"]),
                }
            )
    logger.info("Loaded %d rows from %s", len(rows), csv_path)
    return rows


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    if args.csv_in:
        all_rows = _load_rows_from_csv(args.csv_in)
    else:
        if not args.checkpoint_dir:
            raise ValueError("checkpoint_dir is required unless --csv-in is provided")
        train_cfg = _config.get_config(args.config)
        dataset_pairs = _load_dataset_pairs(args)

        all_rows: list[dict] = []
        for dataset in dataset_pairs:
            all_rows.extend(_eval_single_dataset(train_cfg, dataset, args))

    # Aggregate by dataset and strategy
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for row in all_rows:
        key = (row["dataset"], row["strategy"])
        summaries.setdefault(key, {"mse_sum": 0.0, "l1_sum": 0.0, "count": 0})
        summaries[key]["mse_sum"] += row["mse"]
        summaries[key]["l1_sum"] += row["l1"]
        summaries[key]["count"] += 1

    print("\n=== DROID retrieval multi-dataset ablation (action prediction) ===")
    for (dataset_name, strat), agg in sorted(summaries.items()):
        count = agg["count"]
        mse = agg["mse_sum"] / count if count else float("nan")
        l1 = agg["l1_sum"] / count if count else float("nan")
        print(f"{dataset_name:>24} | {strat:>6}: mse={mse:.6f}, l1={l1:.6f}, n={count}")

    if args.csv_out and not args.csv_in:
        csv_path = pathlib.Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "strategy", "episode", "step", "mse", "l1"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote CSV to {csv_path}")

    if args.plot_out or args.per_step_plot_out:
        datasets = sorted({row["dataset"] for row in all_rows})
        strategies = sorted({row["strategy"] for row in all_rows})

        def _mean_metric(ds: str, strat: str, metric: str) -> float:
            agg = summaries.get((ds, strat))
            if not agg or agg["count"] == 0:
                return float("nan")
            return agg[f"{metric}_sum"] / agg["count"]

        if args.plot_out:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
            x = np.arange(len(datasets), dtype=np.float32)
            width = 0.8 / max(1, len(strategies))
            for idx, strat in enumerate(strategies):
                offsets = x + (idx - (len(strategies) - 1) / 2) * width
                l1_vals = [_mean_metric(ds, strat, "l1") for ds in datasets]
                mse_vals = [_mean_metric(ds, strat, "mse") for ds in datasets]
                axes[0].bar(offsets, l1_vals, width=width, label=strat)
                axes[1].bar(offsets, mse_vals, width=width, label=strat)

            for ax, title, ylabel in [
                (axes[0], "Mean L1 error by dataset", "L1 error (lower is better)"),
                (axes[1], "Mean MSE by dataset", "MSE (lower is better)"),
            ]:
                ax.set_xticks(x)
                ax.set_xticklabels(datasets, rotation=20)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3, axis="y")
            axes[1].set_xlabel("Dataset")
            axes[0].legend(title="strategy")

            plot_path = pathlib.Path(args.plot_out)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Wrote plot to {plot_path}")

        if args.per_step_plot_out:
            fig, (ax_l1, ax_mse) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
            combos = sorted({(row["dataset"], row["strategy"]) for row in all_rows})
            rng = np.random.default_rng(0)

            for ds, strat in combos:
                l1_vals = [row["l1"] for row in all_rows if row["dataset"] == ds and row["strategy"] == strat]
                mse_vals = [row["mse"] for row in all_rows if row["dataset"] == ds and row["strategy"] == strat]
                if not l1_vals:
                    continue
                if args.max_plot_points and len(l1_vals) > args.max_plot_points:
                    idx = rng.choice(len(l1_vals), size=args.max_plot_points, replace=False)
                    l1_vals = [l1_vals[i] for i in idx]
                    mse_vals = [mse_vals[i] for i in idx]
                x_idx = np.arange(len(l1_vals))
                label = f"{ds}/{strat}"
                ax_l1.plot(x_idx, l1_vals, ".", markersize=3, alpha=0.7, label=label)
                ax_mse.plot(x_idx, mse_vals, ".", markersize=3, alpha=0.7)

            ax_l1.set_ylabel("L1 error per step")
            ax_mse.set_ylabel("MSE per step")
            ax_mse.set_xlabel("Evaluation step index (ordered by CSV)")
            ax_l1.set_title("Per-step errors by dataset/strategy")
            ax_mse.grid(alpha=0.3)
            ax_l1.grid(alpha=0.3)
            ax_l1.legend(title="dataset/strategy")

            per_step_path = pathlib.Path(args.per_step_plot_out)
            per_step_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(per_step_path)
            plt.close(fig)
            print(f"Wrote per-step plot to {per_step_path}")


if __name__ == "__main__":
    tyro.cli(main)
