import dataclasses
import json
import logging
import os
import pathlib
from typing import Iterable

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    # Model checkpoint / config
    checkpoint_dir: str
    config: str = "pi0_fast_droid_ricl"
    demos_dir: str = "ricl_droid_preprocessing/collected_demos_training"  # retrieval corpus for the policy

    # Evaluation dataset (processed_demo.npz folders)
    eval_demos_dir: str = "ricl_droid_preprocessing/collected_demos_eval"

    retrieval_strategies: tuple[str, ...] = ("knn", "random", "none")
    retrieval_seed: int | None = 0

    max_episodes: int | None = 20
    max_steps_per_episode: int | None = 100
    step_stride: int = 1  # sample every N steps

    csv_out: str | None = "data/droid/results/retrieval_ablation.csv"
    plot_out: str | None = "data/droid/results/retrieval_ablation.png"


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
    """Build query_* observation dict for the policy."""
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
        "prefix": "eval",  # avoid per-step folder explosion in recorder paths
    }
    return obs


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((pred - gt) ** 2))
    l1 = float(np.mean(np.abs(pred - gt)))
    return {"mse": mse, "l1": l1}


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    train_cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_ricl_policy(
        train_cfg,
        checkpoint_dir=args.checkpoint_dir,
        demos_dir=args.demos_dir,
        retrieval_strategy="knn",
        retrieval_seed=args.retrieval_seed,
    )
    action_horizon = policy._action_horizon  # pylint: disable=protected-access

    all_rows: list[dict] = []
    for strategy in args.retrieval_strategies:
        logger.info("Evaluating retrieval_strategy=%s", strategy)
        for ep_idx, (ep_id, data) in enumerate(_iter_processed_demos(args.eval_demos_dir)):
            if args.max_episodes is not None and ep_idx >= args.max_episodes:
                break
            num_steps = data["actions"].shape[0]
            max_steps = min(num_steps, args.max_steps_per_episode or num_steps)
            for step_idx in range(0, max_steps, args.step_stride):
                obs = _prepare_obs(data, step_idx, strategy)
                pred = policy.infer(obs, debug=False)["query_actions"]
                gt = _policy.get_action_chunk_at_inference_time(data["actions"], step_idx, action_horizon)
                metrics = _compute_metrics(gt, pred)
                all_rows.append(
                    {
                        "strategy": strategy,
                        "episode": ep_id,
                        "step": step_idx,
                        "mse": metrics["mse"],
                        "l1": metrics["l1"],
                    }
                )

    # Aggregate
    summaries: dict[str, dict[str, float]] = {}
    for row in all_rows:
        strat = row["strategy"]
        summaries.setdefault(strat, {"mse_sum": 0.0, "l1_sum": 0.0, "count": 0})
        summaries[strat]["mse_sum"] += row["mse"]
        summaries[strat]["l1_sum"] += row["l1"]
        summaries[strat]["count"] += 1

    print("\n=== DROID retrieval ablation (action prediction) ===")
    for strat, agg in summaries.items():
        count = agg["count"]
        mse = agg["mse_sum"] / count if count else float("nan")
        l1 = agg["l1_sum"] / count if count else float("nan")
        print(f"{strat:>6}: mse={mse:.4f}, l1={l1:.4f}, n={count}")

    # Optional CSV
    if args.csv_out:
        import csv

        csv_path = pathlib.Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["strategy", "episode", "step", "mse", "l1"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote CSV to {csv_path}")

    # Optional plot
    if args.plot_out:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            print(f"Skipping plot; matplotlib not available ({exc})")
        else:
            strategies = list(summaries.keys())
            mse_vals = [summaries[s]["mse_sum"] / summaries[s]["count"] for s in strategies]
            plt.figure(figsize=(6, 4))
            plt.bar(strategies, mse_vals, color="#4C72B0")
            plt.ylabel("MSE (lower is better)")
            plt.xlabel("Retrieval strategy")
            plt.title("DROID action prediction error by retrieval strategy")
            plt.grid(axis="y", alpha=0.3)
            plot_path = pathlib.Path(args.plot_out)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    tyro.cli(main)
