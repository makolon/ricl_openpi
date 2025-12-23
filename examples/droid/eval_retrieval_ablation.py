import dataclasses
import logging
import os
import pathlib
from typing import Iterable

import csv
import numpy as np
import matplotlib.pyplot as plt
import tyro
from matplotlib import ticker

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
    retrieval_counts: tuple[int, ...] | None = None

    max_episodes: int | None = 20
    max_steps_per_episode: int | None = 100
    step_stride: int = 1  # sample every N steps

    csv_out: str | None = "data/droid/results/retrieval_ablation.csv"
    plot_out: str | None = "data/droid/results/retrieval_ablation.png"
    per_step_plot_out: str | None = "data/droid/results/retrieval_ablation_per_step.png"
    max_scatter_points: int = 8000  # cap scatter plot points for readability


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


def _prepare_obs(data: np.lib.npyio.NpzFile, step_idx: int, strategy: str, retrieval_count: int) -> dict:
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
        "retrieval_count": retrieval_count,
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
    max_retrievals = policy.num_retrieved_observations
    if args.retrieval_counts is None:
        retrieval_counts = tuple(range(max_retrievals + 1))
    else:
        retrieval_counts = tuple(int(ct) for ct in args.retrieval_counts)
    invalid_counts = [ct for ct in retrieval_counts if ct < 0 or ct > max_retrievals]
    if invalid_counts:
        raise ValueError(f"retrieval_counts must be between 0 and {max_retrievals}, got {invalid_counts}")
    retrieval_counts = tuple(dict.fromkeys(retrieval_counts))

    all_rows: list[dict] = []
    per_step_data: dict[tuple[str, int], dict[str, list]] = {}
    for strategy in args.retrieval_strategies:
        counts_for_strategy = retrieval_counts if strategy != _policy.RetrievalStrategy.NONE.value else (0,)
        for retrieval_count in counts_for_strategy:
            combo_key = (strategy, retrieval_count)
            per_step_data.setdefault(combo_key, {"mse": [], "l1": [], "gt_flat": [], "pred_flat": []})
            logger.info("Evaluating retrieval_strategy=%s retrieval_count=%s", strategy, retrieval_count)
            for ep_idx, (ep_id, data) in enumerate(_iter_processed_demos(args.eval_demos_dir)):
                if args.max_episodes is not None and ep_idx >= args.max_episodes:
                    break
                num_steps = data["actions"].shape[0]
                max_steps = min(num_steps, args.max_steps_per_episode or num_steps)
                for step_idx in range(0, max_steps, args.step_stride):
                    obs = _prepare_obs(data, step_idx, strategy, retrieval_count)
                    pred = policy.infer(obs, debug=False)["query_actions"]
                    gt = _policy.get_action_chunk_at_inference_time(data["actions"], step_idx, action_horizon)
                    metrics = _compute_metrics(gt, pred)
                    per_step_data[combo_key]["mse"].append(metrics["mse"])
                    per_step_data[combo_key]["l1"].append(metrics["l1"])
                    per_step_data[combo_key]["gt_flat"].append(np.asarray(gt, dtype=np.float32).ravel())
                    per_step_data[combo_key]["pred_flat"].append(np.asarray(pred, dtype=np.float32).ravel())
                    all_rows.append(
                        {
                            "strategy": strategy,
                            "retrieval_count": retrieval_count,
                            "episode": ep_id,
                            "step": step_idx,
                            "mse": metrics["mse"],
                            "l1": metrics["l1"],
                        }
                    )

    # Aggregate
    summaries: dict[tuple[str, int], dict[str, float]] = {}
    for row in all_rows:
        key = (row["strategy"], row["retrieval_count"])
        summaries.setdefault(key, {"mse_sum": 0.0, "l1_sum": 0.0, "count": 0})
        summaries[key]["mse_sum"] += row["mse"]
        summaries[key]["l1_sum"] += row["l1"]
        summaries[key]["count"] += 1

    print("\n=== DROID retrieval ablation (action prediction) ===")
    for (strat, retrieval_count), agg in sorted(summaries.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        count = agg["count"]
        mse = agg["mse_sum"] / count if count else float("nan")
        l1 = agg["l1_sum"] / count if count else float("nan")
        print(f"{strat:>6} k={retrieval_count}: mse={mse:.4f}, l1={l1:.4f}, n={count}")

    # Optional CSV
    if args.csv_out:
        csv_path = pathlib.Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["strategy", "retrieval_count", "episode", "step", "mse", "l1"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote CSV to {csv_path}")

    # Optional plots
    if args.plot_out or args.per_step_plot_out:
        if args.plot_out:
            combos = list(summaries.keys())
            labels = [f"{strat}-k{count}" for strat, count in combos]
            mse_vals = [summaries[c]["mse_sum"] / summaries[c]["count"] for c in combos]
            max_mse = max(mse_vals) if mse_vals else 0.0

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, mse_vals, color="#4C72B0")
            ax.set_ylabel("MSE (lower is better)")
            ax.set_xlabel("Retrieval strategy / in-context count")
            ax.set_title("DROID action prediction error by retrieval setting")
            ax.grid(axis="y", alpha=0.3)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))  # show small values in scientific notation
            if max_mse > 0:
                ax.set_ylim(0, max_mse * 1.15)

            plot_path = pathlib.Path(args.plot_out)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Wrote plot to {plot_path}")

        if args.per_step_plot_out:
            fig, (ax_err, ax_scatter) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [1, 1]})

            # Per-step error trace
            for (strat, retrieval_count), data in per_step_data.items():
                if not data["l1"]:
                    continue
                x_idx = np.arange(len(data["l1"]))
                label = f"{strat}-k{retrieval_count}"
                ax_err.plot(x_idx, data["l1"], ".", markersize=3, label=label, alpha=0.7)

            ax_err.set_ylabel("L1 error per step")
            ax_err.set_xlabel("Evaluation step index (ordered)")
            ax_err.set_title("Per-step action error across retrieval settings")
            ax_err.grid(alpha=0.3)
            ax_err.legend(title="strategy/count")

            # Scatter of predictions vs ground truth
            rng = np.random.default_rng(0)
            for (strat, retrieval_count), data in per_step_data.items():
                if not data["gt_flat"]:
                    continue
                gt_concat = np.concatenate(data["gt_flat"])
                pred_concat = np.concatenate(data["pred_flat"])
                if gt_concat.size > args.max_scatter_points:
                    idx = rng.choice(gt_concat.size, size=args.max_scatter_points, replace=False)
                    gt_concat = gt_concat[idx]
                    pred_concat = pred_concat[idx]
                label = f"{strat}-k{retrieval_count}"
                ax_scatter.scatter(gt_concat, pred_concat, s=6, alpha=0.45, label=label)

            if ax_scatter.collections:
                all_vals = np.concatenate(
                    [np.concatenate(d["gt_flat"]) for d in per_step_data.values() if d["gt_flat"]]
                    + [np.concatenate(d["pred_flat"]) for d in per_step_data.values() if d["pred_flat"]]
                )
                diag_min, diag_max = float(np.min(all_vals)), float(np.max(all_vals))
                ax_scatter.plot([diag_min, diag_max], [diag_min, diag_max], "k--", linewidth=1, label="y=x")

            ax_scatter.set_xlabel("Ground truth action values")
            ax_scatter.set_ylabel("Predicted action values")
            ax_scatter.set_title("Action predictions vs ground truth (flattened)")
            ax_scatter.grid(alpha=0.3)
            ax_scatter.legend(title="strategy/count")

            fig.tight_layout()
            per_step_path = pathlib.Path(args.per_step_plot_out)
            per_step_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(per_step_path)
            plt.close(fig)
            print(f"Wrote per-step plot to {per_step_path}")


if __name__ == "__main__":
    tyro.cli(main)
