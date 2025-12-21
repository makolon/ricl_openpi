import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10  # Number of rollouts per task (use smaller number for quick ablations)

    #################################################################################################################
    # In-context retrieval ablation
    #################################################################################################################
    retrieval_strategies: tuple[str, ...] = ("knn", "random", "none")  # Tried in order, per strategy summary printed

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    use_ricl_client: bool = False  # Set True if the server is scripts/serve_policy_ricl.py (RICL policy)
    csv_out: str | None = None  # Optional: write per-task/per-strategy results to CSV
    plot_out: str | None = None  # Optional: save a bar plot of overall success rates


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _eval_single_strategy(args: Args, retrieval_strategy: str) -> dict[str, float | int | str]:
    # Set random seed
    np.random.seed(args.seed)
    logging.info("Evaluating retrieval_strategy=%s", retrieval_strategy)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    video_dir = pathlib.Path(args.video_out_path) / retrieval_strategy
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_summaries = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"tasks ({retrieval_strategy})"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            logging.info("\nTask: %s", task_description)

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info("Starting episode %d...", task_episodes + 1)
            done = False
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    right_img = img  # LIBERO only exposes agentview + wrist; reuse top image for "right"

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        base_state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        if args.use_ricl_client:
                            element = {
                                "query_top_image": img,
                                "query_right_image": right_img,
                                "query_wrist_image": wrist_img,
                                "query_state": base_state,
                                "query_prompt": str(task_description),
                                "retrieval_strategy": retrieval_strategy,
                            }
                            action_chunk = client.infer(element)["query_actions"]
                        else:
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": base_state,
                                "prompt": str(task_description),
                                "retrieval_strategy": retrieval_strategy,
                            }
                            action_chunk = client.infer(element)["actions"]

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as exc:  # pylint: disable=broad-except
                    logging.error("Caught exception: %s", exc)
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                video_dir / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info("# successes: %s (%.1f%%)", total_successes, total_successes / total_episodes * 100)

        task_rate = float(task_successes) / float(task_episodes)
        task_summaries.append(
            {
                "strategy": retrieval_strategy,
                "task_id": task_id,
                "task": task_description,
                "successes": task_successes,
                "episodes": task_episodes,
                "success_rate": task_rate,
            }
        )
        # Log per-task results
        logging.info("Current task success rate: %s", task_rate)
        logging.info("Current total success rate: %s", float(total_successes) / float(total_episodes))

    final_rate = float(total_successes) / float(total_episodes)
    logging.info("Total success rate (%s): %s", retrieval_strategy, final_rate)
    logging.info("Total episodes: %s", total_episodes)
    return {
        "strategy": retrieval_strategy,
        "total_episodes": total_episodes,
        "successes": total_successes,
        "success_rate": final_rate,
        "tasks": task_summaries,
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    summaries = []
    for strategy in args.retrieval_strategies:
        summaries.append(_eval_single_strategy(args, strategy))

    print("\n=== Retrieval ablation summary ===")
    for summary in summaries:
        print(
            f"{summary['strategy']:>8}: "
            f"{summary['successes']}/{summary['total_episodes']} "
            f"({summary['success_rate'] * 100:.1f}%)"
        )

    # Optional CSV export
    if args.csv_out:
        import csv

        csv_path = pathlib.Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["strategy", "task_id", "task", "successes", "episodes", "success_rate"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                for task_row in summary["tasks"]:
                    writer.writerow(task_row)
        print(f"Wrote CSV to {csv_path}")

    # Optional plot export (overall success rates per strategy)
    if args.plot_out:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - only triggered if matplotlib missing
            print(f"Skipping plot; matplotlib not available ({exc})")
        else:
            strategies = [s["strategy"] for s in summaries]
            rates = [s["success_rate"] * 100 for s in summaries]
            plt.figure(figsize=(6, 4))
            plt.bar(strategies, rates, color="#4C72B0")
            plt.ylabel("Success rate (%)")
            plt.xlabel("Retrieval strategy")
            plt.ylim(0, 100)
            plt.title("LIBERO success by retrieval strategy")
            plt.grid(axis="y", alpha=0.3)
            plot_path = pathlib.Path(args.plot_out)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    tyro.cli(main)
