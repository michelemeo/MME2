import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json_dict(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    parsed: dict[int, float] = {}
    for key, value in data.items():
        parsed[int(str(key).strip())] = float(value)
    return parsed


def resolve_paths(results_dir: Path, task: str) -> tuple[Path, Path]:
    loop_path = results_dir / task / f"{task}_loop.json"
    best_pair_path = results_dir / task / f"{task}_best_pair.json"

    if not loop_path.exists():
        raise FileNotFoundError(f"Missing loop file for {task}: {loop_path}")
    if not best_pair_path.exists():
        raise FileNotFoundError(f"Missing best_pair file for {task}: {best_pair_path}")

    return loop_path, best_pair_path


def list_available_tasks(results_dir: Path) -> list[str]:
    tasks: list[str] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name == "general":
            continue

        task = child.name
        loop_path = child / f"{task}_loop.json"
        best_pair_path = child / f"{task}_best_pair.json"
        if loop_path.exists() and best_pair_path.exists():
            tasks.append(task)

    if not tasks:
        raise FileNotFoundError(
            f"No task folders with both *_loop.json and *_best_pair.json found in: {results_dir}"
        )
    return tasks


def sorted_common_blocks(loop_data: dict[int, float], best_pair_data: dict[int, float]) -> list[int]:
    common = sorted(set(loop_data).intersection(best_pair_data))
    if not common:
        raise ValueError("No common block indices found between loop and best_pair files.")
    return common


def plot_task_on_axis(
    ax: plt.Axes,
    task: str,
    loop_data: dict[int, float],
    best_pair_data: dict[int, float],
    y_upper: float,
) -> None:
    blocks = sorted_common_blocks(loop_data, best_pair_data)
    x = np.arange(len(blocks))

    loop_values = [loop_data[b] for b in blocks]
    best_pair_values = [best_pair_data[b] for b in blocks]

    # Draw best_pair first with transparency, then loop in front for emphasis.
    ax.bar(
        x,
        best_pair_values,
        width=0.72,
        color="#ff7f0e",
        alpha=0.35,
        label="Reduced model",
        zorder=1,
    )
    ax.bar(
        x,
        loop_values,
        width=0.50,
        color="#1f77b4",
        alpha=0.95,
        label="Loop",
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Merged block index")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{task}: Loop vs Reduced model")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend()
    ax.tick_params(axis="y", labelleft=True)

    ax.set_ylim(0, y_upper)


def plot_tasks_grid(
    task_data: list[tuple[str, dict[int, float], dict[int, float]]],
    output_path: Path,
    n_cols: int,
) -> None:
    global_max = 0.0
    for _, loop_data, best_pair_data in task_data:
        task_max = max(max(loop_data.values()), max(best_pair_data.values()))
        global_max = max(global_max, task_max)

    y_upper = min(1.05, global_max * 1.12 if global_max > 0 else 1.0)

    n_tasks = len(task_data)
    n_cols = max(1, min(n_cols, n_tasks))
    n_rows = math.ceil(n_tasks / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.2 * n_rows),
        sharey=True,
    )
    axes_flat = np.array(axes, dtype=object).reshape(-1)

    for ax, (task, loop_data, best_pair_data) in zip(axes_flat, task_data):
        plot_task_on_axis(
            ax=ax,
            task=task,
            loop_data=loop_data,
            best_pair_data=best_pair_data,
            y_upper=y_upper,
        )

    # Hide any remaining subplot axes if the grid has unused slots.
    for ax in axes_flat[len(task_data):]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-task bar plots comparing loop accuracies against "
            "best_pair accuracies."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Tasks to compare. If omitted, all tasks found under --results-dir "
            "with both *_loop.json and *_best_pair.json are used."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root folder containing task result JSON files.",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=4,
        help="Number of subplot columns in the grid (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/general/loop_vs_best_pair/loop_vs_reduced_grid.png"),
        help="Output image path for the task comparison grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = args.tasks if args.tasks is not None else list_available_tasks(args.results_dir)
    task_data: list[tuple[str, dict[int, float], dict[int, float]]] = []

    for task in tasks:
        loop_path, best_pair_path = resolve_paths(args.results_dir, task)
        loop_data = load_json_dict(loop_path)
        best_pair_data = load_json_dict(best_pair_path)
        task_data.append((task, loop_data, best_pair_data))

    plot_tasks_grid(task_data=task_data, output_path=args.output, n_cols=args.n_cols)
    print(f"Saved grid comparison to: {args.output}")


if __name__ == "__main__":
    main()
