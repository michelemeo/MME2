import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json_dict(path: Path) -> dict[str, float]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	# Ensure values are numeric even if serialized as strings.
	return {str(k): float(v) for k, v in data.items()}


def normalize_task_name(task: str) -> str:
	alias_map = {
		"EusoSAT": "EuroSAT",
		"Eurosat": "EuroSAT",
		"EUROSAT": "EuroSAT",
	}
	return alias_map.get(task, task)


def find_existing_file(paths: list[Path]) -> Path:
	for candidate in paths:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(
		"None of the candidate files exist: "
		+ ", ".join(str(p) for p in paths)
	)


def resolve_single_path(results_dir: Path, task: str) -> Path:
	candidates = [
		results_dir / task / f"{task}_best_pair.json",
	]

	return find_existing_file(candidates)


def resolve_pair_path(results_dir: Path, task: str) -> Path:
	return find_existing_file([
		results_dir / task / "best_2_pairs" / f"{task}_best_2_pairs.json"
	])


def best_accuracy(path: Path) -> float:
	data = load_json_dict(path)
	if not data:
		raise ValueError(f"No entries found in JSON file: {path}")
	return max(data.values())


def best_key_and_accuracy(path: Path) -> tuple[str, float]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if not data:
		raise ValueError(f"No entries found in JSON file: {path}")

	best_key = max(data, key=lambda k: float(data[k]))
	return str(best_key), float(data[best_key])


def format_single_block_label(key: str) -> str:
	block_idx = int(str(key).strip())
	return f"[{block_idx},{block_idx + 1}]"


def format_pair_block_label(key: str) -> str:
	try:
		indices = ast.literal_eval(str(key))
		if isinstance(indices, (list, tuple)) and len(indices) == 2:
			left = int(indices[0])
			right = int(indices[1])
			# Use two lines to keep pair labels compact under each bar.
			return f"[{left},{left + 1}]\n[{right},{right + 1}]"
	except (ValueError, SyntaxError, TypeError):
		pass

	# Fallback: return raw key when format is unexpected.
	return str(key)


def collect_ratios(
	tasks: list[str],
	results_dir: Path,
	finetuned_path: Path,
) -> tuple[
	list[str],
	list[float],
	list[float],
	list[float],
	list[float],
	list[str],
	list[str],
]:
	finetuned_data = load_json_dict(finetuned_path)

	plotted_tasks: list[str] = []
	single_accuracies: list[float] = []
	pair_accuracies: list[float] = []
	single_ratios: list[float] = []
	pair_ratios: list[float] = []
	single_best_keys: list[str] = []
	pair_best_keys: list[str] = []

	for raw_task in tasks:
		task = normalize_task_name(raw_task)
		if task not in finetuned_data:
			raise KeyError(
				f"Task '{task}' not found in finetuned accuracy file: {finetuned_path}"
			)

		single_path = resolve_single_path(results_dir, task)
		pair_path = resolve_pair_path(results_dir, task)

		single_best_key, best_single = best_key_and_accuracy(single_path)
		pair_best_key, best_pair = best_key_and_accuracy(pair_path)
		ft_acc = finetuned_data[task]

		plotted_tasks.append(task)
		single_accuracies.append(best_single)
		pair_accuracies.append(best_pair)
		single_ratios.append(best_single / ft_acc)
		pair_ratios.append(best_pair / ft_acc)
		single_best_keys.append(single_best_key)
		pair_best_keys.append(pair_best_key)

	return (
		plotted_tasks,
		single_accuracies,
		pair_accuracies,
		single_ratios,
		pair_ratios,
		single_best_keys,
		pair_best_keys,
	)


def plot_ratios_bar(
	tasks: list[str],
	single_ratios: list[float],
	pair_ratios: list[float],
	single_best_keys: list[str],
	pair_best_keys: list[str],
	output_path: Path,
	show_plot: bool,
) -> None:
	x = np.arange(len(tasks))
	bar_width = 0.38

	plt.figure(figsize=(11, 6))
	bars_single = plt.bar(
		x - bar_width / 2,
		single_ratios,
		width=bar_width,
		label="Best 1-block reduction",
	)
	bars_pair = plt.bar(
		x + bar_width / 2,
		pair_ratios,
		width=bar_width,
		label="Best 2-block reduction",
	)

	plt.xticks(x, tasks)
	plt.ylabel("Acc/Acc_ft")
	plt.title("Accuracy of the best reduced configurations w.r.t. their finetuned configuration")
	plt.grid(axis="y", alpha=0.25)
	plt.legend()
	plt.ylim(0, max(single_ratios + pair_ratios) * 1.12)

	for bars in (bars_single, bars_pair):
		for bar in bars:
			height = bar.get_height()
			plt.text(
				bar.get_x() + bar.get_width() / 2,
				height + 0.01,
				f"{height:.3f}",
				ha="center",
				va="bottom",
				fontsize=9,
			)

	for xpos, single_key, pair_key in zip(x, single_best_keys, pair_best_keys):
		single_label = format_single_block_label(single_key)
		pair_label = format_pair_block_label(pair_key)

		# Place each configuration label under its own bar to avoid overlap.
		plt.text(
			xpos - bar_width / 2,
			-0.08,
			single_label,
			ha="center",
			va="top",
			fontsize=9,
			transform=plt.gca().get_xaxis_transform(),
		)
		plt.text(
			xpos + bar_width / 2,
			-0.08,
			pair_label,
			ha="center",
			va="top",
			fontsize=9,
			transform=plt.gca().get_xaxis_transform(),
		)

	plt.tight_layout()
	plt.subplots_adjust(bottom=0.28)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=200)

	if show_plot:
		plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Plot best merge accuracies (1 block vs 2 blocks) for multiple tasks, "
			"normalized by each task's fine-tuned model accuracy."
		)
	)
	parser.add_argument(
		"--tasks",
		nargs="+",
		default=["MNIST", "EuroSAT", "DTD", "SVHN", "GTSRB", "Cars", "RESISC45", "SUN397"],
		help="Tasks to include in the plot (e.g., MNIST EuroSAT DTD SVHN GTSRB Cars RESISC45 SUN397).",
	)
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=Path("results"),
		help="Root folder containing per-task result JSON files.",
	)
	parser.add_argument(
		"--finetuned-path",
		type=Path,
		default=Path("results/general/ft_acc.json"),
		help="JSON file containing fine-tuned accuracies by task.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("results/general/tasks_scaling_ratios.png"),
		help="Output image path for the plot.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display the plot interactively in addition to saving it.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	(
		tasks,
		single_accuracies,
		pair_accuracies,
		single_ratios,
		pair_ratios,
		single_best_keys,
		pair_best_keys,
	) = collect_ratios(
		tasks=args.tasks,
		results_dir=args.results_dir,
		finetuned_path=args.finetuned_path,
	)

	plot_ratios_bar(
		tasks=tasks,
		single_ratios=single_ratios,
		pair_ratios=pair_ratios,
		single_best_keys=single_best_keys,
		pair_best_keys=pair_best_keys,
		output_path=args.output,
		show_plot=args.show,
	)

	for task, s_acc, p_acc, s_ratio, p_ratio, s_key, p_key in zip(
		tasks,
		single_accuracies,
		pair_accuracies,
		single_ratios,
		pair_ratios,
		single_best_keys,
		pair_best_keys,
	):
		print(
			f"{task}: "
			f"single[{s_key}]={s_acc:.5f} ({s_ratio:.5f}), "
			f"pair[{p_key}]={p_acc:.5f} ({p_ratio:.5f})"
		)
	print("Saved plot to:", args.output)


if __name__ == "__main__":
	main()
