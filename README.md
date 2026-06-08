## MME Model Merging for Efficiency 
This repository contains the code and experiments developed for my **Master’s thesis in Data Science (MSc)** on **Model Merging for Efficiency**.

## Project Overview
Recent studies have demonstrated that intermediate layers in LLMs can be reordered with a negligible effect on the accuracy. Furthermore, consecutive Transformer layers of Llama 2 7B and Llama 3.2 3B models were grouped in pairs and evaluated in parallel, basically halving the depth of the mid-part of the model and reducing inference time.

- For the reference paper: [https://arxiv.org/abs/2502.02790](https://arxiv.org/abs/2502.02790)

At the same time Task Singular Vectors (TSVs) emerged as an efficient tool to merge ViT models fine-tuned on different tasks, both compressing model layers and reducing interference among tasks. This result is possible because TSVs contain information about task interference at layer level, exploiting its matrix structure, while previous Task Arithmetic (TA) approach treat the entire model as a flat vector.

- For the reference paper: [https://arxiv.org/abs/2412.00081](https://arxiv.org/abs/2412.00081)
- For the TSVs repository: [https://github.com/AntoAndGar/task_singular_vectors](https://github.com/AntoAndGar/task_singular_vectors)
- For the MASS paper, that combines TSVs with a data-free router strategy to select best task-specific block based on the input: [https://arxiv.org/abs/2504.05342](https://arxiv.org/abs/2504.05342)
- For the MASS repository: [https://github.com/crisostomi/mass](https://github.com/crisostomi/mass)

## Goal
The aim of the project is to push efficiency in ViT models (especially their mid-part) and probe their depth, applying TSVs not per-layer but per-grouped-consecutive-layers. The starting idea is to adapt the strategy proposed above for pairs of consecutive layers, merging them with TSVs instead of parallelizing them.

## Repository structure
The repository is split between source logic, runnable scripts, setup dictionaries, and experiment results.

* **Root directory (`/`)**:
  * **Main Experiments**: `run_exp.py`, `run_exp_ft.py`
  * **Combinations Evaluation**: `run_best2pairs.py`, `run_best3pairs.py` (with corresponding Jupyter notebooks `.ipynb` to exploit colab gpu).
  * **Distillation**: `run_distillation.py`
  * **Analyses and evaluation**: `evaluate_model.py`, `evaluate_distilled_model.py`.
  * **Reporting and plots**: `generate_accuracy_table.py` and various `plot_*.py` scripts, I have some of them just in the local repo (e.g., `plot_training_logs.py`, `plot_merge_scaling.py`, `plot_accuracy_comparison.py`).
  * **Environment configurations**: `requirements.txt`, `virt_env.yaml`

* **`src/`**: Core utilities and function definitions.
  * *Data & Metrics:* `dataset_utils.py`, `load_evaluate_utils.py`, `dict_utils.py`
  * *Merging Logic:* `merging.py`, `eff_merge.py`, `tsv_merge_utils.py`
  * *Finetuning Logic for best merging configurations:* `distillation.py`, `distillation_colab.py`
  * *Models:* `model_utils.py`, `single_task.py`

## How to Run

### a. Setup the environment
You can setup the virtual environment using Conda (`virt_env.yaml`) or alternatively load from `requirements.txt`:

```bash
# Using conda
conda env create -f virt_env.yaml
conda activate <your_env_name>

# OR using pip natively
pip install -r requirements.txt
```

### b. Typical Workflows

**1. Main Experiments**
To begin executing fundamental layer mergers and evaluations:
```bash
python run_exp.py
# Or for evaluating post fine-tuning:
python run_exp_ft.py
```

**2. Optimizing 2-Pair and 3-Pair Structures**
Once initial layer tests are done, you can test consecutive merged pairs:
```bash
python run_best2pairs.py
python run_best3pairs.py
```
*(Optionally you can use `run_best2pairs.ipynb` and `run_best3pairs.ipynb` on colab to use their gpu, I did this best configuration search through these)*

**3. Running Finetuning on best configurations**
Run teacher-student layer distillations across tasks:
```bash
python run_distillation.py
# To evaluate distilled models against regular tasks:
python evaluate_distilled_model.py
```

**4. Generating Results and Visualizations**
To compile the raw resulting JSONs into markdown tables or scaling plots (I have following scripts in the local repo):
```bash
python generate_accuracy_table.py
python plot_accuracy_comparison.py
python plot_merge_scaling.py
python plot_loop_vs_best_pair.py
```
