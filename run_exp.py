import sys
import json
import ast
from src.single_task import *

# Get command line arguments
if len(sys.argv) < 5:
    print("Usage: python3 run_exp.py <task_name> <sv_portion> <list_of_pairs> <results_path>")
    print("<list_of_pairs> can be JSON or Python literal, e.g. [[(9,10)],[(1,2),(3,4)]]")
    sys.exit(1)

# Parse command line arguments
task_name = sys.argv[1]
sv_portion = int(sys.argv[2])
raw_pairs = sys.argv[3]
try:
    list_of_pairs = json.loads(raw_pairs)
except json.JSONDecodeError:
    try:
        list_of_pairs = ast.literal_eval(raw_pairs)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            "list_of_pairs must be JSON or Python literal, e.g. [[(9,10)],[(1,2),(3,4)]]"
        ) from exc
results_path = sys.argv[4]  # Path for the results JSON file

def _is_pair(obj):
    return (
        isinstance(obj, (list, tuple))
        and len(obj) == 2
        and all(isinstance(x, int) for x in obj)
    )

def _group_integers_into_pairs(integers):
    """Convert a list of integers into pairs: [8, 9, 3, 4] -> [(8, 9), (3, 4)]"""
    if len(integers) % 2 != 0:
        raise ValueError(f"List of integers must have even length to form pairs, got {len(integers)} elements")
    pairs = []
    for i in range(0, len(integers), 2):
        pairs.append((integers[i], integers[i+1]))
    return pairs

def _normalize_pairs(obj):
    if not isinstance(obj, list):
        raise ValueError("list_of_pairs must be a list")

    normalized = []
    for group in obj:
        if not isinstance(group, (list, tuple)):
            raise ValueError("Each element in list_of_pairs must be a list/tuple of integers")
        if not group:
            raise ValueError("Each group must be non-empty")
        
        # Each group must contain integers that will be paired
        if all(isinstance(x, int) for x in group):
            # Convert list of integers to pairs
            pairs = _group_integers_into_pairs(group)
            normalized.append(pairs)
        else:
            raise ValueError("Each group must contain integers to be paired")

    return normalized

list_of_pairs = _normalize_pairs(list_of_pairs)

# Update the model_reduction with the task_name
model_reduction = SingleTaskEffMerger(ft_model_name=dataset_mapping[task_name]['model_id'],
                                      layers_type=layers_map)

# Use sv_portion in the SVD dictionary
model_svd_dec = model_reduction.svd_dict(k=sv_portion) 

# Evaluation part-1: manage datasets for the specified task
ds_list = [dataset_mapping[task_name]["dataset_id"]]
data_list, lab_list = load_dataset_tasks(ds_list)
data_load, lab_map, inv_lab_map = create_dataloader_from_list(data_list)
temp_list = [dataset_mapping[task_name]["template"]]

# Evaluation part-2: generate text embeddings and evaluate the merged model
text_embeds, label_per_template, template_labels, total_num_templates = generate_text_embeddings(lab_list, temp_list, 
                                                                                                processor=model_reduction.processor, 
                                                                                                model=model_reduction.pt_model,
                                                                                                device=model_reduction.device)

res_acc = {}

# Add the evaluation logic for each pair of blocks to merge
for pairs_to_reduce in list_of_pairs:

    depth = 12-len(pairs_to_reduce)

    print(f"\n🔄 Reducing model to {depth} numb.of blocks by merging blocks: {pairs_to_reduce}")

    reduced_model, reduced_layers_pair_map = model_reduction.reduce_model(num_blocks=depth, block_to_reduce=pairs_to_reduce)

    efficient_model = model_reduction.tsv_merge_efficiency(new_model=reduced_model,
                                                            layers_pair_map=reduced_layers_pair_map,
                                                            svd_dec=model_svd_dec,
                                                            alpha=1.0)

    # Assuming evaluate_model is defined and works with the provided data
    acc = evaluate_model(efficient_model, data_load, 
                     model_reduction.processor, text_embeds, label_per_template, lab_map, 
                     device=model_reduction.device)
    res_acc[str(pairs_to_reduce)] = acc
    print(f"Accuracy on {task_name} reducing finetuned model for {pairs_to_reduce} blocks: {acc:.5f}")

# Save results to a JSON file
with open(results_path, 'w') as f:
    json.dump(res_acc, f, indent=4)