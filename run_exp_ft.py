import sys
import json
import ast
import argparse
from transformers import CLIPModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from src.eff_merge import *
from src.dataset_utils import *
from src.load_evaluate_utils import *

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Run fine-tuned CLIP merging experiments.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("task_name",   type=str, help="Task name (e.g. MNIST)")
parser.add_argument("sv_portion",  type=int, help="Singular value denominator k")
parser.add_argument("list_of_pairs", type=str, help="Groups of layer indices, e.g. [[9]] or [[9,1],[3,5]]")
parser.add_argument("--results_path", type=str, default=None, help="Optional path to save results")
parser.add_argument("--whitening",    type=lambda v: v.lower() in ('true', '1', 'yes'), default=True,
                    help="Apply whitening in TSV merge (default: True)")
parser.add_argument("--loop",         type=lambda v: v.lower() in ('true', '1', 'yes'), default=False,
                    help="Looped model after merge: model doesn't shrink (default: False)")

args = parser.parse_args()

task_name    = args.task_name
sv_portion   = args.sv_portion
results_path = args.results_path
whitening    = args.whitening
loop         = args.loop

raw_pairs = args.list_of_pairs
try:
    list_of_pairs = json.loads(raw_pairs)
except json.JSONDecodeError:
    try:
        list_of_pairs = ast.literal_eval(raw_pairs)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            "list_of_pairs must be JSON or Python literal, e.g. [[9,1],[3,5]]"
        ) from exc

def _pairs_from_integers(integers):
    """Convert a list of integers into pairs: [8, 3] -> [(8, 9), (3, 4)]."""
    values = set(integers)
    for value in values:
        if value + 1 in values:
            raise ValueError(
                "Each group must not contain two following integers (n and n+1)"
            )
    return [(value, value + 1) for value in integers]

def _normalize_pairs(obj):
    if not isinstance(obj, list):
        raise ValueError("list_of_pairs must be a list")

    normalized = []
    for group in obj:
        if not isinstance(group, (list, tuple)):
            raise ValueError("Each element in list_of_pairs must be a list/tuple of integers")
        if not group:
            raise ValueError("Each group must be non-empty")
        
        # Each group must contain integers that will be paired as (n, n+1)
        if all(isinstance(x, int) for x in group):
            pairs = _pairs_from_integers(group)
            normalized.append(pairs)
        else:
            raise ValueError("Each group must contain integers to be paired as (n, n+1)")

    return normalized

list_of_pairs = _normalize_pairs(list_of_pairs)
print(f"Normalized list_of_pairs: {list_of_pairs}")

# Load the pretrained and finetuned models

pt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
pt_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load datasets and create dataloaders for evaluation, compute text embeddings for evaluation

# Evaluation part-1: manage datasets for the specified task
ds_list = [dataset_mapping[task_name]["dataset_id"]]
data_list, lab_list = load_dataset_tasks(ds_list)
data_load, lab_map, inv_lab_map = create_dataloader_from_list(data_list)
temp_list = [dataset_mapping[task_name]["template"]]

# Evaluation part-2: generate text embeddings and evaluate the merged model
text_embeds, label_per_template, template_labels, total_num_templates = generate_text_embeddings(lab_list, temp_list, 
                                                                                                processor=pt_processor, 
                                                                                                model=pt_model,
                                                                                                device='cpu')

res_acc = {}


# Creating the merged model and merging the blocks according to the specified pairs
for pairs in list_of_pairs:

    for pair in pairs:
        idx1, idx2 = pair

        ft_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ft_vision_model = CLIPVisionModel.from_pretrained(dataset_mapping[task_name]['model_id'])
        ft_model.vision_model.load_state_dict(ft_vision_model.vision_model.state_dict())

        merge_clip_blocks(ft_model, layer_idx=idx1, op_matrix=tsv_merge, op_vector=ft_average, k=sv_portion, whitening=whitening, loop=loop)
        
        acc = evaluate_model(ft_model, data_load, 
                     pt_processor, text_embeds, label_per_template, lab_map, 
                     device='cpu')

        print(f"Accuracy on {task_name} reducing finetuned model for ({idx1}, {idx2}) blocks: {acc:.5f}")
        res_acc[idx1] = acc

if results_path is not None:
    with open(results_path, 'w') as f:
        json.dump(res_acc, f, indent=4)
        
