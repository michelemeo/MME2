import argparse
import torch
import copy
import json
import os
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from src.eff_merge import *
from src.dataset_utils import *
from src.load_evaluate_utils import *

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Evaluate a distilled CLIP model with merged blocks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("task_name", type=str, help="Task name (e.g. MNIST)")
parser.add_argument("--merged_blocks", type=int, nargs="+", required=True, help="List of block indices where merging starts (e.g. 3 9 for [3,4] and [9,10])")
parser.add_argument("--model_paths", type=str, nargs="+", required=True, help="Paths to the dicts where weights of merged block are saved, corresponding to merged_blocks")
parser.add_argument("--dict_path", type=str, help="Path to dictionary to save results", default=None)

args = parser.parse_args()

task_name = args.task_name
merged_blocks = args.merged_blocks
model_paths = args.model_paths

if len(merged_blocks) != len(model_paths):
    raise ValueError("The number of merged_blocks must match the number of model_paths provided.")

print("Loading the finetuned version of the CLIP model from huggingface...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
ft_vision_model = CLIPVisionModel.from_pretrained(dataset_mapping[task_name]['model_id'])
model.vision_model.load_state_dict(ft_vision_model.vision_model.state_dict())

import torch.nn as nn

new_layers = nn.ModuleList()
original_layers = model.vision_model.encoder.layers

skip_next = False

for i, orig_layer in enumerate(original_layers):
    if skip_next:
        skip_next = False
        continue
        
    if i in merged_blocks:
        idx = merged_blocks.index(i)
        path = model_paths[idx]
        
        print(f"Loading weights for merged block {i} from {path}...")
        state_dict = torch.load(path, map_location="cpu")
        
        # Check if the dictionary has the whole model or just the block by checking common full-model prefixes
        is_full_model = any(k.startswith("vision_model") or k.startswith("text_model") for k in state_dict.keys())
        
        new_layer = copy.deepcopy(orig_layer)
        
        if is_full_model:
            # Extract just weights of the specific layer
            # Distilled layer might be saved at the same index 'i' in the distilled model checkpoint
            prefix = f"vision_model.encoder.layers.{i}."
            layer_state_dict = {
                k.replace(prefix, "") : v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if not layer_state_dict:
                raise ValueError(f"Could not find weights with prefix {prefix} in the checkpoint at {path}")
            new_layer.load_state_dict(layer_state_dict)
        else:
            # The dictionary just has the block.
            # In case it has some prefix like "vision_model.encoder.layers.X." anyway, we can strip it.
            prefix = f"vision_model.encoder.layers.{i}."
            if any(k.startswith(prefix) for k in state_dict.keys()):
                layer_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
                new_layer.load_state_dict(layer_state_dict)
            else:
                # No standard prefix found for this layer index, assume it's just the exact layer variables
                new_layer.load_state_dict(state_dict)
                
        new_layers.append(new_layer)
        # Assuming merging is over two following blocks (e.g., [3,4]), so we skip the next layer
        skip_next = True
    else:
        new_layers.append(orig_layer)

# Replace the original layers with the new layers (which replaces and drops the merged ones)
model.vision_model.encoder.layers = new_layers
model.eval()

# Load the pretrained model to generate text embeddings, exactly as run_exp_ft.py / evaluate_model.py does
pt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
pt_model.eval()
pt_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f"Evaluating on task: {task_name}...")
# Evaluation part-1: manage datasets for the specified task
ds_list = [dataset_mapping[task_name]["dataset_id"]]
data_list, lab_list = load_dataset_tasks(ds_list)
data_load, lab_map, inv_lab_map = create_dataloader_from_list(data_list)
temp_list = [dataset_mapping[task_name]["template"]]

# Evaluation part-2: generate text embeddings and evaluate the mapped model
text_embeds, label_per_template, template_labels, total_num_templates = generate_text_embeddings(
    lab_list, temp_list, 
    processor=pt_processor, 
    model=pt_model,
    device='cpu'
)

acc = evaluate_model(
    model, data_load,
    pt_processor, text_embeds, label_per_template, lab_map,
    device='cpu'
)

print(f"Accuracy on {task_name}: {acc:.5f}")

if args.dict_path:
    dict_file = args.dict_path
    
    if os.path.exists(dict_file):
        with open(dict_file, 'r') as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    save = True
    if task_name in results_dict:
        answer = input(f"Result for {task_name} already exists ({results_dict[task_name]}). Overwrite? (y/n): ")
        if answer.lower() != 'y':
            save = False
            
    if save:
        results_dict[task_name] = round(acc, 5)
        with open(dict_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Saved result to {dict_file}")
    else:
        print("Result not saved.")
