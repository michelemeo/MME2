import argparse
import torch
from transformers import CLIPModel
from src.eff_merge import *
from src.dataset_utils import *
from src.load_evaluate_utils import *

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Evaluate a saved CLIP model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("model_path", type=str, help="Path to the saved model state dict (.pth)")
parser.add_argument("task_name", type=str, help="Task name (e.g. MNIST)")
parser.add_argument("--dict_path", type=str, help="Path to dictionary to save results", default=None)

args = parser.parse_args()

model_path = args.model_path
task_name = args.task_name

print(f"Loading model from {model_path}...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Dynamically adjust the number of vision layers based on the saved state dict
state_dict = torch.load(model_path, map_location="cpu")
vision_layer_keys = [int(k.split('.')[3]) for k in state_dict.keys() if 'vision_model.encoder.layers.' in k]
if vision_layer_keys:
    num_layers = max(vision_layer_keys) + 1
    # Truncate layers if the saved model is smaller
    if num_layers < len(model.vision_model.encoder.layers):
        model.vision_model.encoder.layers = model.vision_model.encoder.layers[:num_layers]

model.load_state_dict(state_dict)
model.eval()

# Load the pretrained model to generate text embeddings, exactly as run_exp_ft.py does
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
    import json
    import os
    
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

