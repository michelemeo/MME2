import sys
sys.path.append('/content/drive/MyDrive/MME_git_trial/MME')

from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from datasets import load_dataset, concatenate_datasets, Features, Value, ClassLabel
from torch.utils.data import DataLoader
from typing import List, Optional

from transformers import AutoModel, AutoTokenizer
import torch
from src.tsv_merge_utils import *
from src.model_utils import *
from src.load_evaluate_utils import *
from src.dataset_utils import *
from src.dict_utils import *

# EVALUATION
device = "cuda" if torch.cuda.is_available() else "cpu"
merging_model_path = "/content/drive/MyDrive/MME_git_trial/MME/models/MTmodel/TSV-Merge_check.pt"
tsv_m_model = AutoModel.from_pretrained(merging_model_path)
tsv_m_model.to(device)

model_base = "openai/clip-vit-base-patch32"
model_pt, processor = load_model(model_base)

# Evaluation part-1: manage datasets for all tasks
ds_list = [dataset_mapping['Cars']["dataset_id"], dataset_mapping['MNIST']["dataset_id"], dataset_mapping['DTD']["dataset_id"], dataset_mapping['EuroSAT']["dataset_id"],
           dataset_mapping['GTSRB']['dataset_id'], dataset_mapping['RESISC45']['dataset_id'], dataset_mapping['SUN397']['dataset_id'], dataset_mapping['SVHN']['dataset_id']]
data_list, lab_list = load_dataset_tasks(ds_list)
data_load, lab_map, inv_lab_map = create_dataloader_from_list(data_list)
temp_list = [dataset_mapping['Cars']["template"], dataset_mapping['MNIST']["template"], dataset_mapping['DTD']["template"], dataset_mapping['EuroSAT']["template"],
             dataset_mapping['GTSRB']['template'], dataset_mapping['RESISC45']['template'], dataset_mapping['SUN397']['template'], dataset_mapping['SVHN']['template']]

model_pt.to(device)
text_embeds, label_per_template, template_labels, total_num_templates = generate_text_embeddings(lab_list, temp_list, processor, model_pt, device=device)
del model_pt

# Evaluation part-2: generate text embeddings and evaluate the merged model
acc = evaluate_model(tsv_m_model, data_load, processor, text_embeds, label_per_template, lab_map, device=device)
print(f"Accuracy for TSV-M merging on 8 tasks: {acc}")

# Clean up Colab runtime
from google.colab import runtime
runtime.unassign()

