from tqdm import tqdm
import torch
import json
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from datasets import load_dataset, concatenate_datasets, Features, Value, ClassLabel
from torch.utils.data import DataLoader
import copy
import torch.nn as nn
from typing import List, Optional
from src.dataset_utils import *
from src.load_evaluate_utils import *
from src.merging import *
from src.tsv_merge_utils import *
from src.model_utils import *

layers_map_path = "/home/michele/Projects/MME2/dicts/layers_type_for_tsvmerge.json"

with open(layers_map_path, "r") as f:
    layers_map = json.load(f)



class SingleTaskEffMerger:

    def __init__(self, ft_model_name: str, layers_type: dict, device: Optional[str] = None):
        self.model_name = "openai/clip-vit-base-patch32"
        self.ft_model_name = ft_model_name
        self.layers_type = layers_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pt_model, self.processor = load_model(self.model_name)
        self.ft_model, _ = load_model(self.model_name, self.ft_model_name)


    def svd_dict(self, k: Optional[int] = 2, option: Optional[str] = "task matrix"):

        task_matrices_decomposition = {}
        pt_sd = self.pt_model.state_dict()
        ft_sd = self.ft_model.state_dict()

        for layer in self.layers_type:

            if self.layers_type[layer] == "svd":

                if option == "task matrix":
                    task_matrix = ft_sd[layer] - pt_sd[layer]
                elif option == "ft matrix":
                    task_matrix = ft_sd[layer]
                else:
                    raise ValueError("Invalid option. Choose 'task matrix' or 'ft matrix'.")

                U, S, Vh = torch.linalg.svd(task_matrix, full_matrices=False)

                top_r = int(len(S)/k)
                #print(f"Top {top_r} singular values selected for layer {layer}")

                U = U[:, :top_r]
                S = S[:top_r]
                Vh = Vh[:top_r, :]

                task_matrices_decomposition[layer] = {
                    "U": U,
                    "S": S,
                    "Vh": Vh
                }

        return task_matrices_decomposition
    

    def reduce_model(self, num_blocks: int, block_to_reduce: list = [], layer_type: str = "all"):

        reduced_model = build_resized_clip_from_pretrained(num_blocks)
        reduced_model = copy_sd(reduced_model, self.ft_model)
        layers_pair_map = build_grouped_layer_map(self.pt_model.vision_model.state_dict().keys(),
                                                  block_pair=block_to_reduce, layer_type=layer_type)

        return reduced_model, layers_pair_map
    

    def tsv_merge_efficiency(self, new_model, layers_pair_map: dict, svd_dec: dict,
                             alpha: float = 1.0, save_path: Optional[str] = None, 
                             option: Optional[str] = "task matrix"):
        
        pt_sd = self.pt_model.vision_model.state_dict()
        ft_sd = self.ft_model.vision_model.state_dict()
        new_sd = new_model.vision_model.state_dict()

        for vision_layer in new_sd.keys():

            num_layers = len(layers_pair_map[vision_layer])
            layer = 'vision_model.' + vision_layer

            if num_layers == 2:

                if self.layers_type[layer] == 'svd':

                    U_list, S_list, Vh_list = [], [], []

                    for l in layers_pair_map[vision_layer]:
                        l = "vision_model."+l
                        U_list.append(svd_dec[l]["U"])
                        S_list.append(svd_dec[l]["S"])
                        Vh_list.append(svd_dec[l]["Vh"])

                    U_cat = torch.cat(U_list, dim=1)  #col
                    S_cat = torch.cat(S_list, dim=0)  #vec
                    Vh_cat = torch.cat(Vh_list, dim=0)  #rows

                    # Safe SVD with fallback (WHITENING)
                    try:
                        Pu, _, QuT = torch.linalg.svd(U_cat, full_matrices=False)
                    except torch._C._LinAlgError:
                        print(f"⚠️ SVD failed for U in {layer}, fallback to QR.")
                        Pu, _ = torch.linalg.qr(U_cat)
                        QuT = Pu.T

                    try:
                        Pv, _, QvT = torch.linalg.svd(Vh_cat, full_matrices=False)
                    except torch._C._LinAlgError:
                        print(f"⚠️ SVD failed for Vh in {layer}, fallback to QR.")
                        Pv, _ = torch.linalg.qr(Vh_cat)
                        QvT = Pv.T


                    U_orth = Pu @ QuT
                    V_orth = Pv @ QvT

                    M = (U_orth * S_cat) @ V_orth

                    layer1 = layers_pair_map[vision_layer][0]
                    layer2 = layers_pair_map[vision_layer][1]

                    if option == "task matrix":
                        new_sd[vision_layer] = (pt_sd[layer1] + pt_sd[layer2])/2 + alpha * M
                    elif option == "ft matrix":
                        new_sd[vision_layer] = M
                    else:
                        raise ValueError("Invalid option. Choose 'task matrix' or 'ft matrix'.")

                if self.layers_type[layer] == 'ta':

                    if 'layer_norm' in vision_layer:
                        layer2 = layers_pair_map[vision_layer][1]
                        new_sd[vision_layer] = ft_sd[layer2]
                    else:
                        layer1 = layers_pair_map[vision_layer][0]
                        layer2 = layers_pair_map[vision_layer][1]
                        new_sd[vision_layer] = (ft_sd[layer1] + ft_sd[layer2]) / 2

            if num_layers == 1:

                layer = layers_pair_map[vision_layer][0]
    
                new_sd[vision_layer] = ft_sd[layer]   

        # Update all weights in the vision encoder
        new_model.vision_model.load_state_dict(new_sd)

        # Save to disk
        if save_path:
            new_model.save_pretrained(save_path)
            print(f"\n✅ Saved merged-reduced model to {save_path}")

        return new_model
    

    def remove_clip_vision_block(self, block_index: int):
        """
        Removes the vision transformer encoder block at `block_index`
        from a HuggingFace CLIPModel and returns the modified model.
        """
        model = copy.deepcopy(self.ft_model)  # Use the fine-tuned model as the base
        layers = model.vision_model.encoder.layers
        num_layers = len(layers)

        if not (0 <= block_index < num_layers):
            raise ValueError(f"block_index must be in [0, {num_layers - 1}]")

        # Create a new ModuleList without the selected block
        new_layers = nn.ModuleList(
            [layer for i, layer in enumerate(layers) if i != block_index]
        )

        # Replace layers in the model
        model.vision_model.encoder.layers = new_layers

        # Update config to keep everything consistent
        model.config.vision_config.num_hidden_layers = len(new_layers)

        return model
    

    def remove_clip_vision_blocks(self, block_indices):
        """
        Removes multiple vision transformer encoder blocks from a 
        HuggingFace CLIPModel and returns the modified model.

        Args:
            block_indices (Iterable[int]): Indices of blocks to remove.
        """

        model = copy.deepcopy(self.ft_model)
        layers = model.vision_model.encoder.layers
        num_layers = len(layers)

        # Convert to sorted unique set
        block_indices = sorted(set(block_indices))

        # Validate indices
        for idx in block_indices:
            if not (0 <= idx < num_layers):
                raise ValueError(f"All indices must be in [0, {num_layers - 1}]")

        # Keep only layers NOT in block_indices
        new_layers = nn.ModuleList(
            [layer for i, layer in enumerate(layers) if i not in block_indices]
        )

        if len(new_layers) == 0:
            raise ValueError("Cannot remove all vision transformer blocks.")

        # Replace layers
        model.vision_model.encoder.layers = new_layers

        # Update config
        model.config.vision_config.num_hidden_layers = len(new_layers)

        return model
                