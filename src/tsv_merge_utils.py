import torch
from transformers import CLIPModel, AutoModel, AutoTokenizer
from typing import List, Optional
from src.load_evaluate_utils import load_model


def compute_and_save_svd(model_pt_name, model_ft_list, layers_list, 
                         save_path=None, top_r=None, device="cpu", normalize=None):
    """
    Compute and optionally save the SVD decomposition of task matrices
    for multiple fine-tuned CLIP/ViT models hosted on Hugging Face,
    relative to a single pretrained base model.

    A *task matrix* for a given layer is defined as:
        ΔW = W_finetuned - W_pretrained

    Args:
        model_pt_name (str):
            Path of the pretrained (base) model on Hugging Face.
            This model serves as the reference for all comparisons.

        model_ft_list (list[str]):
            List of model paths for the fine-tuned models.

        layers_list (list[str]):
            List of layer names (keys in the model.state_dict())
            for which SVD of the task matrices should be computed.

        save_path (str, optional):
            File path where the resulting dictionary is saved with torch.save().
            If None, results are only returned in memory.

        top_r (int, optional):
            If provided, truncate SVD results to the top-r singular components.

        device (str):
            Device for computation. "cpu" is safer for large models
            because SVD on GPU can easily run out of memory.

        normalize (str, optional):
            If "frobenius", normalize each task matrix by its Frobenius norm
            before computing the SVD.

    Returns:
        dict: Nested dictionary with structure:
              {
                  "finetuned_model_name": {
                      "layer_name": {"U": U, "S": S, "Vh": Vh},
                      ...
                  },
                  ...
              }
    """

    task_matrices_decomposition = {}

    model_pt, _ = load_model(model_pt_name)
    sd_pt = model_pt.state_dict()

    for model_ft_name in model_ft_list:
      model_ft, _ = load_model(model_pt_name, model_ft_name)

      task_matrices_decomposition[model_ft_name] = {}
      sd_ft = model_ft.state_dict()

      for layer in layers_list:
        task_matrix = sd_ft[layer] - sd_pt[layer]

        if layers_list[layer] == "svd":

          if normalize == "frobenius":
            task_matrix = task_matrix / torch.norm(task_matrix, p='fro')

          U, S, Vh = torch.linalg.svd(task_matrix, full_matrices=False)

          if top_r is not None:
            U = U[:, :top_r]
            S = S[:top_r]
            Vh = Vh[:top_r, :]

          task_matrices_decomposition[model_ft_name][layer] = {
              "U": U,
              "S": S,
              "Vh": Vh
          }

        else:
          task_matrices_decomposition[model_ft_name][layer] = task_matrix

      del model_ft

    if save_path:
      torch.save(task_matrices_decomposition, save_path)

    return task_matrices_decomposition



def tsv_merge(svd_path, layer_list, save_path=None):

  merged_layers = {}

  svd_dict = torch.load(svd_path)
  task_list = list(svd_dict.keys())
  T = len(task_list)

  for layer in layer_list:

    if layer_list[layer] == "svd":

      U_list, S_list, Vh_list = [], [], []

      for task in task_list:
        total_r = svd_dict[task][layer]["S"].shape[0]  # to take numb. of sing. values
        k = total_r // T

        U_list.append(svd_dict[task][layer]["U"][:, :k])
        S_list.append(svd_dict[task][layer]["S"][:k])
        Vh_list.append(svd_dict[task][layer]["Vh"][:k, :])

      U_cat = torch.cat(U_list, dim=1)   # columns
      S_cat = torch.cat(S_list, dim=0)   # vector
      Vh_cat = torch.cat(Vh_list, dim=0) # rows

      # Safe SVD with fallback
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

      merged_layers[layer] = M

    else:

      merged_layers[layer] = torch.zeros_like(svd_dict[task_list[0]][layer])

      for task in task_list:
        merged_layers[layer] += svd_dict[task][layer]/T

  # Save to disk
  if save_path:
    torch.save(merged_layers, save_path)
    print(f"\n✅ Saved merged layers to {save_path}")

  return merged_layers



def inject_tsv_merge_weights(
    base_model_name: str,
    merged_weights_path: str,
    alpha: float = 1.0,
    save_path: str = None,
):
    """
    Inject TSV-Merged weights into a pretrained CLIP model with weighted addition.

    Args:
        base_model_name (str): Hugging Face model name, e.g. "openai/clip-vit-base-patch32".
        merged_weights_path (str): Path to TSV-merged weights (torch .pt or .bin file).
        alpha (float): Scaling factor for merged weights in W_new = W_pre + alpha * W_merged.
        save_path (str, optional): Directory to save the updated model.

    Returns:
        model (CLIPModel): Updated CLIP model with merged weights injected.
    """

    # 1. Load pretrained CLIP
    model = CLIPModel.from_pretrained(base_model_name)

    sd_pt = model.state_dict()

    # 2. Load merged TSV weights
    merged_sd = torch.load(merged_weights_path)
    print(f"Loaded merged weights with {len(merged_sd)} layers.")

    # 3. Iterate on the merged layers and consider it to build the merged model
    for layer in merged_sd.keys():
        sd_pt[layer] = sd_pt[layer] + alpha * merged_sd[layer]

    # 4. Update pretrained state dict with new state dict
    model.load_state_dict(sd_pt)

    # 5. Optionally save
    if save_path:
        model.save_pretrained(save_path)
        print(f"💾 Saved updated model to {save_path}")

    return model



def tsv_merge_pairs(svd_path, layer_map, layer_pair_map,
                    new_model, pt_model, alpha=1.0,
                    task_list=None, save_path=None):

  pt_sd = pt_model.vision_model.state_dict()
  new_sd = new_model.vision_model.state_dict()

  svd_dict = torch.load(svd_path)

  if task_list is None:
    task_list = list(svd_dict.keys())

  T = len(task_list)

  for vision_layer in new_sd.keys():

    layer = "vision_model."+vision_layer
    num_lay = len(layer_pair_map[vision_layer])

    if layer_map[layer] == "svd":

      U_list, S_list, Vh_list = [], [], []

      for task in task_list:
        total_r = svd_dict[task][layer]["S"].shape[0]
        k = total_r // (num_lay*T)

        for l in layer_pair_map[vision_layer]:
          l = "vision_model."+l
          U_list.append(svd_dict[task][l]["U"][:, :k])
          S_list.append(svd_dict[task][l]["S"][:k])
          Vh_list.append(svd_dict[task][l]["Vh"][:k, :])

      U_cat = torch.cat(U_list, dim=1)  #col
      S_cat = torch.cat(S_list, dim=0)  #vec
      Vh_cat = torch.cat(Vh_list, dim=0)  #rows

      # Safe SVD with fallback
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

      if num_lay == 2:
        # Take pretrained configurations of layers in the pair
        layer1 = layer_pair_map[vision_layer][0]
        layer2 = layer_pair_map[vision_layer][1]

        # Compute new layer weights
        new_sd[vision_layer] = (pt_sd[layer1] + pt_sd[layer2])/2 + alpha * M

      elif num_lay == 1:
        new_sd[vision_layer] = pt_sd[vision_layer] + alpha * M

    elif layer_map[layer] == "ta":

      M = torch.zeros_like(svd_dict[task_list[0]][layer])

      for task in task_list:
        for l in layer_pair_map[vision_layer]:
          l = "vision_model."+l
          M += svd_dict[task][l]/(num_lay*T)  # like there are 2 tasks (1 per consecutive layer)

      if num_lay == 2:
        # Take pretrained configurations of layers in the pair
        layer1 = layer_pair_map[vision_layer][0]
        layer2 = layer_pair_map[vision_layer][1]

        # Compute new layer weights
        new_sd[vision_layer] = (pt_sd[layer1] + pt_sd[layer2])/2 + alpha * M

      elif num_lay == 1:
        new_sd[vision_layer] = pt_sd[vision_layer] + alpha * M

  # Update all weights in the vision encoder
  new_model.vision_model.load_state_dict(new_sd)

  # Save to disk
  if save_path:
    new_model.save_pretrained(save_path)
    print(f"\n✅ Saved merged/reduced model to {save_path}")

  del svd_dict

  return new_model



def tsv_merge_pairs_correction(svd_path, layer_map, layer_pair_map,
                    new_model, pt_model, alpha=1.0,
                    task_list=None, save_path=None):

  pt_sd = pt_model.vision_model.state_dict()
  new_sd = new_model.vision_model.state_dict()

  svd_dict = torch.load(svd_path)

  if task_list is None:
    task_list = list(svd_dict.keys())

  T = len(task_list)

  for vision_layer in new_sd.keys():

    layer = "vision_model."+vision_layer
    num_lay = len(layer_pair_map[vision_layer])

    if layer_map[layer] == "svd":

      U_list, S_list, Vh_list = [], [], []

      for task in task_list:
        total_r = svd_dict[task][layer]["S"].shape[0]
        k = total_r // (num_lay*T)

        for l in layer_pair_map[vision_layer]:
          l = "vision_model."+l
          U_list.append(svd_dict[task][l]["U"][:, :k])
          S_list.append(svd_dict[task][l]["S"][:k])
          Vh_list.append(svd_dict[task][l]["Vh"][:k, :])

      U_cat = torch.cat(U_list, dim=1)  #col
      S_cat = torch.cat(S_list, dim=0)  #vec
      Vh_cat = torch.cat(Vh_list, dim=0)  #rows

      # Safe SVD with fallback
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

      if num_lay == 2:
        # Take pretrained configurations of layers in the pair
        layer1 = layer_pair_map[vision_layer][0]
        layer2 = layer_pair_map[vision_layer][1]

        # Compute new layer weights
        new_sd[vision_layer] = (pt_sd[layer1] + pt_sd[layer2])/2 + alpha * M

      elif num_lay == 1:
        layer1 = layer_pair_map[vision_layer][0]
        new_sd[vision_layer] = pt_sd[layer1] + alpha * M

    elif layer_map[layer] == "ta":

      M = torch.zeros_like(svd_dict[task_list[0]][layer])

      for task in task_list:
        for l in layer_pair_map[vision_layer]:
          l = "vision_model."+l
          M += svd_dict[task][l]/(num_lay*T)  # like there are 2 tasks (1 per consecutive layer)

      if num_lay == 2:
        # Take pretrained configurations of layers in the pair
        layer1 = layer_pair_map[vision_layer][0]
        layer2 = layer_pair_map[vision_layer][1]

        # Compute new layer weights
        new_sd[vision_layer] = (pt_sd[layer1] + pt_sd[layer2])/2 + alpha * M

      elif num_lay == 1:
        layer1 = layer_pair_map[vision_layer][0]
        new_sd[vision_layer] = pt_sd[layer1] + alpha * M

  # Update all weights in the vision encoder
  new_model.vision_model.load_state_dict(new_sd)

  # Save to disk
  if save_path:
    new_model.save_pretrained(save_path)
    print(f"\n✅ Saved merged/reduced model to {save_path}")

  del svd_dict

  return new_model