import sys
from transformers import CLIPModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
import torch



def tsv_merge(w1_pt, w1_ft, w2_pt, w2_ft, 
              k=2, whitening=True):
    
    task_matrix = w1_ft - w1_pt
    task_matrix_2 = w2_ft - w2_pt

    U1, S1, Vh1 = torch.linalg.svd(task_matrix, full_matrices=False)
    U2, S2, Vh2 = torch.linalg.svd(task_matrix_2, full_matrices=False)

    top_r = int(len(S1)/k)

    U1, U2 = U1[:, :top_r], U2[:, :top_r]
    S1, S2 = S1[:top_r], S2[:top_r]
    Vh1, Vh2 = Vh1[:top_r, :], Vh2[:top_r, :]

    U_list = [U1, U2]
    S_list = [S1, S2]
    Vh_list = [Vh1, Vh2]

    U_cat = torch.cat(U_list, dim=1)
    S_cat = torch.cat(S_list, dim=0)
    Vh_cat = torch.cat(Vh_list, dim=0)

    if whitening:
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

    else:
        
        M = (U_cat * S_cat) @ Vh_cat
    
    w_new = (w1_pt+w2_pt)/2 + M

    return w_new


def ft_average(w1_ft, w2_ft):
    return (w1_ft + w2_ft) / 2


def merge_blocks(block1_ft, block2_ft, merged_block, 
                 block1_pt, block2_pt,
                 merge_matrix, merge_vector,
                 k=2, whitening=True):

    for (name, w_merged) in merged_block.named_parameters():
        
        w1_ft = dict(block1_ft.named_parameters())[name]
        w2_ft = dict(block2_ft.named_parameters())[name]
        w1_pt = dict(block1_pt.named_parameters())[name]
        w2_pt = dict(block2_pt.named_parameters())[name]

        if w1_ft.dim() == 2:
            w_merged.data.copy_(merge_matrix(w1_pt, w1_ft, w2_pt, w2_ft, k=k, whitening=whitening))

        elif w1_ft.dim() == 1 and 'layer_norm' not in name:
            w_merged.data.copy_(merge_vector(w1_ft, w2_ft))

        elif w1_ft.dim() == 1 and 'layer_norm' in name:
            w_merged.data.copy_(w2_ft)


def merge_clip_blocks(ft_model, layer_idx, op_matrix, op_vector, k=2, whitening=True, loop=False):

    pt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    pt_encoder = pt_model.vision_model.encoder
    pt_layers = pt_encoder.layers

    ft_encoder = ft_model.vision_model.encoder
    ft_layers = ft_encoder.layers
    config = ft_model.config.vision_config

    layer_idx_to_delete = []

    for idx in layer_idx:

        pt_layer1 = pt_layers[idx]
        pt_layer2 = pt_layers[idx + 1]

        ft_layer1 = ft_layers[idx]
        ft_layer2 = ft_layers[idx + 1]

        merged_layer = CLIPEncoderLayer(config)

        merge_blocks(ft_layer1, ft_layer2, merged_layer, pt_layer1, pt_layer2, 
                    merge_matrix=tsv_merge, merge_vector=ft_average,
                    k=k, whitening=whitening)
        
        ft_layers[idx] = merged_layer
        layer_idx_to_delete.append(idx + 1)

        if loop==True:
            # Copy merged layer into the next slot to maintain model depth
            ft_layers[idx + 1] = merged_layer

    if loop==False:
        # Replace and delete: model shrinks by one block per pair
        for idx in sorted(layer_idx_to_delete, reverse=True):
            del ft_layers[idx]
        ft_model.config.vision_config.num_hidden_layers = len(ft_layers)


    return ft_model