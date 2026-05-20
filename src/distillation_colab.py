import sys
from typing import Optional, Sequence
from src.dataset_utils import create_dataloader_from_list, load_dataset_tasks, dataset_mapping
from transformers import CLIPModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
import torch
import torch.nn.functional as F
from src.load_evaluate_utils import load_model
from src.eff_merge import tsv_merge, ft_average, merge_clip_blocks
import copy


device = "cuda" if torch.cuda.is_available() else "cpu"


def dataset_for_distillation(task_name: str, bs: int = 32):

    ds_list = [dataset_mapping[task_name]["dataset_id"]]
    data_list, lab_list = load_dataset_tasks(ds_list, split="train")
    data_load, lab_map, inv_lab_map = create_dataloader_from_list(data_list, batch_size=bs, shuffle=True)
    return data_load


def preprocessing_batch(batch, processor):
    images = batch[0]
    processor_out = processor(images=images, return_tensors="pt", padding=True)
    return processor_out["pixel_values"]


def forward_until_layer_vision(model, pixel_values, layer_idx):

    vision = model.vision_model

    hidden_states = vision.embeddings(pixel_values)
    hidden_states = vision.pre_layrnorm(hidden_states)

    for i in range(layer_idx):
        hidden_states = vision.encoder.layers[i](hidden_states,
                                                attention_mask=None,
                                                causal_attention_mask=None,
                                                output_attentions=False,
                                                )
    return hidden_states


def forward_two_blocks_teacher(model, hidden_states, layer_idx):

    layers = model.vision_model.encoder.layers

    hidden_states = layers[layer_idx](hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False)
    hidden_states = layers[layer_idx + 1](hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False)

    return hidden_states


def distillation_step(input_data, teacher, student, layer_idx):

    # ---- Teacher step ----
    with torch.no_grad():
        x = forward_until_layer_vision(teacher, input_data, layer_idx)
        y_teacher = forward_two_blocks_teacher(teacher, x, layer_idx)

    # ---- Student step ----
    x_student = forward_until_layer_vision(student, input_data, layer_idx)
    y_student = student.vision_model.encoder.layers[layer_idx](x_student, attention_mask=None, causal_attention_mask=None, output_attentions=False)

    loss = F.mse_loss(y_student, y_teacher)

    return loss, y_student, y_teacher


def distill_clip_teacher(
    model_name: str,
    ft_weights_path: Optional[str] = None,
) -> CLIPModel:
    """
    Load the fine-tuned teacher model and set it to evaluation mode (frozen).
    
    The teacher model is used as reference for knowledge distillation.
    It must NOT be updated during training.
    
    Args:
        model_name: Base model identifier (e.g., "openai/clip-vit-base-patch32")
        ft_weights_path: Path to fine-tuned weights. If provided, will replace 
                        the vision encoder weights with these pre-trained weights.

    Returns:
        Teacher CLIP model in evaluation mode 
    """

    # Load model with optionally fine-tuned weights
    teacher_model, _ = load_model(model_name, ft_weights_path)
    
    # Move to device
    teacher_model = teacher_model.to(device)
    
    # Set to evaluation mode (disables dropout, batch norm, etc.)
    teacher_model.eval()
    
    # Freeze all parameters - no gradients needed for teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    return teacher_model


def distill_clip_student(
    teacher_model: CLIPModel,
    layer_idx_to_merge: Sequence[int],
    sv_portion: int = 2,
    whitening: bool = True,
    loop: bool = False
) -> CLIPModel:
    """
    Create a student model by merging specified layers of the teacher model.
    
    The student model is initialized by merging pairs of layers from the teacher 
    using the TSV merge method. The merged student model will have fewer layers 
    than the teacher, and will be trained to mimic the teacher's hidden representations after
    the merged layers.
    
    Args:
        teacher_model: The pre-trained teacher CLIP model to distill from
        layer_idx_to_merge: List of layer indices to merge (e.g., [9] to merge layers 9 and 10)
        sv_portion: Denominator for singular value selection in TSV merge (default: 2)
        whitening: Whether to apply whitening in TSV merge (default: True)
        loop: Whether to loop the merged model back into itself (default: False)

    Returns:
        Student CLIP model with merged layers ready for distillation training.
    """

    # Create a copy of the teacher model to modify as the student
    student_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    student_model.load_state_dict(teacher_model.state_dict())  # Start with teacher weights

    # Merge specified layers
    merge_clip_blocks(student_model, layer_idx_to_merge, tsv_merge, ft_average, 
                      k=sv_portion, whitening=whitening, loop=loop)
    
    # Freeze student model parameters not belonging to the merged layers for training
    for p in student_model.parameters():
        p.requires_grad = False  # Freeze student model parameters for training

    # Unfreeze only the merged layers for training
    numb_reduced_blocks = 0
    for idx in layer_idx_to_merge:
        student_model.vision_model.encoder.layers[idx-numb_reduced_blocks].requires_grad_(True)  # Unfreeze merged layers for training
        numb_reduced_blocks += 1 # Keep track of how many blocks have been merged to adjust indices for subsequent merges

    return student_model


def save_student_model(student_model: CLIPModel, save_path: str, single_block: bool = False):
    """
    Save the student model's state dictionary to the specified path.
    
    Args:
        student_model: The trained student CLIP model to save
        save_path: File path to save the model (e.g., "student_model.pth")
        single_block: If True, save only the trainable parameters (e.g. the single block)
    """
    if single_block:
        state_dict_to_save = {
            name: param.cpu()
            for name, param in student_model.named_parameters()
            if param.requires_grad
        }
    else:
        state_dict_to_save = student_model.state_dict()
        
    torch.save(state_dict_to_save, save_path)
