import sys
import json
import ast
import argparse
from transformers import CLIPModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
import torch.nn.functional as F
from src.eff_merge import *
from src.dataset_utils import *
from src.load_evaluate_utils import *
from src.distillation import *

task = 'MNIST'
batch_size = 128
merging_idx = 9
sv_portion = 2

data_loader = dataset_for_distillation(task, bs=batch_size)

teacher_model = distill_clip_teacher(
    model_name="openai/clip-vit-base-patch32",
    ft_weights_path=dataset_mapping[task]['model_id']
)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

student_model = distill_clip_student(
    teacher_model=teacher_model,
    layer_idx_to_merge=[merging_idx],
    sv_portion=sv_portion)

merged_block = student_model.vision_model.encoder.layers[merging_idx]

optimizer = torch.optim.AdamW(
    merged_block.parameters(),
    lr=1e-4,
    weight_decay=1e-2
)


loss_registry = []
rel_error_registry = []

for epoch in range(5):
    loss_during_epoch = []
    rel_error_during_epoch = []

    for batch in data_loader:
        #print(batch)
        #processor_out = processor(images=batch[0], return_tensors="pt", padding=True)
        #print(processor_out['pixel_values'].shape)
        '''
        i += 1
        if i >= 1:
            teacher_outputs = forward_until_layer_vision(teacher_model, processor_out["pixel_values"], 10)
            student_outputs = forward_until_layer_vision(student_model, processor_out["pixel_values"], 9)
            loss = F.mse_loss(student_outputs, teacher_outputs)
            print(f"Distillation loss at iteration {i}: {loss.item():.5f}")
            break
        '''

        input_data = preprocessing_batch(batch, processor)

        # Loss computation
        loss, y_student, y_teacher = distillation_step(input_data, teacher_model, student_model, layer_idx=merging_idx)

        # Relative error computation
        rel_error = torch.norm(y_student - y_teacher) / torch.norm(y_teacher)

        loss_during_epoch.append(loss.item())
        rel_error_during_epoch.append(rel_error.item())

        print(f"Loss: {loss.item():.6f} | RelErr: {rel_error.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_registry.append(loss_during_epoch)
    rel_error_registry.append(rel_error_during_epoch)
    print(f"Epoch {epoch + 1} - Average Loss: {sum(loss_during_epoch) / len(loss_during_epoch):.6f}")
    print(f"Epoch {epoch + 1} - Average Relative Error: {sum(rel_error_during_epoch) / len(rel_error_during_epoch):.6f}")

save_student_model(student_model, "/home/michele/Projects/MME2/dicts/distillation_models/MNIST_layer9_student_bs128.pth")
 
# A line to save the loss and relative error registries for later analysis, if needed
with open("/home/michele/Projects/MME2/dicts/distillation_models/MNIST_layer9_training_log.json", 'w') as f:
    json.dump({
        "loss": loss_registry,
        "relative_error": rel_error_registry
    }, f, indent=4)
