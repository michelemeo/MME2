from typing import List, Optional
from transformers import CLIPModel


def merge_two(model_name: str, model1: CLIPModel, model2: CLIPModel, alpha: float = 0.5, device: str = "cpu"):
    """
    Merge the vision encoder weights of two CLIP models using their weights linear combination.

    Args:
        model_name (str): The name or path of the pretrained CLIP model to initialize the merged model.
        model1 (CLIPModel): The first CLIP model, whose vision encoder weights will be merged.
        model2 (CLIPModel): The second CLIP model, whose vision encoder weights will be merged.
        alpha (float, optional): Weighting factor for combining model1 and model2's vision encoder parameters.
                                 Defaults to 0.5 for an equal combination.
        device (str, optional): Device to load the merged model onto (e.g., "cpu", "cuda").
                                Defaults to "cpu".
    Returns:
        CLIPModel: A new CLIP model with a merged vision encoder.
    """

    # Extract the vision encoder state dictionaries from both models
    m1 = model1.vision_model.state_dict()
    m2 = model2.vision_model.state_dict()

    # Dictionary to hold the merged vision encoder weights
    merged_state_dict = {}

    # Iterate over all layer parameters in model1's vision encoder
    for key in m1.keys():
        if key in m2:
            # Linear combination of the parameters
            merged_state_dict[key] = alpha * m1[key] + (1 - alpha) * m2[key]
        else:
            # Warn if a parameter exists in model1 but not in model2
            print(f"Key {key} not found in model2")

    # Initialize a fresh CLIP model from the base checkpoint
    merged_model = CLIPModel.from_pretrained(model_name).to(device)

    # Load the merged vision encoder weights into the model
    merged_model.vision_model.load_state_dict(merged_state_dict)

    return merged_model



def merge_isotropic(
    model_name: str,
    models: List[CLIPModel],
    alphas: Optional[List[float]] = None,
    device: str = "cpu"):
    """
    Merge the vision encoder weights of multiple CLIP models isotropically.

    Args:
        model_name (str): The name or path of the pretrained CLIP model to initialize the merged model.
        models (List[CLIPModel]): List of CLIP models to merge.
        alphas (List[float], optional): Weighting factors for each model.
                                        Must sum to 1.0. If None, uses equal weights.
        device (str, optional): Device to load the merged model onto (e.g., "cpu", "cuda").
                                Defaults to "cpu".
    Returns:
        CLIPModel: A new CLIP model with merged vision encoder weights.
    """
    num_models = len(models)
    if num_models < 2:
        raise ValueError("Need at least 2 models to perform merging.")

    # If no alphas are given, assign equal weights
    if alphas is None:
        alphas = [1.0 / num_models] * num_models
    else:
        if len(alphas) != num_models:
            raise ValueError("Length of alphas must match number of models.")
        # Normalize in case they don't sum exactly to 1
        total = sum(alphas)
        alphas = [a / total for a in alphas]

    # Extract all state_dicts
    state_dicts = [m.vision_model.state_dict() for m in models]

    # Initialize merged dictionary
    merged_state_dict = {}

    # Iterate over keys in the first model
    for key in state_dicts[0].keys():
        # Weighted sum across all models
        merged_state_dict[key] = sum(
            alphas[i] * state_dicts[i][key] for i in range(num_models)
        )

    # Load into a fresh CLIP model
    merged_model = CLIPModel.from_pretrained(model_name).to(device)
    merged_model.vision_model.load_state_dict(merged_state_dict)

    return merged_model