import torch
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from typing import List, Optional


def build_resized_clip_from_pretrained(num_layers: int):
    """
    Build a CLIP model with a resized vision transformer depth from pretrained weights.         
    Args:
        num_layers (int): The desired number of transformer layers in the vision encoder.               
    Returns:
        CLIPModel: A CLIP model with the specified number of vision transformer layers.
    """
    # Load original model + config
    base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    config = base.config

    # Modify config
    config.vision_config.num_hidden_layers = num_layers

    # Create empty model with resized depth
    new_model = CLIPModel(config)

    return new_model



def build_grouped_layer_map(state_dict_keys,
                            block_pair="all",
                            layer_type="all"):
    """
    Build a CLIP-compatible mapping: new_layer_name -> list(old_layers).

    If block_pair=(i,j), produces a new model where:
        - layers < i remain unchanged
        - layer i becomes the merge of i and j
        - layers > j shift down by 1 (j+1 → i+1)

    If block_pair="all", groups (0,1), (2,3), ..., (10,11).

    Args:
        state_dict_keys: iterable of CLIP state_dict keys
        block_pair: a tuple, or a list of tuples, or "all"
        layer_type: "fc1", "fc2", "self_attn", a list of them, or "all"
    """

    import re
    block_re = re.compile(r"encoder\.layers\.(\d+)\.(.*)")

    # Extract all blocks
    blocks = {}
    for k in state_dict_keys:
        m = block_re.match(k)
        if m:
            idx = int(m.group(1))
            suf = m.group(2)
            blocks.setdefault(idx, []).append((k, suf))

    max_block = max(blocks.keys())

    # Determine block pairs to merge
    if block_pair == "all":
        pairs = [(i, i + 1) for i in range(0, max_block, 2)]

    elif isinstance(block_pair, tuple):
        pairs = [block_pair]

    elif isinstance(block_pair, list):
        pairs = block_pair

    else:
        raise ValueError("block_pair must be a tuple, list of tuples, or 'all'.")

    # Extract start and end blocks of merges
    merge_start = {i for i, j in pairs}
    merge_end   = {j for i, j in pairs}

    # Map old block index → new block index (due to merging)
    mapping_block_index = {}
    shift = 0
    b = 0
    while b <= max_block:
        if b in merge_start:
            mapping_block_index[b] = b - shift
            partner = next(j for i, j in pairs if i == b)
            mapping_block_index[partner] = b - shift  # merged
            b = partner + 1
            shift += 1
        else:
            mapping_block_index[b] = b - shift
            b += 1

    # --------------------------------------------------------
    # NEW: normalize layer_type (string, list, tuple, or 'all')
    # --------------------------------------------------------
    if layer_type == "all":
        allowed_types = None
    elif isinstance(layer_type, (list, tuple, set)):
        allowed_types = set(layer_type)
    else:
        allowed_types = {layer_type}

    def is_merge_sublayer_fn(suffix):
        """Return True if this sublayer should be merged."""
        if allowed_types is None:
            return True
        return any(t in suffix for t in allowed_types)

    # Build final map
    new_map = {}

    for old_key in state_dict_keys:
        m = block_re.match(old_key)
        if not m:
            new_map[old_key] = [old_key]
            continue

        old_block = int(m.group(1))
        suffix = m.group(2)

        is_merge_sublayer = is_merge_sublayer_fn(suffix)
        new_block = mapping_block_index[old_block]

        # Block NOT merged
        if old_block not in merge_end and old_block not in merge_start:
            new_key = f"encoder.layers.{new_block}.{suffix}"
            new_map[new_key] = [old_key]
            continue

        # Block is start of merged pair
        if old_block in merge_start:
            partner = next(j for i, j in pairs if i == old_block)
            oldA = f"encoder.layers.{old_block}.{suffix}"
            oldB = f"encoder.layers.{partner}.{suffix}"

            new_key = f"encoder.layers.{new_block}.{suffix}"

            if not is_merge_sublayer:
                new_map[new_key] = [oldA]
            else:
                new_map[new_key] = [oldA, oldB]

            continue

        # Block is end of merged pair → absorbed, no output
        if old_block in merge_end:
            continue

    return new_map




def compare_vit_layers(model1: str, model2: str, verbose: bool = True):
    """
    Compare layer parameters between two saved ViT-B/32 models.

    Args:
        model1 (str): First model.
        path2 (str): Second model.
        verbose (bool): Whether to print details layer by layer.

    Returns:
        dict: Summary of which layers match and which differ.
    """
    # Load both models' state_dicts
    #model1 = AutoModel.from_pretrained(path1)
    #model2 = AutoModel.from_pretrained(path2)
    state_dict_1 = model1.state_dict()
    state_dict_2 = model2.state_dict()

    # Ensure both have the same keys
    keys_1 = set(state_dict_1.keys())
    keys_2 = set(state_dict_2.keys())

    if keys_1 != keys_2:
        missing_in_1 = keys_2 - keys_1
        missing_in_2 = keys_1 - keys_2
        print("⚠️ Mismatched layer keys detected!")
        if missing_in_1:
            print(f" - Missing in model 1: {missing_in_1}")
        if missing_in_2:
            print(f" - Missing in model 2: {missing_in_2}")

    shared_keys = sorted(keys_1 & keys_2)

    same_layers = []
    diff_layers = []

    for key in shared_keys:
        p1 = state_dict_1[key]
        p2 = state_dict_2[key]

        if torch.equal(p1, p2):
            same_layers.append(key)
        else:
            diff_layers.append(key)

        if verbose:
            status = "✅ SAME" if torch.equal(p1, p2) else "❌ DIFFERENT"
            print(f"{status}: {key}")

    summary = {
        "same_layers": same_layers,
        "different_layers": diff_layers,
        "num_same": len(same_layers),
        "num_diff": len(diff_layers),
        "total": len(shared_keys),
    }

    if verbose:
        print("\n=== SUMMARY ===")
        print(f"Total layers: {summary['total']}")
        print(f"Same: {summary['num_same']}")
        print(f"Different: {summary['num_diff']}")

    return summary



def copy_sd(new: str, old: str, verbose: bool = True):

    new_sd = new.state_dict()
    old_sd = old.state_dict()

    for key in new_sd.keys():
        new_sd[key] = old_sd[key]

    new.load_state_dict(new_sd)

    return new