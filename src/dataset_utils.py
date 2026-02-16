# List of supported datasets
datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

# Prompt templates for each dataset
cars_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"i love my {c}!",
    lambda c: f"a photo of my dirty {c}.",
    lambda c: f"a photo of my clean {c}.",
    lambda c: f"a photo of my new {c}.",
    lambda c: f"a photo of my old {c}.",
]

dtd_template = [
    lambda c: f"a photo of a {c} texture.",
    lambda c: f"a photo of a {c} pattern.",
    lambda c: f"a photo of a {c} thing.",
    lambda c: f"a photo of a {c} object.",
    lambda c: f"a photo of the {c} texture.",
    lambda c: f"a photo of the {c} pattern.",
    lambda c: f"a photo of the {c} thing.",
    lambda c: f"a photo of the {c} object.",
]

eurosat_template = [
    lambda c: f"a centered satellite photo of {c}.",
    lambda c: f"a centered satellite photo of a {c}.",
    lambda c: f"a centered satellite photo of the {c}.",
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

resisc45_template = [
    lambda c: f"satellite imagery of {c}.",
    lambda c: f"aerial imagery of {c}.",
    lambda c: f"satellite photo of {c}.",
    lambda c: f"aerial photo of {c}.",
    lambda c: f"satellite view of {c}.",
    lambda c: f"aerial view of {c}.",
    lambda c: f"satellite imagery of a {c}.",
    lambda c: f"aerial imagery of a {c}.",
    lambda c: f"satellite photo of a {c}.",
    lambda c: f"aerial photo of a {c}.",
    lambda c: f"satellite view of a {c}.",
    lambda c: f"aerial view of a {c}.",
    lambda c: f"satellite imagery of the {c}.",
    lambda c: f"aerial imagery of the {c}.",
    lambda c: f"satellite photo of the {c}.",
    lambda c: f"aerial photo of the {c}.",
    lambda c: f"satellite view of the {c}.",
    lambda c: f"aerial view of the {c}.",
]

sun397_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

# Mapping dictionary for datasets
dataset_mapping = {
    "Cars": {
        "dataset_id": "tanganke/stanford_cars",
        "model_id": "tanganke/clip-vit-base-patch32_stanford-cars",
        "template": cars_template
    },
    "DTD": {
        "dataset_id": "tanganke/dtd",
        "model_id": "tanganke/clip-vit-base-patch32_dtd",
        "template": dtd_template
    },
    "EuroSAT": {
        "dataset_id": "tanganke/eurosat",
        "model_id": "tanganke/clip-vit-base-patch32_eurosat",
        "template": eurosat_template
    },
    "GTSRB": {
        "dataset_id": "tanganke/gtsrb",
        "model_id": "tanganke/clip-vit-base-patch32_gtsrb",
        "template": gtsrb_template
    },
    "MNIST": {
        "dataset_id": "ylecun/mnist",
        "model_id": "tanganke/clip-vit-base-patch32_mnist",
        "template": mnist_template
    },
    "RESISC45": {
        "dataset_id": "tanganke/resisc45",
        "model_id": "tanganke/clip-vit-base-patch32_resisc45",
        "template": resisc45_template
    },
    "SUN397": {
        "dataset_id": "tanganke/sun397",
        "model_id": "tanganke/clip-vit-base-patch32_sun397",
        "template": sun397_template
    },
    "SVHN": {
        "dataset_id": "ufldl-stanford/svhn",
        "model_id": "tanganke/clip-vit-base-patch32_svhn",
        "template": svhn_template
    },
}

# Functions to load datasets and build concatenated dataset

from datasets import load_dataset, concatenate_datasets, Dataset, Features, ClassLabel, Value
from torch.utils.data import DataLoader
import torch


def load_dataset_tasks(dataset_names_list: list, split: str = "test"):
    """
    Load multiple datasets and extract their label names.

    Args:
        dataset_names_list (list): A list of dataset names (Hugging Face Hub identifiers).
        split (str): The dataset split to load (default: "test").
                     Typically "test" since this is for inference.

    Returns:
        tuple:
            - dataset_list (list): List of loaded datasets (one per dataset name).
            - labels_list (list): List of label name lists (aligned with dataset_list).
    """

    dataset_list = []  # Will store the loaded datasets
    labels_list = []   # Will store the list of labels for each dataset

    for dataset_name in dataset_names_list:

        # Special case: SVHN dataset requires a specific configuration ("cropped_digits")
        if dataset_name == 'ufldl-stanford/svhn':
            dataset = load_dataset(dataset_name, 'cropped_digits', split=split)
            dataset_list.append(dataset)

        else:
            # For all other datasets, load directly with the given split
            dataset = load_dataset(dataset_name, split=split)
            dataset_list.append(dataset)

        # Extract label names from the dataset metadata (class names)
        labels_list.append(dataset.features["label"].names)

    # Return both datasets and their corresponding labels
    return dataset_list, labels_list




def create_dataloader_from_list(dataset_list, labels_list=None, batch_size: int = 32, shuffle: bool = False):
    """
    Combine multiple Hugging Face datasets into a single PyTorch DataLoader with unified labels.

    Args:
        dataset_list (list[Dataset]): List of HF Dataset objects.
        labels_list (list[list[str]]|None): Optional list of label-name lists (one per dataset).
                                            If None, the function will try to extract names from each dataset's ClassLabel.
        batch_size (int): batch size.
        shuffle (bool): whether to shuffle.

    Returns:
        dataloader: PyTorch DataLoader yielding (images_list, labels_tensor)
        label_mapping: dict mapping label_name -> unified_int
        inverse_mapping: dict mapping unified_int -> label_name
    """

    # --- 1) collect per-dataset label name lists (strings) ---
    per_ds_label_names = []
    for i, ds in enumerate(dataset_list):
        feat = ds.features.get("label", None)
        if isinstance(feat, ClassLabel):
            # dataset has ClassLabel: we can use the names directly
            names = feat.names
        elif labels_list is not None and i < len(labels_list):
            # user supplied a labels_list for this dataset
            names = [str(x) for x in labels_list[i]]
        else:
            # fallback: infer unique label values from the dataset and cast to str
            # note: this produces names like "0", "1", ... if labels are ints
            uniq = sorted(set(dataset_list[i]["label"]))
            names = [str(x) for x in uniq]
        per_ds_label_names.append(names)

    # --- 2) build combined (global) label list and mapping ---
    # If you want to deduplicate identical label names across datasets, replace the next two lines
    # combined_labels = []
    # for names in per_ds_label_names: combined_labels.extend(names)
    combined_labels = [lab for names in per_ds_label_names for lab in names]

    label_mapping = {lab: idx for idx, lab in enumerate(combined_labels)}
    inverse_mapping = {v: k for k, v in label_mapping.items()}

    print("Unified label mapping (sample):", dict(list(label_mapping.items())[:10]))

    # --- 3) create a mapper for each dataset that converts the dataset's label -> unified int ---
    remapped_datasets = []
    for ds, names in zip(dataset_list, per_ds_label_names):
        feat = ds.features.get("label", None)

        if isinstance(feat, ClassLabel):
            # safe: use int2str to get the readable name
            def mapper(example, _feat=feat):
                name = _feat.int2str(example["label"])
                return {"label": int(label_mapping[name])}
        else:
            # labels are not ClassLabel. They could be ints indexing into `names` or raw strings
            # We'll try to interpret numeric labels as indices into names; otherwise use str(value).
            def mapper(example, _names=names):
                val = example["label"]
                # if integer and within range of provided names, take that name
                if isinstance(val, int) and 0 <= val < len(_names):
                    name = _names[val]
                else:
                    # fallback to stringing the value
                    name = str(val)
                return {"label": int(label_mapping[name])}

        # Force features so resulting dataset's "label" column is a plain int (avoid ClassLabel checks)
        new_features = Features({
            # keep original image feature if exists, otherwise rely on dataset to infer
            "image": ds.features["image"] if "image" in ds.features else ds.features.get("image"),
            "label": Value("int64")
        })

        remapped = ds.map(mapper, remove_columns=["label"], features=new_features)
        remapped_datasets.append(remapped)

    # --- 4) concatenate datasets and build DataLoader ---
    combined_dataset = concatenate_datasets(remapped_datasets)

    def collate_fn(batch):
        images = [x["image"] for x in batch]            # keep PIL images (or path) as-is
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return images, labels

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return dataloader, label_mapping, inverse_mapping
