import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.nico_dataset import NICODataset

################
### SETTINGS ###
################

confounder_settings = {
    "CelebA": {"constructor": CelebADataset},
    "CUB": {"constructor": CUBDataset},
    "MultiNLI": {"constructor": MultiNLIDataset},
    "NICO": {"constructor": NICODataset},
}


########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    # Pass 4-way split params to CelebA if available
    extra_kwargs = {}
    num_val = getattr(args, "num_val_samples_per_class", None)
    if num_val is not None:
        extra_kwargs["num_val_samples_per_class"] = num_val
        extra_kwargs["split_seed"] = getattr(args, "seed", 0)

    full_dataset = confounder_settings[args.dataset]["constructor"](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
        **extra_kwargs,
    )
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )

    use_4way = num_val is not None and train
    if train:
        if use_4way:
            splits = ["train", "id_val", "test"]
        else:
            splits = ["train", "val", "test"]
    else:
        splits = ["test"]
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [
        DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
        for split in splits
    ]

    # Create OOD val subset sampled with replacement from test
    if use_4way:
        split_seed = getattr(args, "seed", 0)
        rng = np.random.RandomState(split_seed)
        test_indices = subsets["test"].indices
        ood_val_indices = []
        for cls in range(full_dataset.n_classes):
            class_mask = full_dataset.y_array[test_indices] == cls
            class_indices = test_indices[class_mask]
            sampled = rng.choice(class_indices, size=num_val, replace=True)
            ood_val_indices.extend(sampled)
        ood_val_subset = Subset(full_dataset, ood_val_indices)
        ood_val_dro = DRODataset(
            ood_val_subset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
        # Return: [train, id_val, ood_val, test]
        dro_subsets.insert(2, ood_val_dro)

    return dro_subsets
