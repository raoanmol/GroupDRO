import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset


class CelebADataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        model_type,
        augment_data,
        num_val_samples_per_class=None,
        split_seed=0,
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.model_type = model_type

        # Read in attributes
        self.attrs_df = self._read_attrs(root_dir)

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root_dir, "img_align_celeba")
        if not os.path.isdir(self.data_dir):
            self.data_dir = os.path.join(self.root_dir, "data", "img_align_celeba")
        self.filename_array = self.attrs_df["image_id"].values
        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (
            self.y_array * (self.n_groups / 2) + self.confounder_array
        ).astype("int")

        # Read in train/val/test splits
        self.split_df = self._read_split(root_dir)
        self.split_array = self.split_df["partition"].values.copy()
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        # 4-way split: redistribute original val into train/test,
        # then sample in-domain val from expanded train
        if num_val_samples_per_class is not None:
            rng = np.random.RandomState(split_seed)

            # 1. Redistribute original val (split==1) 50/50 into train/test
            orig_val_indices = np.where(self.split_array == 1)[0]
            rng.shuffle(orig_val_indices)
            mid = len(orig_val_indices) // 2
            self.split_array[orig_val_indices[:mid]] = 0  # → train
            self.split_array[orig_val_indices[mid:]] = 2  # → test

            # 2. Sample in-domain val from expanded train (without replacement, per class)
            train_indices = np.where(self.split_array == 0)[0]
            id_val_indices = []
            for cls in range(self.n_classes):
                class_mask = self.y_array[train_indices] == cls
                class_indices = train_indices[class_mask]
                sampled = rng.choice(
                    class_indices, size=num_val_samples_per_class, replace=False
                )
                id_val_indices.extend(sampled)
            id_val_indices = np.array(id_val_indices)
            self.split_array[id_val_indices] = 1  # → id_val

            # 3. Update split_dict (OOD val handled separately in confounder_utils)
            self.split_dict = {"train": 0, "id_val": 1, "test": 2}

        if model_attributes[self.model_type]["feature_type"] == "precomputed":
            self.features_mat = torch.from_numpy(
                np.load(
                    os.path.join(
                        root_dir,
                        "features",
                        model_attributes[self.model_type]["feature_filename"],
                    )
                )
            ).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_celebA(
                self.model_type, train=True, augment_data=augment_data
            )
            self.eval_transform = get_transform_celebA(
                self.model_type, train=False, augment_data=augment_data
            )

    @staticmethod
    def _find_file(root_dir, basename):
        """Look for basename.csv or basename.txt in root_dir or root_dir/data/."""
        for directory in [root_dir, os.path.join(root_dir, "data")]:
            for ext in [".csv", ".txt"]:
                path = os.path.join(directory, basename + ext)
                if os.path.isfile(path):
                    return path, ext
        raise FileNotFoundError(
            f"Could not find {basename}.csv or {basename}.txt "
            f"in {root_dir} or {os.path.join(root_dir, 'data')}"
        )

    @staticmethod
    def _read_attrs(root_dir):
        path, ext = CelebADataset._find_file(root_dir, "list_attr_celeba")
        if ext == ".csv":
            return pd.read_csv(path)
        # Original CelebA .txt: line 1 = count, line 2 = attr names, rest = data
        df = pd.read_csv(path, sep=r"\s+", skiprows=1)
        df.index.name = "image_id"
        df.reset_index(inplace=True)
        return df

    @staticmethod
    def _read_split(root_dir):
        path, ext = CelebADataset._find_file(root_dir, "list_eval_partition")
        if ext == ".csv":
            return pd.read_csv(path)
        # Original CelebA .txt: no header, columns are filename and partition
        return pd.read_csv(
            path, sep=r"\s+", header=None, names=["image_id", "partition"]
        )

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)


def get_transform_celebA(model_type, train, augment_data):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if model_attributes[model_type]["target_resolution"] is not None:
        target_resolution = model_attributes[model_type]["target_resolution"]
    else:
        target_resolution = (orig_w, orig_h)

    if (not train) or (not augment_data):
        transform = transforms.Compose(
            [
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transform
