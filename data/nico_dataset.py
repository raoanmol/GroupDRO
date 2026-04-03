import os
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from data.confounder_dataset import ConfounderDataset

ALL_CONTEXTS = ["autumn", "dim", "grass", "outdoor", "rock", "water"]
EXCLUDED_CONTEXTS = {"dim"}
NUM_CLASSES = 60


class NICODataset(ConfounderDataset):
    """
    NICO++ DG Benchmark dataset.

    Uses context (e.g. autumn, grass, outdoor, rock, water) as the confounder.
    Groups are defined as (class, context) pairs.
    """

    def __init__(self, root_dir, target_name, confounder_names,
                 model_type, augment_data,
                 num_val_samples_per_class=None, split_seed=0):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir = os.path.join(root_dir, "DG_Benchmark", "NICO_DG_Benchmark")
        annotation_dir = os.path.join(root_dir, "DG_Benchmark", "NICO_DG_Benchmark_annotation")

        # Determine which contexts to use
        self.contexts = sorted([c for c in ALL_CONTEXTS if c not in EXCLUDED_CONTEXTS])
        self.context_to_id = {c: i for i, c in enumerate(self.contexts)}
        n_contexts = len(self.contexts)

        # Parse annotation files
        filenames = []
        labels = []
        context_ids = []
        splits = []  # 0 = train, 2 = test

        for context in self.contexts:
            ctx_id = self.context_to_id[context]
            for split_name, split_val in [("train", 0), ("test", 2)]:
                annotation_file = os.path.join(annotation_dir, f"{context}_{split_name}.txt")
                with open(annotation_file, "r") as f:
                    for line in f:
                        parts = line.strip().rsplit(maxsplit=1)
                        if len(parts) != 2:
                            continue
                        rel_path = parts[0]
                        class_id = int(parts[1])

                        # Annotation paths start with "NICO_DG/"; strip and
                        # use path relative to data_dir
                        rel_path = rel_path.replace("NICO_DG/", "", 1)

                        filenames.append(rel_path)
                        labels.append(class_id)
                        context_ids.append(ctx_id)
                        splits.append(split_val)

        self.filename_array = np.array(filenames)
        self.y_array = np.array(labels, dtype=int)
        self.confounder_array = np.array(context_ids, dtype=int)
        self.split_array = np.array(splits, dtype=int)

        # Group definition: (class, context) pairs
        self.n_classes = NUM_CLASSES
        self.n_confounders = 1
        self.n_groups = self.n_classes * n_contexts
        self.group_array = (self.y_array * n_contexts + self.confounder_array).astype(int)

        # Handle validation splits
        rng = np.random.RandomState(split_seed)

        if num_val_samples_per_class is not None:
            # 4-way split: sample ID val from train
            train_indices = np.where(self.split_array == 0)[0]
            id_val_indices = []
            for cls in range(self.n_classes):
                class_mask = self.y_array[train_indices] == cls
                class_indices = train_indices[class_mask]
                if len(class_indices) < num_val_samples_per_class:
                    sampled = class_indices
                else:
                    sampled = rng.choice(class_indices, size=num_val_samples_per_class, replace=False)
                id_val_indices.extend(sampled)
            id_val_indices = np.array(id_val_indices)
            self.split_array[id_val_indices] = 1  # id_val

            self.split_dict = {
                'train': 0,
                'id_val': 1,
                'test': 2
            }
        else:
            # 3-way split: sample ~10% of train per class as val
            train_indices = np.where(self.split_array == 0)[0]
            val_indices = []
            for cls in range(self.n_classes):
                class_mask = self.y_array[train_indices] == cls
                class_indices = train_indices[class_mask]
                n_val = max(1, int(0.1 * len(class_indices)))
                sampled = rng.choice(class_indices, size=n_val, replace=False)
                val_indices.extend(sampled)
            val_indices = np.array(val_indices)
            self.split_array[val_indices] = 1  # val

            self.split_dict = {
                'train': 0,
                'val': 1,
                'test': 2
            }

        # Transforms
        self.features_mat = None
        target_resolution = model_attributes[self.model_type]['target_resolution']
        if target_resolution is None:
            target_resolution = (224, 224)

        if augment_data:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.eval_transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def group_str(self, group_idx):
        n_contexts = len(self.contexts)
        y = group_idx // n_contexts
        c = group_idx % n_contexts
        context_name = self.contexts[c]
        return f"class = {y}, context = {context_name}"
