import os
import torch
import numpy as np
from torch.utils.data import Subset
from data.label_shift_utils import prepare_label_shift_data
from data.confounder_utils import prepare_confounder_data

root_dir = '/u/scr/nlp/dro/'

dataset_attributes = {
    'CelebA': {
        'root_dir': 'celebA'
    },
    'CUB': {
        'root_dir': 'cub'
    },
    'CIFAR10': {
        'root_dir': 'CIFAR10/data'
    },
    'MultiNLI': {
        'root_dir': 'multinli'
    }
}

for dataset in dataset_attributes:
    dataset_attributes[dataset]['root_dir'] = os.path.join(root_dir, dataset_attributes[dataset]['root_dir'])

shift_types = ['confounder', 'label_shift_step']

def prepare_data(args, train, return_full_dataset=False):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    if args.shift_type=='confounder':
        return prepare_confounder_data(args, train, return_full_dataset)
    elif args.shift_type.startswith('label_shift'):
        assert not return_full_dataset
        return prepare_label_shift_data(args, train)

def log_data(data, logger):
    for split_name in ['train_data', 'id_val_data', 'val_data', 'ood_val_data', 'test_data']:
        if split_name not in data or data[split_name] is None:
            continue
        label = split_name.replace('_data', '').replace('_', ' ').title()
        logger.write(f'{label} Data...\n')
        for group_idx in range(data[split_name].n_groups):
            logger.write(f'    {data[split_name].group_str(group_idx)}: '
                         f'n = {data[split_name].group_counts()[group_idx]:.0f}\n')
