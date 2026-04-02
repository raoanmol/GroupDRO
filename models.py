import torch.nn as nn
from torchvision import models


model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'resnet18': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'efficientnet_b0': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'convnext_base': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'vit_base16': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'swin_base': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    }
}


# --- Head-replacement helpers ---

def _replace_resnet_head(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)


def _replace_efficientnet_b0_head(model, num_classes):
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)


def _replace_convnext_base_head(model, num_classes):
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)


def _replace_vit_base16_head(model, num_classes):
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)


def _replace_swin_base_head(model, num_classes):
    model.head = nn.Linear(model.head.in_features, num_classes)


# --- Model registry ---

MODEL_REGISTRY = {
    "resnet18": {
        "constructor": models.resnet18,
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
        "replace_head": _replace_resnet_head,
    },
    "resnet34": {
        "constructor": models.resnet34,
        "weights": models.ResNet34_Weights.IMAGENET1K_V1,
        "replace_head": _replace_resnet_head,
    },
    "resnet50": {
        "constructor": models.resnet50,
        "weights": models.ResNet50_Weights.IMAGENET1K_V1,
        "replace_head": _replace_resnet_head,
    },
    "wideresnet50": {
        "constructor": models.wide_resnet50_2,
        "weights": models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        "replace_head": _replace_resnet_head,
    },
    "efficientnet_b0": {
        "constructor": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "replace_head": _replace_efficientnet_b0_head,
    },
    "convnext_base": {
        "constructor": models.convnext_base,
        "weights": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "replace_head": _replace_convnext_base_head,
    },
    "vit_base16": {
        "constructor": models.vit_b_16,
        "weights": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "replace_head": _replace_vit_base16_head,
    },
    "swin_base": {
        "constructor": models.swin_b,
        "weights": models.Swin_B_Weights.IMAGENET1K_V1,
        "replace_head": _replace_swin_base_head,
    },
}


def create_model(model_name, num_classes, pretrained=True):
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")

    entry = MODEL_REGISTRY[model_name]
    weights = entry["weights"] if pretrained else None
    model = entry["constructor"](weights=weights)
    entry["replace_head"](model, num_classes)
    return model
