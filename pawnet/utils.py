from pathlib import Path
from loguru import logger
import torchvision.models as models


def get_model(model_version: str, use_weights: bool = True):
    match model_version:
        case "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if use_weights else None
            model = models.efficientnet_b0(weights=weights)
        case "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if use_weights else None
            model = models.efficientnet_b1(weights=weights)
        case "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if use_weights else None
            model = models.efficientnet_b2(weights=weights)
        case "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if use_weights else None
            model = models.efficientnet_b3(weights=weights)
        case "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if use_weights else None
            model = models.efficientnet_b4(weights=weights)
        case _:
            logger.error(f"Model version {model_version} not recognized.")
            exit(1)

    return model, weights


def get_dataset_paths(parent_dir: Path, model_version: str, split: str):
    features_path = parent_dir / f"{model_version}_{split}_features.pt"
    labels_path = parent_dir / f"{model_version}_{split}_labels.pt"
    return features_path, labels_path
