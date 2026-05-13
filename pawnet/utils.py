from loguru import logger
import torchvision.models as models


def get_model(model_version: str, use_weights: bool = True):
    match model_version:
        case 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if use_weights else None
            model = models.efficientnet_b0(weights=weights)
        case 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if use_weights else None
            model = models.efficientnet_b1(weights=weights)
        case 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_weights else None
            model = models.resnet18(weights=weights)
        case _:
            logger.error(f"Model version {model_version} not recognized.")
            exit(1)

    return model, weights
