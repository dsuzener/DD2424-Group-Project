from pathlib import Path
from typing import cast
from loguru import logger
from tqdm import tqdm
import typer

from pawnet.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

app = typer.Typer()


@app.command()
def main(
    val_loader,
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    model = efficientnet_b0()
    model.classifier[1] = nn.Linear(
        cast(nn.Linear, model.classifier[1]).in_features,
        2
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    weights = EfficientNet_B0_Weights.DEFAULT
    model.eval()


    for images, labels in tqdm(val_loader, desc="Predicting"):
        with torch.no_grad():
            output = model(images)

        probs = torch.nn.functional.softmax(output[0], dim=0)

        class_id = int(probs.argmax().item())
        label = weights.meta["categories"][class_id]
        if label == 'tench':
            label = 'cat'
        else:
            label = 'dog'
        score = probs[class_id].item()

        print(label, score)


if __name__ == "__main__":
    app()
