from pathlib import Path
from typing import cast
from loguru import logger
from tqdm import tqdm
import typer

from pawnet.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
import torch.nn as nn

from torchvision import models
from PIL import Image

from pawnet.utils import get_model

app = typer.Typer()


@app.command()
def main(
    val_loader,
    model_version: str = "efficientnet_b0",
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    model, _ = get_model(model_version=model_version, use_weights=False)
    model.classifier[1] = nn.Linear(cast(nn.Linear, model.classifier[1]).in_features, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    num_correct = 0
    num_total = 0

    all_preds = []
    all_labels = []
    for images, labels in tqdm(val_loader, desc="Predicting"):
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            preds = outputs.argmax(dim=1)

            num_correct += (preds == labels).sum().item()

            num_total += labels.size(0)

            all_preds.append(preds.cpu())

            all_labels.append(labels.cpu())

    accuracy = num_correct / num_total

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    app()
