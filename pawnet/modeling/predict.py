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

app = typer.Typer()


@app.command()
def main(
    val_loader,
    model_version: int = 0,
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    match model_version:
        case 0:
            model = models.efficientnet_b0()
        case 1:
            model = models.efficientnet_b1()
        case _:
            logger.error(f"Model version {model_version} not recognized.")
            return
    model.classifier[1] = nn.Linear(
        cast(nn.Linear, model.classifier[1]).in_features,
        2
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    num_correct = 0
    num_wrong = 0
    for images, labels in tqdm(val_loader, desc="Predicting"):
        with torch.no_grad():
            output = model(images)

        for i in range(len(output)):
            probs = torch.nn.functional.softmax(output[i], dim=0)

            class_id = int(probs.argmax().item())
            if class_id == 0:
                label = 'cat'
            else:
                label = 'dog'
            score = probs[class_id].item()

            if not class_id == labels[i]:
                num_wrong += 1
            else:
                num_correct += 1

    print(f"Correct: {num_correct} / {num_correct + num_wrong} = {num_correct/(num_correct + num_wrong)}")


if __name__ == "__main__":
    app()
