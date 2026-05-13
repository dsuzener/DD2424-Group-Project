from pathlib import Path
from typing import cast

from loguru import logger
import torch
from tqdm import tqdm
import typer

from pawnet.config import MODELS_DIR, PROCESSED_DATA_DIR
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn

app = typer.Typer()


@app.command()
def main(
    train_loader,
    validation_loader,
    model_version: int = 0,
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    epochs: int = 20,
):
    # Load model and initial weights
    model = None
    weights = None
    match model_version:
        case 0:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        case 1:
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
            model = models.efficientnet_b1(weights=weights)
        case _:
            logger.error(f"Model version {model_version} not recognized.")
            return
        
    model.train()

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    # New classifier layer is unfrozen per default
    model.classifier[1] = nn.Linear(
        cast(nn.Linear, model.classifier[1]).in_features,
        2
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=None) # TODO: add class weights
    optimizer = optim.Adam(
        model.classifier.parameters()
    )

    # Train model
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Optimize
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on validation set
        model.eval()

        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                num_correct += (
                    preds == labels
                ).sum().item()

                num_total += labels.size(0)

        val_acc = num_correct / num_total

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"loss={avg_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    # Save trained model
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model trained and saved to {model_path}.")


if __name__ == "__main__":
    app()
