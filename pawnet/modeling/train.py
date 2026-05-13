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

from pawnet.utils import get_model

app = typer.Typer()


@app.command()
def main(
    train_loader,
    validation_loader,
    model_version: str = "efficientnet_b0",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    epochs: int = 1,
):
    # Load model and initial weights
    model, _ = get_model(model_version=model_version, use_weights=True)

    model.train()

    # Freeze all layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    # New classifier layer is unfrozen per default
    model.classifier[1] = nn.Linear(cast(nn.Linear, model.classifier[1]).in_features, 2)

    for param in model.classifier.parameters():
        param.requires_grad = True

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=None)  # TODO: add class weights
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Train model
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

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
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                num_correct += (preds == labels).sum().item()

                num_total += labels.size(0)

        val_acc = num_correct / num_total

        print(f"Epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    # Save trained model
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model trained and saved to {model_path}.")


if __name__ == "__main__":
    app()
