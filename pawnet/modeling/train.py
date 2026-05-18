import os
from pathlib import Path
from typing import cast

from loguru import logger
import torch
from tqdm import tqdm
import typer
import glob

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
    epochs: int = 20,
    target_types: str = "binary-category",
    num_layers: int = 0,
):
    model_path = MODELS_DIR / f"{model_version}/model.pkl"
    model_checkpoint_paths = glob.glob(str(MODELS_DIR / f"{model_version}/epoch_*.pkl"))
    max_epoch_num = 0
    if model_checkpoint_paths:
        max_epoch_num = max(
            [int(Path(path).stem.split("_")[1]) for path in model_checkpoint_paths]
        )
    latest_epoch_checkpoint_path = MODELS_DIR / f"{model_version}/epoch_{max_epoch_num}.pkl"

    os.makedirs(model_path.parent, exist_ok=True)

    num_outputs = 2 if target_types == "binary-category" else 37

    # Load model and initial weights
    if os.path.exists(latest_epoch_checkpoint_path):
        logger.info(
            f"Model checkpoint {latest_epoch_checkpoint_path} already exists. Loading model..."
        )
        model, _ = get_model(model_version=model_version, use_weights=False)
    else:
        logger.info(
            f"Model checkpoint {latest_epoch_checkpoint_path} does not exist. Training model..."
        )
        model, _ = get_model(model_version=model_version, use_weights=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(cast(nn.Linear, model.classifier[1]).in_features, num_outputs)

    if os.path.exists(latest_epoch_checkpoint_path):
        model.load_state_dict(torch.load(latest_epoch_checkpoint_path, weights_only=True))
        logger.success(f"Model checkpoint {latest_epoch_checkpoint_path} loaded successfully.")

    model.train()

    for param in model.classifier.parameters():
        param.requires_grad = True

    # Optionally fine-tune the last N feature blocks
    if num_layers > 0:
        # EfficientNet
        if hasattr(model, "features"):
            feature_blocks = list(model.features)
            logger.info(
                f"Unfreezing last {num_layers} EfficientNet feature blocks "
                f"(total={len(feature_blocks)})"
            )

            for block in feature_blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]

    logger.info("Trainable parameters:")
    for name in trainable:
        logger.info(name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=None)  # TODO: weights
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Train model
    best_acc = 0.0
    best_model_path = MODELS_DIR / f"{model_version}/best_model.pkl"
    epochs = max_epoch_num + epochs
    for epoch in range(max_epoch_num + 1, epochs + 1):
        model.train()
        total_loss = 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
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
            for images, labels, _ in validation_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                num_correct += (preds == labels).sum().item()

                num_total += labels.size(0)

        val_acc = num_correct / num_total

        print(f"Epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.success(f"New best model saved (val_acc={val_acc:.4f}) to {best_model_path}")

        if epoch % 10 == 0:
            model_checkpoint_path = MODELS_DIR / f"{model_version}/epoch_{epoch}.pkl"
            print(f"Saving model after epoch {epoch} to checkpoint {model_checkpoint_path}...")
            torch.save(model.state_dict(), model_checkpoint_path)
            logger.success(f"Model checkpoint trained and saved to {model_checkpoint_path}.")

    # Save trained model
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model trained and saved to {model_path}.")


if __name__ == "__main__":
    app()
