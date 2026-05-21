import os
import csv
import threading
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
from pawnet.modeling.run_paths import RunConfig, ensure_run_dir, find_latest_checkpoint

app = typer.Typer()


@app.command()
def main(
    train_loader,
    validation_loader,
    unlabeled_loader=None,
    model_version: str = "efficientnet_b0",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    epochs: int = 20,
    target_types: str = "binary-category",
    num_layers: int = 0,
    train_size: float = 0.80,
    batch_size: int = 128,
    stratify: bool = True,
    gradual_unfreezing: bool = False,
    labeled_fraction: float = 1.0,
    use_pseudolabels: bool = False,
    pseudolabel_threshold: float = 0.9,
    pseudolabel_weight: float = 1.0,
    pseudolabel_start_epoch: int = 1,
    l2: float = 0.0,
):
    # config stuff
    # if changing the config, change the parameters of this function too!
    run_config = RunConfig(
        model_version=model_version,
        target_types=target_types,
        train_size=train_size,
        labeled_fraction=labeled_fraction,
        batch_size=batch_size,
        stratify=stratify,
        num_layers=num_layers,
        gradual_unfreezing=gradual_unfreezing,
        use_pseudolabels=use_pseudolabels,
        pseudolabel_threshold=pseudolabel_threshold,
        pseudolabel_weight=pseudolabel_weight,
        pseudolabel_start_epoch=pseudolabel_start_epoch,
    )
    run_dir = ensure_run_dir(run_config)
    best_model_path = run_dir / "best.pt"
    final_model_path = run_dir / "final.pt"

    latest = find_latest_checkpoint(run_dir)
    max_epoch_num = latest[0] if latest else 0
    latest_epoch_checkpoint_path = latest[1] if latest else (run_dir / "checkpoints" / "epoch_0.pt")

    metrics_csv_path = run_dir / "metrics.csv"

    num_outputs = 2 if target_types == "binary-category" else 37

    # Load model and initial weights
    if latest and latest_epoch_checkpoint_path.exists():
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

    if latest and latest_epoch_checkpoint_path.exists():
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA/MPS not available; training will run on CPU. "
            "If you expected a GPU (e.g. on Modal), install a CUDA-enabled PyTorch build."
        )
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=None)  # TODO: weights
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # for storing metrics
    def _append_csv_row(path: Path, header: list[str], row: dict[str, object], lock: threading.Lock):
        with lock:
            exists = path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)

    metrics_lock = threading.Lock()

    # Train model
    best_acc = 0.0
    epochs = max_epoch_num + epochs
    for epoch in range(max_epoch_num + 1, epochs + 1):
        model.train()
        total_loss = 0

        # pseudolabeling stats
        total_unsup_loss = 0.0
        total_unsup_kept = 0
        total_unsup_seen = 0

        use_unsup = (
            use_pseudolabels
            and unlabeled_loader is not None
            and epoch >= pseudolabel_start_epoch
            and pseudolabel_weight > 0.0
        )

        unlabeled_iter = iter(unlabeled_loader) if use_unsup else None

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            unsup_loss = None
            if use_unsup and unlabeled_iter is not None:
                # get unlabeled batch and loop around if end
                try:
                    u_images, _u_paths = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    u_images, _u_paths = next(unlabeled_iter)

                u_images = u_images.to(device)
                u_logits = model(u_images)
                u_probs = torch.softmax(u_logits, dim=1)
                u_conf, u_pseudo = torch.max(u_probs, dim=1)
                keep = u_conf >= pseudolabel_threshold
                total_unsup_seen += int(u_images.size(0))

                # Only keep if above threshold
                if keep.any():
                    total_unsup_kept += int(keep.sum().item())
                    unsup_loss = criterion(u_logits[keep], u_pseudo[keep])
                    loss = loss + (pseudolabel_weight * unsup_loss)
            # L2 regularization (simple weight decay term)
            if l2 > 0:
                l2_reg = sum(
                    p.pow(2).sum() for name, p in model.named_parameters() if "bias" not in name
                )
                loss = loss + (l2 * l2_reg)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Optimize
            optimizer.step()

            total_loss += loss.item()
            if unsup_loss is not None:
                total_unsup_loss += float(unsup_loss.item())

        avg_loss = total_loss / len(train_loader)
        avg_unsup_loss = (total_unsup_loss / max(1, len(train_loader))) if use_unsup else 0.0

        # Evaluate on validation set
        model.eval()

        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for images, labels, _ in tqdm(validation_loader, desc=f"Validation loss"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                num_correct += (preds == labels).sum().item()

                num_total += labels.size(0)

        val_acc = num_correct / num_total

        # PSeudolabing stats logging
        extras = ""
        if use_unsup:
            extras = (
                f" unsup_loss={avg_unsup_loss:.4f}"
                f" kept={total_unsup_kept}/{total_unsup_seen}"
            )
        print(f"Epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}{extras}")
        _append_csv_row(
            metrics_csv_path,
            header=["epoch", "loss", "val_acc", "unsup_loss", "unsup_kept", "unsup_seen"],
            row={
                "epoch": epoch,
                "loss": avg_loss,
                "val_acc": val_acc,
                "unsup_loss": avg_unsup_loss if use_unsup else "",
                "unsup_kept": total_unsup_kept if use_unsup else "",
                "unsup_seen": total_unsup_seen if use_unsup else "",
            },
            lock=metrics_lock,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.success(f"New best model saved (val_acc={val_acc:.4f}) to {best_model_path}")

        if epoch % 10 == 0:
            model_checkpoint_path = run_dir / "checkpoints" / f"epoch_{epoch}.pt"
            print(f"Saving model after epoch {epoch} to checkpoint {model_checkpoint_path}...")
            torch.save(model.state_dict(), model_checkpoint_path)
            logger.success(f"Model checkpoint trained and saved to {model_checkpoint_path}.")

    # Save trained model
    torch.save(model.state_dict(), final_model_path)
    logger.success(f"Model trained and saved to {final_model_path}.")


if __name__ == "__main__":
    app()
