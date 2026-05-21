from pathlib import Path
from typing import cast
from loguru import logger

from sklearn.metrics import f1_score
from tqdm import tqdm
import typer
import csv
import json

from pawnet.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
import torch.nn as nn

from torchvision import models
from PIL import Image

from pawnet.utils import get_model
from pawnet.modeling.run_paths import RunConfig, get_run_dir, resolve_model_path_for_predict

app = typer.Typer()


@app.command()
def main(
    val_loader,
    model_version: str = "efficientnet_b0",
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    target_types: str = "binary-category",
    train_size: float = 0.80,
    batch_size: int = 128,
    stratify: bool = True,
    num_layers: int = 0,
    gradual_unfreezing: bool = False,
    labeled_fraction: float = 1.0,
    use_pseudolabels: bool = False,
    pseudolabel_threshold: float = 0.9,
    pseudolabel_weight: float = 1.0,
    pseudolabel_start_epoch: int = 1,
    prefer_weights: str = "best",
    force_model_path: Path | None = None,
    write_predictions: bool = True,
):
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
    run_dir = get_run_dir(run_config)
    model_path = resolve_model_path_for_predict(run_dir, prefer=prefer_weights)
    num_outputs = 2 if target_types == "binary-category" else 37
    if force_model_path and force_model_path.exists():
        logger.info(f"Using forced model path: {force_model_path}")
        model_path = force_model_path

    logger.info(f"Loading model weights from {model_path}")
    model, _ = get_model(model_version=model_version, use_weights=False)
    model.classifier[1] = nn.Linear(cast(nn.Linear, model.classifier[1]).in_features, num_outputs)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # use the same number of classes as model outputs
    num_classes = num_outputs
    num_correct = {i: 0 for i in range(num_classes)}
    num_total = {i: 0 for i in range(num_classes)}

    all_preds = []
    all_labels = []
    incorrect_predictions = []
    prediction_rows: list[dict[str, object]] = []
    for images, labels, paths in tqdm(val_loader, desc="Predicting"):
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)
            confs = probs[torch.arange(outputs.size(0), device=outputs.device), preds]

            wrong_mask = preds != labels
            for i in torch.where(wrong_mask)[0]:
                i = int(i.item())
                incorrect_predictions.append(
                    {
                        "path": paths[i],
                        "true": labels[i].item(),
                        "pred": preds[i].item(),
                        "confidence": probs[i, preds[i]].item(),
                    }
                )

            if write_predictions:
                for i in range(outputs.size(0)):
                    prediction_rows.append(
                        {
                            "path": paths[i],
                            "true": int(labels[i].item()),
                            "pred": int(preds[i].item()),
                            "confidence": float(confs[i].item()),
                        }
                    )

            # per-class counting: count correct predictions for each class
            for clas in range(num_classes):
                class_mask = labels == clas
                num_total[clas] += int(class_mask.sum().item())
                if class_mask.any():
                    num_correct[clas] += int(
                        (preds[class_mask] == labels[class_mask]).sum().item()
                    )

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # concatenate collected tensors into 1D arrays for sklearn
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = (
        sum(num_correct.values()) / sum(num_total.values()) if sum(num_total.values()) > 0 else 0.0
    )
    per_class_acc = {
        clas: num_correct[clas] / num_total[clas] if num_total[clas] > 0 else 0.0
        for clas in range(num_classes)
    }

    # compute f1 score with a suitable averaging strategy
    average_mode = "binary" if num_outputs == 2 else "macro"
    f1 = f1_score(y_true=all_labels, y_pred=all_preds, average=average_mode)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 ({average_mode}): {f1:.4f}")
    for clas in range(num_classes):
        print(f"Class {clas} accuracy: {per_class_acc[clas]}")

    if write_predictions:
        run_dir.mkdir(parents=True, exist_ok=True)
        tag = "forced" if force_model_path else prefer_weights
        out_path = run_dir / f"predictions_{tag}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "true", "pred", "confidence"])
            writer.writeheader()
            writer.writerows(prediction_rows)
        logger.success(f"Wrote predictions to {out_path}")

        stats_path = run_dir / f"prediction_stats_{tag}.json"
        stats = {
            "model_path": str(model_path.relative_to(MODELS_DIR)),
            "prefer_weights": prefer_weights,
            "forced_model_path": str(force_model_path) if force_model_path else None,
            "accuracy": float(accuracy),
            "f1": float(f1),
            "f1_average": average_mode,
            "per_class_accuracy": {str(k): float(v) for k, v in per_class_acc.items()},
        }
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.success(f"Wrote prediction stats to {stats_path}")

    if False:  # debug
        print("\nIncorrect predictions:")
        print("=" * 80)

        for item in sorted(
            incorrect_predictions,
            key=lambda x: x["confidence"],
            reverse=True,
        ):
            print(
                f"{item['path']} | "
                f"true={item['true']} "
                f"pred={item['pred']} "
                f"confidence={item['confidence']:.4f}"
            )


if __name__ == "__main__":
    app()
