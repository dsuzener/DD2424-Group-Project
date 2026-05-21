from pathlib import Path
import typer
from pawnet.dataset import main as dataset
from pawnet.modeling.train import main as train
from pawnet.modeling.predict import main as predict
import matplotlib.pyplot as plt

import torch

app = typer.Typer()


@app.command()
def main(
    model_version: str = "efficientnet_b2",
    train_model: bool = True,
    epochs: int = 30,
    train_size: float = 0.90,
    stratify: bool = True,
    target_types="category",
    num_layers: int = 0,
    batch_size: int = 64,
    gradual_unfreezing: bool = False,
    labeled_fraction: float = 1.0,
    use_pseudolabels: bool = False,
    pseudolabel_threshold: float = 0.9,
    pseudolabel_weight: float = 1.0,
    pseudolabel_start_epoch: int = 1,
    augment: bool = False,
    l2: float = 0.0,
):
    print("Hello from dd2424-group-project!")

    test_loader = dataset(
        split="test",
        target_types=target_types,
        model_version=model_version,
        batch_size=batch_size,
        stratify=stratify,
        train_size=train_size,
        num_layers=num_layers,
        gradual_unfreezing=gradual_unfreezing,
        labeled_fraction=labeled_fraction,
    )
    trainval_result = dataset(
        target_types=target_types,
        train_size=train_size,
        model_version=model_version,
        stratify=stratify,
        batch_size=batch_size,
        num_layers=num_layers,
        gradual_unfreezing=gradual_unfreezing,
        labeled_fraction=labeled_fraction,
        augment=augment,
    )
    unlabeled_loader = None
    if isinstance(trainval_result, tuple) and len(trainval_result) == 3: # this is a bit scuffed but it works
        train_loader, validation_loader, unlabeled_loader = trainval_result
    else:
        train_loader, validation_loader = trainval_result

    if train_model:
        train(
            train_loader=train_loader,
            validation_loader=validation_loader,
            unlabeled_loader=unlabeled_loader,
            model_version=model_version,
            epochs=epochs,
            target_types=target_types,
            num_layers=num_layers,
            train_size=train_size,
            batch_size=batch_size,
            stratify=stratify,
            gradual_unfreezing=gradual_unfreezing,
            labeled_fraction=labeled_fraction,
            use_pseudolabels=use_pseudolabels,
            pseudolabel_threshold=pseudolabel_threshold,
            pseudolabel_weight=pseudolabel_weight,
            pseudolabel_start_epoch=pseudolabel_start_epoch,
            l2=l2,
        )
    predict(
        val_loader=test_loader,
        model_version=model_version,
        target_types=target_types,
        train_size=train_size,
        batch_size=batch_size,
        stratify=stratify,
        num_layers=num_layers,
        gradual_unfreezing=gradual_unfreezing,
        labeled_fraction=labeled_fraction,
        use_pseudolabels=use_pseudolabels,
        pseudolabel_threshold=pseudolabel_threshold,
        pseudolabel_weight=pseudolabel_weight,
        pseudolabel_start_epoch=pseudolabel_start_epoch,
    )



if __name__ == "__main__":
    app()
