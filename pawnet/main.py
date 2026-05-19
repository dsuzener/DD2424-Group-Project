from pathlib import Path
import typer
from pawnet.dataset import main as dataset
from pawnet.modeling.train import main as train
from pawnet.modeling.predict import main as predict

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
    augment: bool = False,
    L2: bool = False
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
    )
    train_loader, validation_loader = dataset(
        target_types=target_types,
        train_size=train_size,
        model_version=model_version,
        stratify=stratify,
        batch_size=batch_size,
        num_layers=num_layers,
        gradual_unfreezing=gradual_unfreezing,
        augment = augment,
    )

    if train_model:
        train(
            train_loader=train_loader,
            validation_loader=validation_loader,
            model_version=model_version,
            epochs=epochs,
            target_types=target_types,
            num_layers=num_layers,
            train_size=train_size,
            batch_size=batch_size,
            stratify=stratify,
            gradual_unfreezing=gradual_unfreezing,
            L2 = L2,
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
        # force_model_path=Path("models/baseline/efficientnet_b2/model_9891.pkl"),
    )


if __name__ == "__main__":
    app()
