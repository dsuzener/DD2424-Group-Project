from pathlib import Path
import typer
from pawnet.dataset import main as dataset
from pawnet.modeling.train import main as train
from pawnet.modeling.predict import main as predict

app = typer.Typer()


@app.command()
def main(
    model_version: str = "efficientnet_b0",
    train_model: bool = True,
    epochs: int = 1,
    train_size: float = 0.80,
    stratify: bool = True,
    target_types="binary-category",
    num_layers: int = 0,
    batch_size: int = 128,
    gradual_unfreezing: bool = False,
):
    print("Hello from dd2424-group-project!")

    test_loader = dataset(
        split="test",
        target_types=target_types,
        model_version=model_version,
        batch_size=batch_size,
        stratify=stratify,
    )
    train_loader, validation_loader = dataset(
        target_types=target_types,
        train_size=train_size,
        model_version=model_version,
        stratify=stratify,
        batch_size=batch_size,
    )

    if train_model:
        train(
            train_loader=train_loader,
            validation_loader=validation_loader,
            model_version=model_version,
            epochs=epochs,
            target_types=target_types,
            num_layers=num_layers,
        )
    predict(
        val_loader=test_loader,
        model_version=model_version,
        target_types=target_types,
        # force_model_path=Path("models/baseline/efficientnet_b2/model_9891.pkl"),
    )


if __name__ == "__main__":
    app()
