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
    num_layers: int = 1,
):
    print("Hello from dd2424-group-project!")

    test_loader = dataset(
        split="test", target_types="binary-category", model_version=model_version
    )
    train_loader, validation_loader = dataset(
        train_size=train_size, model_version=model_version, stratify=stratify
    )

    if train_model:
        train(
            train_loader=train_loader,
            validation_loader=validation_loader,
            model_version=model_version,
            epochs=epochs,
            num_layers=2,
        )
    predict(val_loader=test_loader, model_version=model_version)


if __name__ == "__main__":
    app()
