from pawnet.dataset import main as dataset
from pawnet.modeling.train import main as train
from pawnet.modeling.predict import main as predict


def main():
    MODEL_VERSION = 0
    print("Hello from dd2424-group-project!")

    test_loader = dataset(split="test", model_version=MODEL_VERSION)
    train_loader, validation_loader = dataset(train_size=0.8, model_version=MODEL_VERSION)

    train(
        train_loader=train_loader, validation_loader=validation_loader, model_version=MODEL_VERSION
    )
    predict(val_loader=test_loader, model_version=MODEL_VERSION)


if __name__ == "__main__":
    main()
