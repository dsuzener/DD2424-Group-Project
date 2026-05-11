from pawnet.dataset import main as dataset
from pawnet.modeling.train import main as train
from pawnet.modeling.predict import main as predict

def main():
    print("Hello from dd2424-group-project!")

    test_loader = dataset(split="test")
    train_loader, val_loader = dataset(train_size=1.0)

    train(train_loader=train_loader)
    predict(val_loader=test_loader)


 
if __name__ == "__main__":
    main()
