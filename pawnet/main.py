from pawnet.dataset import main as dataset
from pawnet.modeling.train import main as train
from pawnet.modeling.predict import main as predict

def main():
    print("Hello from dd2424-group-project!")

    train_loader, val_loader = dataset()

    # train(train_loader=train_loader)
    predict(val_loader=val_loader)



if __name__ == "__main__":
    main()
