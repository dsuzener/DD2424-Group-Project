from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from pawnet.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


app = typer.Typer()


class CustomDataset(datasets.OxfordIIITPet):
    def __init__(
        self, root, split="trainval", target_types="category", transform=None, download=True
    ):
        super().__init__(
            root=root, split=split, target_types=target_types, download=True, transform=transform
        )


@app.command()
def main(
    data_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    split: str = "trainval",
    target_types: str = "binary-category",
    train_size: float = 0.8,
):

    # Need to change some things below probably
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    logger.info("Loading, preporcessing, and splitting dataset...")
    dataset = CustomDataset(
        root=data_path, split=split, target_types=target_types, transform=transform, download=True
    )

    train_size = int(train_size * len(dataset))
    train, val = random_split(dataset, [train_size, len(dataset) - train_size]) # TODO: stratify train/val split

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False)
    logger.success("Train and validation datasets ready.")

    return train_loader, val_loader


if __name__ == "__main__":
    app()
