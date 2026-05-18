import os
from pathlib import Path
from typing import cast

from loguru import logger
import torch
from tqdm import tqdm
import typer

from pawnet.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split
from pawnet.utils import get_dataset_paths, get_model


app = typer.Typer()


class CustomDataset(datasets.OxfordIIITPet):
    def __init__(
        self, root, split="trainval", target_types="category", transform=None, download=True
    ):
        super().__init__(
            root=root, split=split, target_types=target_types, download=True, transform=transform
        )

    def save_preprocessed_dataset(
        self,
        model_version: str,
        split: str,
        indices: list[int] | None = None,
    ):
        features_path, labels_path = get_dataset_paths(PROCESSED_DATA_DIR, model_version, split)

        paths_path = PROCESSED_DATA_DIR / f"{model_version}_{split}_paths.pt"

        if features_path.exists() and labels_path.exists() and paths_path.exists():
            return

        if indices is None:
            indices = list(range(len(self)))

        features, labels, image_paths = [], [], []

        for i in tqdm(indices, desc=f"Preprocessing {split} dataset"):
            img, label = self[i]
            features.append(img)
            labels.append(label)
            image_paths.append(str(self._images[i]))

        features_tensor = torch.stack(features)
        labels_tensor = torch.tensor(labels)

        torch.save(features_tensor, features_path)
        torch.save(labels_tensor, labels_path)
        torch.save(image_paths, paths_path)

        logger.success(f"Preprocessed {split} dataset saved to {features_path} and {labels_path}.")


class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, model_version: str, split: str):
        features_path, labels_path = get_dataset_paths(PROCESSED_DATA_DIR, model_version, split)
        paths_path = PROCESSED_DATA_DIR / f"{model_version}_{split}_paths.pt"

        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path).long()
        self.paths = torch.load(paths_path, weights_only=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.paths[idx]


@app.command()
def main(
    data_path: Path = RAW_DATA_DIR,
    split: str = "trainval",
    target_types: str = "binary-category",
    train_size: float = 0.8,
    batch_size: int = 128,
    model_version: str = "efficientnet_b0",
    stratify: bool = False,
):
    processed_data = PROCESSED_DATA_DIR / f"{model_version}"
    os.makedirs(processed_data.parent, exist_ok=True)

    _, weights = get_model(model_version=model_version, use_weights=True)

    # Need to change some things below probably
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    transform = weights.transforms()  # type: ignore

    logger.info("Loading, preporcessing, and splitting dataset...")
    dataset = CustomDataset(
        root=data_path, split=split, target_types=target_types, transform=transform, download=True
    )

    if split == "trainval":
        num_train_size = int(train_size * len(dataset))
        if stratify:
            indices = list(
                range(len(dataset))
            )  # TODO: test stratify (labels may not work for binary)
            train_indices, val_indices = train_test_split(
                indices,
                train_size=train_size,
                stratify=[dataset[i][1] for i in indices],
                random_state=42,
            )
            dataset.save_preprocessed_dataset(
                model_version,
                "train",
                train_indices,
            )

            dataset.save_preprocessed_dataset(
                model_version,
                "val",
                val_indices,
            )
        else:
            train_set, val_set = random_split(
                dataset,
                [num_train_size, len(dataset) - num_train_size],
                generator=torch.Generator().manual_seed(42),
            )
            dataset.save_preprocessed_dataset(
                model_version,
                "train",
                cast(list[int], train_set.indices),
            )
            dataset.save_preprocessed_dataset(
                model_version,
                "val",
                cast(list[int], val_set.indices),
            )

        train_loader = DataLoader(
            ProcessedDataset(model_version, "train"),
            batch_size=batch_size,
            shuffle=True,
            # num_workers=4,
            # persistent_workers=True,
        )
        val_loader = DataLoader(
            ProcessedDataset(model_version, "val"),
            batch_size=batch_size,
            shuffle=False,
            # num_workers=4,
            # persistent_workers=True,
        )
        logger.success("Train and validation datasets ready.")

        return train_loader, val_loader
    else:
        dataset.save_preprocessed_dataset(
            model_version,
            "test",
        )

        test_loader = DataLoader(
            ProcessedDataset(model_version, "test"),
            batch_size=batch_size,
            shuffle=False,
            # num_workers=4,
            # persistent_workers=True,
        )
        logger.success("Test dataset ready.")

        return test_loader


if __name__ == "__main__":
    app()
