from pathlib import Path
import os
import time
from typing import cast

from loguru import logger
import torch
from tqdm import tqdm
import typer

from pawnet.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split

from pawnet.utils import get_model
from pawnet.modeling.run_paths import RunConfig, ensure_processed_dir
from torchvision.transforms import v2


app = typer.Typer()

def get_imbalanced_classes( 
    dataset, 
    train_indices: list[int],
    keep_fraction = 0.2,
    seed: int = 42
    ):
    
    generator = torch.Generator().manual_seed(seed)
    
    cat_indices = set(range(12))
    
    kept = []
    indexes_by_class = {}
    
    # Populate indexes_by_label
    for idx in train_indices:
        label = dataset[idx][1]
        indexes_by_class.setdefault(label, []).append(idx)
        
    # Keep only 20% of each cat class
    for label, class_idxs in indexes_by_class.items():
        if label in cat_indices:
            n_keep = max(1, int(len(class_idxs) * keep_fraction))
            permutation = torch.randperm(len(class_idxs), generator=generator)[:n_keep]    
            kept.extend([class_idxs[i] for i in permutation.tolist()])
        else:
            kept.extend(class_idxs)
    
    return sorted(kept) # Sort to increase reproducability


class CustomDataset(datasets.OxfordIIITPet):
    def __init__(
        self, root, split="trainval", target_types="category", transform=None, download=True
    ):
        super().__init__(
            root=root, split=split, target_types=target_types, download=True, transform=transform
        )

    def save_preprocessed_dataset(
        self,
        processed_dir: Path,
        split: str,
        indices: list[int] | None = None,
    ):
        features_path = processed_dir / f"{split}_features.pt"
        labels_path = processed_dir / f"{split}_labels.pt"
        paths_path = processed_dir / f"{split}_paths.pt"
        lock_path = processed_dir / f".{split}.lock"

        def _all_exist() -> bool:
            return features_path.exists() and labels_path.exists() and paths_path.exists()

        if _all_exist():
            return

        # Prevent concurrent writers corrupting torch.save output when multiple runs
        # preprocess the same split/config at the same time.
        lock_fd: int | None = None
        start = time.time()
        last_log = 0.0
        while lock_fd is None:
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(lock_fd, f"pid={os.getpid()}\n".encode("utf-8"))
            except FileExistsError:
                if _all_exist():
                    return

                # Stale lock handling: if a previous run crashed, the lock can be left behind.
                # If it's old, remove it and proceed.
                try:
                    age_s = time.time() - lock_path.stat().st_mtime
                    if age_s > 60 * 10:
                        logger.warning(
                            f"Stale dataset lock detected for {split} (age={int(age_s)}s); removing {lock_path}"
                        )
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue

                if (time.time() - last_log) > 5:
                    logger.info(f"Waiting for dataset lock {lock_path} for split={split}...")
                    last_log = time.time()
                if time.time() - start > 60 * 30:
                    raise TimeoutError(f"Timed out waiting for dataset lock {lock_path}")
                time.sleep(0.25)

        try:
            # Re-check after acquiring lock in case another process finished.
            if _all_exist():
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

            # Atomic-ish writes: write to temp then replace.
            tmp_features = processed_dir / f".{split}_features.pt.tmp"
            tmp_labels = processed_dir / f".{split}_labels.pt.tmp"
            tmp_paths = processed_dir / f".{split}_paths.pt.tmp"

            torch.save(features_tensor, tmp_features)
            torch.save(labels_tensor, tmp_labels)
            torch.save(image_paths, tmp_paths)

            os.replace(tmp_features, features_path)
            os.replace(tmp_labels, labels_path)
            os.replace(tmp_paths, paths_path)

            logger.success(
                f"Preprocessed {split} dataset saved to {features_path} and {labels_path}."
            )
        finally:
            try:
                if lock_fd is not None:
                    os.close(lock_fd)
            finally:
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass


class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir: Path, split: str, transform = None):
        features_path = processed_dir / f"{split}_features.pt"
        labels_path = processed_dir / f"{split}_labels.pt"
        paths_path = processed_dir / f"{split}_paths.pt"
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path).long()
        self.paths = torch.load(paths_path, weights_only=False)
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        features = self.transform(self.features[idx]) if self.transform else self.features[idx]
        
        return features, self.labels[idx], self.paths[idx]


class UnlabeledView(torch.utils.data.Dataset):
    def __init__(self, base: torch.utils.data.Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _y, path = self.base[idx]
        return x, path

class FixMatchUnlabeled(torch.utils.data.Dataset):
    def __init__(
        self,
        base,
        weak_transform=None,
        strong_transform=None,
    ):
        self.base = base
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _y, path = self.base[idx]

        weak_x = (
            self.weak_transform(x)
            if self.weak_transform
            else x
        )

        strong_x = (
            self.strong_transform(x)
            if self.strong_transform
            else x
        )

        return weak_x, strong_x, path

@app.command()
def main(
    data_path: Path = RAW_DATA_DIR,
    split: str = "trainval",
    target_types: str = "binary-category",
    train_size: float = 0.8,
    batch_size: int = 128,
    model_version: str = "efficientnet_b0",
    stratify: bool = False,
    num_layers: int = 0,
    gradual_unfreezing: bool = False,
    imbalanced_training: bool = False,
    weighted_loss: bool = False,
    labeled_fraction: float = 1.0,
    augment: bool = False,
    use_fixmatch: bool = False,
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
        imbalanced_training=imbalanced_training,
        weighted_loss=weighted_loss,
        augment=augment,
    )
    processed_dir = ensure_processed_dir(run_config)

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
            )
            train_indices, val_indices = train_test_split(
                indices,
                train_size=train_size,
                stratify=[dataset[i][1] for i in indices],
                random_state=42,
            )
            
            
            if imbalanced_training:
                train_indices = get_imbalanced_classes(dataset, train_indices)
                logger.info(f"Using imbalanced training set with {len(train_indices)} examples.")
            
            
            dataset.save_preprocessed_dataset(
                processed_dir,
                "train",
                train_indices,
            )

            dataset.save_preprocessed_dataset(
                processed_dir,
                "val",
                val_indices[:int(len(dataset) * min(0.2, 1 - train_size))],  # Use max 20% for validation
            )
            
            
        else:
            train_set, val_set = random_split(
                dataset,
                [num_train_size, len(dataset) - num_train_size],
                generator=torch.Generator().manual_seed(42),
            )
            dataset.save_preprocessed_dataset(
                processed_dir,
                "train",
                cast(list[int], train_set.indices),
            )
            dataset.save_preprocessed_dataset(
                processed_dir,
                "val",
                cast(list[int], val_set.indices),
            )
        
        train_transform = (
            v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation((-5, 5)),
                    v2.RandomResizedCrop(size=(260, 260), scale=(0.9, 1.0)),
                ]
            )
            if augment
            else None
        )

        # Pseudolabeling split
        full_train_ds = ProcessedDataset(processed_dir, "train", transform=train_transform)
        unlabeled_loader = None

        if labeled_fraction < 1.0:
            if labeled_fraction <= 0.0:
                raise typer.BadParameter("labeled_fraction must be in (0, 1].")

            n_total = len(full_train_ds)
            n_labeled = max(1, int(round(labeled_fraction * n_total)))
            labeled_split, unlabeled_split = random_split(
                range(n_total),
                [n_labeled, n_total - n_labeled],
                generator=torch.Generator().manual_seed(42),
            )
            labeled_ds = Subset(full_train_ds, cast(list[int], labeled_split.indices))
            subset = Subset(
                full_train_ds,
                cast(list[int], unlabeled_split.indices),
            )

            if use_fixmatch:
                weak_transform = v2.Compose([
                    v2.RandomHorizontalFlip(p=0.5),
                ])
                strong_transform = v2.Compose([
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation((-30, 30)),
                    v2.RandomResizedCrop(
                        size=(260, 260),
                        scale=(0.5, 1.0),
                    ),
                    v2.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    ),
                    v2.RandomErasing(p=0.25),
                ])
                unlabeled_ds = FixMatchUnlabeled(
                    subset,
                    weak_transform=weak_transform,
                    strong_transform=strong_transform,
                )
            else:
                unlabeled_ds = UnlabeledView(subset)

            train_ds_for_loader = labeled_ds
            unlabeled_loader = DataLoader(
                unlabeled_ds,
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            train_ds_for_loader = full_train_ds

        train_loader = DataLoader(
            train_ds_for_loader,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            ProcessedDataset(processed_dir, "val"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        logger.success("Train and validation datasets ready.")

        if unlabeled_loader is not None:
            return train_loader, val_loader, unlabeled_loader
        return train_loader, val_loader
    else:
        dataset.save_preprocessed_dataset(
            processed_dir,
            "test",
        )

        test_loader = DataLoader(
            ProcessedDataset(processed_dir, "test"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        logger.success("Test dataset ready.")

        return test_loader


if __name__ == "__main__":
    app()
