from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "pawnet-train"

# Persist dataset downloads + preprocessing artifacts and training outputs across runs.
DATA_VOLUME_NAME = os.environ.get("PAWNET_MODAL_DATA_VOLUME", "pawnet-data")
MODELS_VOLUME_NAME = os.environ.get("PAWNET_MODAL_MODELS_VOLUME", "pawnet-models")

REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_DATA_DIR = REMOTE_PROJECT_DIR / "data"
REMOTE_MODELS_DIR = REMOTE_PROJECT_DIR / "models"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install dependencies from `pyproject.toml` + `uv.lock` into a venv.
    .uv_sync()
    .workdir(str(REMOTE_PROJECT_DIR))
    # Provide the repo source code to the container filesystem (Modal 1.x replaces Mounts with Image.add_local_*).
    .add_local_dir(
        ".",
        remote_path=str(REMOTE_PROJECT_DIR),
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            "data",
            "models",
            "reports",
            "notebooks",
        ],
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A10G",
    timeout=int(os.environ.get("PAWNET_MODAL_TIMEOUT_S", "60")) * 60,
    volumes={
        str(REMOTE_DATA_DIR): data_volume,
        str(REMOTE_MODELS_DIR): models_volume,
    },
)
def train_remote(
    model_version: str = "efficientnet_b2",
    epochs: int = 30,
    train_size: float = 0.90,
    stratify: bool = True,
    target_types: str = "category",
    num_layers: int = 0,
    batch_size: int = 128,
    gradual_unfreezing: bool = False,
    labeled_fraction: float = 1.0,
    use_pseudolabels: bool = False,
    pseudolabel_threshold: float = 0.5,
    pseudolabel_weight: float = 1.0,
    pseudolabel_start_epoch: int = 3,
    augment: bool = False,
    l2: float = 0.0,
) -> None:
    os.chdir(REMOTE_PROJECT_DIR)

    from pawnet.main import main as pawnet_main

    pawnet_main(
        model_version=model_version,
        train_model=True,
        epochs=epochs,
        train_size=train_size,
        stratify=stratify,
        target_types=target_types,
        num_layers=num_layers,
        batch_size=batch_size,
        gradual_unfreezing=gradual_unfreezing,
        labeled_fraction=labeled_fraction,
        use_pseudolabels=use_pseudolabels,
        pseudolabel_threshold=pseudolabel_threshold,
        pseudolabel_weight=pseudolabel_weight,
        pseudolabel_start_epoch=pseudolabel_start_epoch,
        augment=augment,
        l2=l2,
    )

    data_volume.commit()
    models_volume.commit()


@app.local_entrypoint()
def main(
    detach: bool = False,
    model_version: str = "efficientnet_b2",
    epochs: int = 30,
    train_size: float = 0.90,
    stratify: bool = True,
    target_types: str = "category",
    num_layers: int = 0,
    batch_size: int = 128,
    gradual_unfreezing: bool = False,
    labeled_fraction: float = 1.0,
    use_pseudolabels: bool = False,
    pseudolabel_threshold: float = 0.9,
    pseudolabel_weight: float = 1.0,
    pseudolabel_start_epoch: int = 1,
    augment: bool = False,
    l2: float = 0.0,
) -> None:
    kwargs = dict(
        model_version=model_version,
        epochs=epochs,
        train_size=train_size,
        stratify=stratify,
        target_types=target_types,
        num_layers=num_layers,
        batch_size=batch_size,
        gradual_unfreezing=gradual_unfreezing,
        labeled_fraction=labeled_fraction,
        use_pseudolabels=use_pseudolabels,
        pseudolabel_threshold=pseudolabel_threshold,
        pseudolabel_weight=pseudolabel_weight,
        pseudolabel_start_epoch=pseudolabel_start_epoch,
        augment=augment,
        l2=l2,
    )
    if detach:
        call = train_remote.spawn(**kwargs)
        print(f"Spawned Modal run (detached): {call.object_id}")
        print(
            "Local command finished; training continues remotely. "
            "Open the Modal run URL printed above to follow logs/progress."
        )
        return

    train_remote.remote(**kwargs)
