from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from loguru import logger
from pawnet.config import MODELS_DIR, PROCESSED_DATA_DIR

# function for dirname sanitization
def _slug(value: object) -> str:
    text = str(value)
    text = text.strip().lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_.=-]+", "-", text) # nonalphanum to dash
    text = re.sub(r"-{2,}", "-", text).strip("-") # trim extra dishes
    return text or "na" # fallback

# function to render flaots for dir names
def _float_folder(value: float) -> str:
    # avoid locale issues with commas and avoid rounding tiny values
    if value == 0:
        return "0"

    abs_value = abs(value)
    if abs_value < 1e-4 or abs_value >= 1e4:
        # scientific
        text = f"{value:.12e}"
        mantissa, exp = text.split("e", 1)
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp = str(int(exp))  # drop leading + and zeros
        return f"{mantissa}e{exp}".replace(".", "p")

    # need to do this instead of float rendering sqtuff to avoid locale issues with commas
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")

@dataclass(frozen=True)
class RunConfig:
    model_version: str
    target_types: str
    train_size: float
    labeled_fraction: float
    batch_size: int
    stratify: bool
    num_layers: int
    gradual_unfreezing: bool
    imbalanced_training: bool
    weighted_loss: bool
    augment: bool = False
    l2: float = 0.0
    use_pseudolabels: bool = False
    pseudolabel_threshold: float = 0.9
    pseudolabel_weight: float = 1.0
    pseudolabel_start_epoch: int = 1
    use_fixmatch: bool = False

    def folder_parts(self) -> list[str]:
        parts = [
            f"model={_slug(self.model_version)}",
            f"targets={_slug(self.target_types)}",
            f"train={_float_folder(self.train_size)}",
            f"labeled={_float_folder(self.labeled_fraction)}",
            f"batch={self.batch_size}",
            f"stratify={int(self.stratify)}",
            f"layers={self.num_layers}",
            f"gradual={int(self.gradual_unfreezing)}",
        ]
        parts.append(f"augment={int(self.augment)}")
        parts.append(f"l2={_float_folder(self.l2)}")
        parts.append(f"pseudolabels={int(self.use_pseudolabels)}")
        if self.use_pseudolabels:
            parts.extend(
                [
                    f"plthr={_float_folder(self.pseudolabel_threshold)}",
                    f"plw={_float_folder(self.pseudolabel_weight)}",
                    f"plstart={self.pseudolabel_start_epoch}",
                ]
            )
        parts.append(f"fixmatch={int(self.use_fixmatch)}")
        if self.imbalanced_training:
            parts.append(f"imbalanced={int(self.imbalanced_training)}")
        if self.weighted_loss:
            parts.append(f"weighted={int(self.weighted_loss)}")
        return parts

# function to get run dir path from config (doesn't create)
def get_run_dir(config: RunConfig) -> Path:
    run_dir = MODELS_DIR
    for part in config.folder_parts():
        run_dir = run_dir / part
    return run_dir

# create run dir if not exists
def ensure_run_dir(config: RunConfig) -> Path:
    run_dir = get_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"

    # avoid overwriting
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != asdict(config):
            raise ValueError(
                "Existing run folder config does not match requested config.\n"
                f"run_dir={run_dir}\n"
                f"existing={existing}\n"
                f"requested={asdict(config)}"
            )
    else:
        config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info(f"Wrote run config to {config_path}")
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir

# get run dir path from config, don't create
def get_processed_dir(config: RunConfig) -> Path:
    processed_dir = PROCESSED_DATA_DIR
    for part in config.folder_parts():
        processed_dir = processed_dir / part
    return processed_dir

# create processed dir if not exists
def ensure_processed_dir(config: RunConfig) -> Path:
    processed_dir = get_processed_dir(config)
    processed_dir.mkdir(parents=True, exist_ok=True)
    config_path = processed_dir / "run_config.json"

    # avoid overwriting
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != asdict(config):
            raise ValueError(
                "Existing processed-data folder config does not match requested config.\n"
                f"processed_dir={processed_dir}\n"
                f"existing={existing}\n"
                f"requested={asdict(config)}"
            )
    else:
        config_path.write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.info(f"Wrote processed-data config to {config_path}")
    return processed_dir

def find_latest_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    # extract epoch numbers
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("epoch_*.pt"):
        try:
            epoch = int(path.stem.split("_", 1)[1])
        except Exception:
            continue
        candidates.append((epoch, path))

    if not candidates:
        return None
    
    return max(candidates, key=lambda t: t[0]) # max epoch


def resolve_model_path_for_predict(run_dir: Path, prefer: str = "best") -> Path:
    if prefer == "best":
        path = run_dir / "best.pt"
        if path.exists():
            return path
    elif prefer == "final":
        path = run_dir / "final.pt"
        if path.exists():
            return path
    elif prefer == "latest":
        latest = find_latest_checkpoint(run_dir)
        if latest:
            return latest[1]
            
    # fallback priority
    for candidate in [run_dir / "best.pt", run_dir / "final.pt"]:
        if candidate.exists():
            return candidate
    latest = find_latest_checkpoint(run_dir)
    if latest:
        return latest[1]
    raise FileNotFoundError(f"No model weights found in run_dir={run_dir}")
