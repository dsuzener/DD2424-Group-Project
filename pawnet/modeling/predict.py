from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from pawnet.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Performing inference for model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Inference complete.")
    # # -----------------------------------------
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.eval()

    preprocess = weights.transforms()


    img = Image.open("data/raw/oxford-iiit-pet/images/Siamese_71.jpg").convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    probs = torch.nn.functional.softmax(output[0], dim=0)

    class_id = int(probs.argmax().item())
    label = weights.meta["categories"][class_id]
    score = probs[class_id].item()

    print(label, score)


if __name__ == "__main__":
    app()
