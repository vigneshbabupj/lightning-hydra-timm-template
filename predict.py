import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import json
import hydra
import timm
import torch
import numpy as np
from PIL import Image
from typing import Any
from omegaconf import DictConfig
from hydra import compose, initialize
from cog import BasePredictor, Input, Path
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule



class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient."""
        # self.model = timm.create_model('efficientnet_b3a', pretrained=True)
        # self.model = model
        
        # global initialization
        initialize(version_base="1.2", config_path="configs/model")
        cfg = compose(config_name="timm")

        self.model: LightningModule = hydra.utils.instantiate(cfg)
        # model.load_from_checkpoint(checkpoint_path = "/workspace/Gitpod/lightning-hydra-template/logs/train/runs/2022-09-04_12-10-26/checkpoints/epoch_000.ckpt")
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
            ]
        )

        self.labels = (
            "plane", 
            "car", 
            "bird", 
            "cat",
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"
            )

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to classify")) -> Any:
    # def predict(self, image = "input.jpg") -> Any:
        """Run a single prediction on the model"""
        # Preprocess the image
        img = Image.open(image).convert('RGB')
        img = self.transform(img)

        # Run the prediction
        with torch.no_grad():
            labels = self.model(img[None, ...])
            labels = labels[0] # we'll only do this for one image

        # top 5 preds
        topk = labels.topk(5)[1]
        output = {
            # "labels": labels.cpu().numpy(),
            "topk": [self.labels[x] for x in topk.cpu().numpy().tolist()],
        }

        return output
    

