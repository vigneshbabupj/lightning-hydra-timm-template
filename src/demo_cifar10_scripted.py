import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
import torchvision.transforms as T

from src import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")
    

    def recognize_digit(image):
        if image is None:
            return None
        image = T.ToTensor()(image).unsqueeze(0)
        # image = torch.tensor(image[None, ...], dtype=torch.float32)#.swapaxes(1, 3)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        print(preds)
        labels = [
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
            ]
        return {labels[i]: preds[i] for i in range(10)}

    demo = gr.Interface(
        fn=recognize_digit,
        inputs=[gr.Image(shape=(32,32))],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_name="0.0.0.0", server_port=8080)

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_cifar10_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()