from typing import List, Tuple

import os
import torch
import gradio as gr
import torchvision.transforms as T
import click
import boto3

from aws_cred import access_key_id, secret_access_key

@click.command()
@click.option("--ckpt_path", help="torch script checkpoint path", default ='model.script.pt')
def demo(ckpt_path) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    model = torch.jit.load(ckpt_path)

    # log.info(f"Loaded Model: {model}")
    

    def recognize_digit(image):
        if image is None:
            return None
        image = T.ToTensor()(image).unsqueeze(0)
        # image = torch.tensor(image[None, ...], dtype=torch.float32)#.swapaxes(1, 3)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        labels = [
            "plane", 
            "automobile", 
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

    demo.launch(server_name="0.0.0.0", server_port=80, share=True)

def main() -> None:
    bucket_name = "model-bucket-emlo"
    filename = "model.script.pt"
    if not os.path.exists(filename):
        s3 = boto3.client('s3', aws_access_key_id=access_key_id , aws_secret_access_key=secret_access_key)
        s3.download_file(bucket_name, filename, filename)
    demo()

if __name__ == "__main__":
    main()