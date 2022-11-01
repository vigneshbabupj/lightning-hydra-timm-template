import os
import requests
import subprocess

import pytest
import unittest
# import sys
# sys.path.insert(0, "../../serve/ts_scripts/")
# print(sys.path[0])

## gRPC
# from lightning-hydra-timm-template.serve.ts_scripts.torchserve_grpc_client import infer

import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image

from captum.attr import visualization as viz

@pytest.fixture(scope="class")
def get_param(request):
    request.cls.ip = "localhost"
    request.cls.model_name="cifar10"
    request.cls.model_ver="1.0"

@pytest.mark.usefixtures("get_param")
class TestCifar10Serve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images_path= "/workspace/lightning-hydra-timm-template/tests/cifar10_test_images"
    
    def test_predict(self):
        predict_url = f'http://{self.ip}:8080/predictions/{self.model_name}/{self.model_ver}'
        for img in os.listdir(self.images_path):
            res = requests.post(predict_url, files={'data': open(os.path.join(self.images_path,img), 'rb')})
            print(res.json())
            pred = list(res.json().keys())[0]
            print(f'{img=} {pred=}')
            assert os.path.splitext(img)[0].split('_')[1] == pred

    def test_gRPC(self):
        grpc_script_path="/workspace/lightning-hydra-timm-template/serve/ts_scripts/torchserve_grpc_client.py"
        for img in os.listdir(self.images_path):
            cmd = f'python {grpc_script_path} infer {self.model_name} {os.path.join(self.images_path,img)}'
            res = eval(subprocess.check_output(cmd, shell=True))
            pred = list(res.keys())[0]
            assert os.path.splitext(img)[0].split('_')[1] == pred

    def test_explanations(self):
        explain_url = f'http://{self.ip}:8080/explanations/{self.model_name}/{self.model_ver}'
        img = os.listdir(self.images_path)[0]
        res = requests.post(explain_url, files={'data': open(os.path.join(self.images_path,img), 'rb')})
        ig = res.json()
        inp_image = Image.open(os.path.join(self.images_path,img))
        to_tensor = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor()
        ])
        inp_image = to_tensor(inp_image)

        inp_image = inp_image.numpy()
        attributions = np.array(ig)

        inp_image, attributions = inp_image.transpose(1, 2, 0), attributions.transpose(1, 2, 0)

        assert inp_image.shape == attributions.shape

if __name__ == '__main__':
    unittest.main()