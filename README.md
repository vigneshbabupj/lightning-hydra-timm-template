<div align="center">

# lightning Hydra Timm Template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Pytorch lightning + Hydra + Timm model repo on Cifar10 dataset

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/vigneshbabupj/lightning-hydra-timm-template.git
cd lightning-hydra-timm-template

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Docker

Run as Docker image

```bash
## Build Docker container image
make build

## Run the container with experiment=example.yaml 
make run
```

## COG Inference

Install COG

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```
Grab an image to test run the inference on 
```bash
IMAGE_URL=https://gist.githubusercontent.com/bfirsh/3c2115692682ae260932a67d93fd94a8/raw/56b19f53f7643bb6c0b822c410c366c3a6244de2/mystery.jpg
curl $IMAGE_URL > input.jpg
```

Run inference

```bash
cog predict -i image=@input.jpg
```
returns top 5 predictions for the given input image

## Timm model

Train the model on any of models available on [timm library](https://github.com/rwightman/pytorch-image-models.git) 
by passing thr model name as below

```bash
python src/train.py model.net.name="resnet18"
```
or you can also change the timm.yaml to configure for the model to be run

```bash
  _target_: src.models.timm_module.TimmModel
  name: "resnet18"
  num_classes: 10
```

