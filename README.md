<div align="center">

# lightning Hydra Timm Template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

<div align="center">

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/vigneshbabupj/lightning-hydra-timm-template)

</div>

## Description

Pytorch lightning + Hydra + Timm model repo on Cifar10 dataset

## How to run - train

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

## Train using Docker

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

## Demo App

Demo based on Gradio app can be run from the torch script checkpoint

```bash
#cifar10
python3 src/demo_cifar10_scripted.py -m ckpt_path=<path to model.script.pt>

#mnist
python3 src/demo_scripted.py -m ckpt_path=<path to model.script.pt>

```
## Demo using Docker (Cifar10)

Docker image: [vigneshbabuph/torchscript_gradio_demo](https://hub.docker.com/r/vigneshbabupj/torchscript_gradio_demo/tags)

Cd into the demo folder

```bash
cd /Docker/demo/

#Build image locally or pull from docker hub

#Buid image locally
#docker image build -t vigneshbabupj/torchscript_gradio_demo .

#pull from docker hub
docker pull vigneshbabupj/torchscript_gradio_demo:latest

#run demo from docker image
docker run --rm --volume `pwd`:/opt/src -it -p 8080:8080 vigneshbabupj/torchscript_gradio_demo:latest

```
place the model checkpoint to be used under the demo folder as "model.script.pt" or pass the ckpt_path as argument

Gradio server with the demo app is launched, click on the url to access and use it

## Demo App - Deploy in AWS

cd into `Docker/aws_demo/`

- upload trained model to AWS S3 bucket and update `bucket_name` and `filename` in `s3_cifar10_docker.py`
- create `aws_cred.py` with AWS access configuration keys (access_key_id & secret_access_key)
- Launch an ec2 instance with an IAM role to access ECR and S3
- Upload the files in `aws_demo` to the instance and install aws cli, docker
- Create an ECR repo and build & upload the docker image using the push commands
- Create an ECS FARGATE cluster and task definition to run as a service
- Deploy
- Stop and Delete All the started services to close

## Distributed Training

Code Changes for distributed training and FSDP : [repo](https://github.com/vigneshbabupj/FairScale_distributed_training)

## Model Explainability

Refer to [Explainability.md](log_book\explainability\Explainability.md)

Code 
- Explainabiliy : explainability\explain.py
- Robustness : explainability\robustness.py

## Model Serving

Refer to [torchserve.md](log_book\model_serve\torchserve.md)

Model handlers: src\torch_handlers
Create .mar and serve model
```bash
torch-model-archiver --model-name cifar10_basic --version 1.0 --serialized-file /workspace/lightning-hydra-timm-template/logs/train/runs/2022-11-01_06-55-34/model.script.pt --handler /workspace/lightning-hydra-timm-template/src/torch_handlers/cifar10_handler.py --extra-files /workspace/lightning-hydra-timm-template/src/torch_handlers/cifar10_classes/index_to_name.json

docker run -it --rm --net=host -v `pwd`:/opt/src pytorch/torchserve:latest bash
cd /opt/src

torchserve --start --model-store model_store --models cifar10=cifar10_basic.mar

torchserve --stop

```

Run test scripts for cifar10 model serve : tests\test_serve
```bash

pytest tests/torch_serve

```
