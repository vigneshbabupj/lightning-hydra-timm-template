import os
import timm
import urllib
import torch

import numpy as np

import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

## XAI Imports
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion

from captum.attr import Saliency

## gradcam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


device = torch.device("cuda")
model = timm.create_model("resnet18", pretrained=True)
model.eval()
model = model.to(device)

# Download human-readable labels for ImageNet.
# get the classnames
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
)
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

def get_prediction(imagepath):
  transform = T.Compose([
      T.Resize((244,244)),
      # T.CenterCrop(224),
      T.ToTensor()
  ])

  transform_normalize = T.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
  )

  img = Image.open(imagepath)

  transformed_img = transform(img)

  img_tensor = transform_normalize(transformed_img)
  img_tensor = img_tensor.unsqueeze(0)
  img_tensor = img_tensor.to(device)
  output = model(img_tensor)
  output = F.softmax(output, dim=1)
  prediction_score, pred_label_idx = torch.topk(output, 1)

  pred_label_idx.squeeze_()
  predicted_label = categories[pred_label_idx.item()]
  print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

  return (transformed_img,img_tensor,pred_label_idx)

def integratedgradients(result_set):
    transformed_img,img_tensor,pred_label_idx = result_set[1]
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(img_tensor, target= pred_label_idx, n_steps=100)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)

    outviz = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                method = 'heat_map',
                                cmap = default_cmap,
                                show_colorbar=True,
                                title = result_set[0],
                                sign='positive',
                                outlier_perc=1)
    return outviz


def noisetunnel(result_set):
    transformed_img,img_tensor,pred_label_idx = result_set[1]

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(img_tensor, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
    outviz = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            # ["original_image", "heat_map"],
                                            # ["all", "positive"],
                                            ["heat_map"],
                                            ["positive"],
                                            cmap=default_cmap,
                                            show_colorbar=True)
    return outviz

def shap(result_set):
    transformed_img,img_tensor,pred_label_idx = result_set[1]

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])

    attributions_gs = gradient_shap.attribute(img_tensor,
                                                n_samples=50,
                                                stdevs=0.0001,
                                                baselines=rand_img_dist,
                                                target=pred_label_idx)
    
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
    
    outviz = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["heat_map"],
                                            ["absolute_value"],
                                            cmap=default_cmap,
                                            titles = [result_set[0]],
                                            show_colorbar=True)
    return outviz

def occlusion(result_set):
    transformed_img,img_tensor,pred_label_idx = result_set[1]

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(img_tensor,
                                            strides = (3, 8, 8),
                                            target=pred_label_idx,
                                            sliding_window_shapes=(3,15, 15),
                                            baselines=0)

    outviz = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["heat_map"],
                                            ["positive"],
                                            titles = [result_set[0]],
                                            show_colorbar=True,
                                            outlier_perc=2,
                                        )
    return outviz

def saliency(imagename):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(f'test_images/{imagename}')

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)

    saliency = Saliency(model)
    grads = saliency.attribute(img_tensor, target=int(imagename.split('_')[-1].split('.')[0]))
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    original_image = np.transpose((img_tensor.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    outviz = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                            show_colorbar=True, title=imagename)
    
    return outviz
  


def gradcam(imagename):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(f'test_images/{imagename}')

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    inv_transform= T.Compose([
        T.Normalize(
            mean = (-1 * np.array(mean) / np.array(std)).tolist(),
            std = (1 / np.array(std)).tolist()
        ),
    ])

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(int(imagename.split('_')[-1].split('.')[0]))] #imagename.split('_')[-1].split('.')[0]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # plt.imshow(visualization)

    ## GradCam++
    cam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam_pp = cam(input_tensor=img_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam_pp = grayscale_cam[0, :]
    rgb_img_pp = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization_pp = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # plt.imshow(visualization_pp)
    return visualization, visualization_pp


def main():
    results = {}
    for imgs in os.listdir("test_images"):
        print(imgs)
        results[imgs] = get_prediction(f"test_images/{imgs}")
    
    for res in results.items():
        f = integratedgradients(res)
        f[0].savefig(f'ig/{res[0]}', bbox_inches='tight')

        f = noisetunnel(res)
        f[0].savefig(f'nt/{res[0]}', bbox_inches='tight')

        f = shap(res)
        f[0].savefig(f'shap/{res[0]}', bbox_inches='tight')
        
        f = occlusion(res)
        f[0].savefig(f'occlusion/{res[0]}', bbox_inches='tight')

    for res in os.listdir('test_images'):
        print(res)
        f = saliency(res)
        f[0].savefig(f'sal/{res}', bbox_inches='tight')

        gc,gc_pp = gradcam(res)
        plt.imshow(gc).figure.savefig(f'gc/{res}', bbox_inches='tight')
        plt.imshow(gc_pp).figure.savefig(f'gcpp/{res}', bbox_inches='tight')