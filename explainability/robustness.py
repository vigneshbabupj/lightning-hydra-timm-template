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


from captum.robust import PGD

device = torch.device("cuda")
model = timm.create_model("resnet18", pretrained=True)
model.eval()
model = model.to(device)


def get_prediction(model, image: torch.Tensor):
    model = model.to(device)
    img_tensor = image.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]

    return predicted_label, prediction_score.squeeze().item()

def pgd(imagename):
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
  def image_show(img, pred):
    npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % pred)
    plt.savefig(f'pgd/{imagename}',bbox_inches='tight')
    plt.show()

  pgd = PGD(model, torch.nn.CrossEntropyLoss(reduction='none'), lower_bound=-1, upper_bound=1)  # construct the PGD attacker

  perturbed_image_pgd = pgd.perturb(inputs=img_tensor, radius=0.13, step_size=0.02, 
                                    step_num=7, target=torch.tensor([285]).to(device), targeted=True) 
  new_pred_pgd, score_pgd = get_prediction(model, perturbed_image_pgd)
  image_show(perturbed_image_pgd.cpu(), new_pred_pgd + " " + str(score_pgd))


## Pixel Dropout
from captum.attr import FeatureAblation
from captum.robust import MinParamPerturbation

def robust_pixel_dropout(imagename):
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
    def image_show(img, pred):
        npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
        plt.imshow(npimg)
        plt.title("pred: %s" % pred)
        plt.savefig(f'pd2/{imagename}',bbox_inches='tight')
        plt.show()

    ablator = FeatureAblation(model)
    # attr = ablator.attribute(img_tensor, target=285, feature_mask=feature_mask)
    attr = ablator.attribute(img_tensor, target=int(imagename.split('_')[-1].split('.')[0]))
    # Choose single channel, all channels have same attribution scores
    pixel_attr = attr[:,0:1]

    def pixel_dropout(image, dropout_pixels):
        keep_pixels = image[0][0].numel() - int(dropout_pixels)
        vals, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
        return (pixel_attr < vals.item()) * image

    min_pert_attr = MinParamPerturbation(forward_func=model, attack=pixel_dropout, arg_name="dropout_pixels", mode="linear",
                                        arg_min=0, arg_max=1024, arg_step=16,
                                        preproc_fn=None, apply_before_preproc=True)

    pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(img_tensor, target=int(imagename.split('_')[-1].split('.')[0]), perturbations_per_eval=10)
    print("Minimum Pixels Dropped:", pixels_dropped)

    new_pred_dropout, score_dropout = get_prediction(model, pixel_dropout_im)

    image_show(pixel_dropout_im.cpu(), new_pred_dropout + " " + str(round(score_dropout,1)) + "pixel drop:" + str(pixels_dropped))


from captum.robust import FGSM

def fgsm_robust(imagename):
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

    def image_show(img, pred):
        npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
        plt.imshow(npimg)
        plt.title("prediction: %s" % pred)
        plt.savefig(f'fgsm/{imagename}',bbox_inches='tight')
        plt.show()

    # Construct FGSM attacker
    fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
    perturbed_image_fgsm = fgsm.perturb(img_tensor, epsilon=0.16, target=int(imagename.split('_')[-1].split('.')[0]))
    new_pred_fgsm, score_fgsm = get_prediction(model, perturbed_image_fgsm)


    image_show(perturbed_image_fgsm.cpu(), new_pred_fgsm + " " + str(score_fgsm))


## Random Noise

def robust_random_noise(imagename):
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
    def image_show(img, pred):
        npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
        plt.imshow(npimg)
        plt.title("prediction: %s" % pred)
        plt.savefig(f'noise/{imagename}',bbox_inches='tight')
        plt.show()

    # ablator = FeatureAblation(model)
    # # attr = ablator.attribute(img_tensor, target=285, feature_mask=feature_mask)
    # attr = ablator.attribute(img_tensor, target=int(imagename.split('_')[-1].split('.')[0]))
    # # Choose single channel, all channels have same attribution scores
    # pixel_attr = attr[:,0:1]

    def gaussian_noise(inp, std):
        return inp + std*torch.randn_like(inp)

    min_pert_attr = MinParamPerturbation(forward_func=model, attack=gaussian_noise, arg_name="std", mode="linear",
                                        arg_min=0, arg_max=1, arg_step=0.1,
                                        preproc_fn=None, apply_before_preproc=True)

    noised_image_im, min_std  = min_pert_attr.evaluate(img_tensor, target=int(imagename.split('_')[-1].split('.')[0]), perturbations_per_eval=10)
    print("Minimum noise added:", min_std)

    new_pred_dropout, score_dropout = get_prediction(model, noised_image_im)

    image_show(noised_image_im.cpu(), new_pred_dropout + " " + str(score_dropout) + " Noise added:" + str(min_std))


## Random brightness

def robust_random_brightness(imagename):
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
    def image_show(img, pred):
        npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
        plt.imshow(npimg)
        plt.title("prediction: %s" % pred)
        plt.savefig(f'bt/{imagename}',bbox_inches='tight')
        plt.show()

    # ablator = FeatureAblation(model)
    # # attr = ablator.attribute(img_tensor, target=285, feature_mask=feature_mask)
    # attr = ablator.attribute(img_tensor, target=285)
    # # Choose single channel, all channels have same attribution scores
    # pixel_attr = attr[:,0:1]

    def adjust_bright(img1, ratio):
        img2 = torch.zeros_like(img1)
        ratio = float(ratio)
        bound = 1.0 if img1.is_floating_point() else 255.0
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


    min_pert_attr = MinParamPerturbation(forward_func=model, attack=adjust_bright, arg_name="ratio", mode="linear",
                                        arg_min=0, arg_max=2, arg_step=0.2,
                                        preproc_fn=None, apply_before_preproc=True)

    adjust_bright_im, min_std  = min_pert_attr.evaluate(img_tensor, target=285, perturbations_per_eval=10)
    print("Minimum brightness added:", min_std )

    new_pred_dropout, score = get_prediction(model, adjust_bright_im)

    image_show(adjust_bright_im.cpu(), new_pred_dropout + " " + str(score)+" brightness:" + str(min_std))



def main():
    for res in os.listdir('test_images'):
        print(res)
        pgd(res)

        robust_pixel_dropout(res)

        fgsm_robust(res)

        robust_random_noise(res)

        robust_random_brightness(res)

        


