import os

import torch
import torch.optim as optim
from torch import Tensor

from NeuralStyleTransfer.loader import ImageLoader
from NeuralStyleTransfer.style_transfer import NeuralStyleTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor: Tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


def main():

    style_img, content_img = ImageLoader().load_images()
    input_img = content_img.clone()

    print('Building the style transfer model..')
    model = NeuralStyleTransfer()

    model.get_style_model_and_losses(style_img=style_img, content_img=content_img)

    result = model.fit_transform(input_img)

    plt.figure()
    imshow(result)

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()


if __name__ == '__main__':

    main()
