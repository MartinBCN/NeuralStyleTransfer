import os

import torch
import torch.nn as nn


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    """
    VGG networks are trained on images with each channel normalized by
    mean=[0.485, 0.456, 0.406] and
    std=[0.229, 0.224, 0.225].
    We will use them to normalize the image before sending it into the network.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and (not os.environ.get('USE_CPU', False)) else "cpu")
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(self.cnn_normalization_mean).view(-1, 1, 1)
        self.std = torch.tensor(self.cnn_normalization_std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
