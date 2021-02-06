from __future__ import print_function

import os
from typing import List

import torch
import torch.nn as nn
import torchvision.models as models

from torch import Tensor, optim

from NeuralStyleTransfer.content_loss import ContentLoss
from NeuralStyleTransfer.normalisation import Normalization
from NeuralStyleTransfer.style_loss import StyleLoss


def get_input_optimizer(input_img: Tensor):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


class NeuralStyleTransfer:
    device = torch.device("cuda" if torch.cuda.is_available() and (not os.environ.get('USE_CPU', False)) else "cpu")

    cnn = models.vgg19(pretrained=False)
    p = os.environ.get('MODEL_PATH', '/home/martin/Programming/Python/NeuralStyleTransfer/model/vgg19-dcbb9e9d.pth')
    cnn.load_state_dict(torch.load(p))

    cnn = cnn.features.to(device).eval()
    normalization = Normalization().to(device)

    def __init__(self, content_layers: List[str] = None, style_layers: List[str] = None,
                 num_steps: int = 10):
        self.num_steps = num_steps

        # desired depth layers to compute style/content losses :
        if content_layers is None:
            self.content_layers = ['conv_4']
        else:
            self.content_layers = content_layers

        if content_layers is None:
            self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        else:
            self.style_layers = style_layers

        self.style_losses = []
        self.content_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        self.model = nn.Sequential(self.normalization)
        self.model = self.model.to(self.device)

    def __call__(self, t: Tensor):
        return self.model(t)

    def get_style_model_and_losses(self, style_img: Tensor, content_img: Tensor) -> None:

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            self.model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = self.model(content_img).detach()
                content_loss = ContentLoss(target)
                self.model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = self.model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)]
        self.model = self.model.to(self.device)

        self.style_losses = style_losses
        self.content_losses = content_losses

    def fit_transform(self, input_img: Tensor) -> Tensor:
        optimizer = get_input_optimizer(input_img)

        style_weight = 1000000
        content_weight = 1

        print('Optimizing..')
        for i in range(self.num_steps):

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                self(input_img)
                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                if i % 1 == 0:
                    print(f"run {i}:")
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        print(input_img)

        return input_img


if __name__ == '__main__':
    nst = NeuralStyleTransfer()
