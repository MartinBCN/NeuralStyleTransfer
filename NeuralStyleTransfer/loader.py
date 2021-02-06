from __future__ import print_function

import os
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import Tensor


class ImageLoader(object):

    device = torch.device("cuda" if torch.cuda.is_available() and (not os.environ.get('USE_CPU', False)) else "cpu")

    def __init__(self, image_size: int = None):

        if image_size is None:
            # desired size of the output image
            image_size = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

        self.loader = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # scale imported image
                transforms.ToTensor()])  # transform it into a torch tensor

    def _single_image(self, image_name: str) -> Tensor:
        image = Image.open(image_name)
        return self.pil_to_tensor(image)

    def pil_to_tensor(self, image: Image) -> Tensor:

        print(image)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def load_images(self) -> (Tensor, Tensor):

        style_img = self._single_image("/home/martin/Programming/Python/NeuralStyleTransfer/data/picasso.jpg")
        content_img = self._single_image("/home/martin/Programming/Python/NeuralStyleTransfer/data/dancing.jpg")

        assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"

        return style_img, content_img

    def load_styles(self) -> (Tensor, Tensor, Tensor):
        data = Path(__file__).parents[1] / 'styles'

        filenames = {'Mona Lisa': 'monalisa', 'Picasso': 'picasso', 'Starry Night': 'starry'}
        styles = {k: self._single_image(f'{data}/{v}.jpg') for k, v in filenames.items()}

        return styles


if __name__ == '__main__':
    loader = ImageLoader()
    s, c = loader.load_images()

    print(s.shape)
    print(c.shape)

    style = loader.load_styles()

    for s in style.values():
        print(s.shape)
