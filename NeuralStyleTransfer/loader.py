from __future__ import print_function

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import Tensor


class ImageLoader(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # desired size of the output image
        image_size = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

        self.loader = transforms.Compose([
                transforms.Resize(image_size),  # scale imported image
                transforms.ToTensor()])  # transform it into a torch tensor

    def _single_image(self, image_name: str) -> Tensor:
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def load_images(self) -> (Tensor, Tensor):

        style_img = self._single_image("/home/martin/Programming/Python/NeuralStyleTransfer/data/picasso.jpg")
        content_img = self._single_image("/home/martin/Programming/Python/NeuralStyleTransfer/data/dancing.jpg")

        assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"

        return style_img, content_img


if __name__ == '__main__':
    s, c = ImageLoader().load_images()

    print(s.shape)
    print(c.shape)
