import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, t: Tensor) -> Tensor:
        self.loss = F.mse_loss(t, self.target)
        return t


if __name__ == '__main__':
    test_target = torch.rand(1, 128, 128, 3)
    content_loss = ContentLoss(test_target)
    test_input = torch.rand(1, 128, 128, 3)
    content_loss(test_input)

    print(content_loss.loss)
