import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, t: Tensor):
        G = self.gram_matrix(t)
        self.loss = F.mse_loss(G, self.target)
        return t

    @staticmethod
    def gram_matrix(t: Tensor):
        a, b, c, d = t.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = t.view(a * b, c * d)  # resize F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


if __name__ == '__main__':
    test_target = torch.rand(1, 128, 128, 3)
    style_loss = StyleLoss(test_target)
    test_input = torch.rand(1, 128, 128, 3)
    style_loss(test_input)

    print(style_loss.loss)
