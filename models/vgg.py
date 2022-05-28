from typing import List

import torch
from torch import nn
from torchvision.models import vgg19


class VGG19Model(nn.Module):

    def __init__(self):
        super(VGG19Model, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features

        self.features1 = torch.nn.Sequential()
        self.features2 = torch.nn.Sequential()
        self.features3 = torch.nn.Sequential()
        self.features4 = torch.nn.Sequential()
        self.features5 = torch.nn.Sequential()
        self.features6 = torch.nn.Sequential()
        for x in range(2):
            self.features1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.features2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.features3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.features4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 22):
            self.features5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 30):
            self.features6.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        fe1 = self.features1(x)
        fe2 = self.features2(fe1)
        fe3 = self.features3(fe2)
        fe4 = self.features4(fe3)
        fe5 = self.features5(fe4)
        fe6 = self.features6(fe5)
        out = [fe1, fe2, fe3, fe4, fe5, fe6]
        return out
