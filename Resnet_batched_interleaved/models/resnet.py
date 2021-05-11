from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.optim.rmsprop import RMSprop


class Resnet(nn.Module):
    '''
    Module to make a pretrained Resnet Feature Extractor.
    '''

    def __init__(self, pretrained=True, model='resnet18', requires_grad=False, show_params=False):
        super(Resnet, self).__init__()
        self.model = None

        if model == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)

        # We remove the two end layers of the resnet classifier in order to use the feature extractor
        modules = list(self.model.children())[:-2]
        self.model = nn.Sequential(*modules)

        # In case to finetune network
        for p in self.model.parameters():
            p.requires_grad = requires_grad

        # Adding a 2048-512 conv layer in order to match size.
        if model == 'resnet101':
            self.model = nn.Sequential(
                self.model,
                nn.Conv2d(2048, 512, 3, 1, 1)
            )

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        output["x5"] = self.model(x)
        return output


if __name__ == "__main__":
    resnet18 = Resnet(model='resnet18')
    resnet101 = Resnet(model='resnet101')

    x = torch.randn(1, 3, 160, 160)
    print(resnet18)
    print(resnet101)
    print(resnet18(x)['x5'].size())
    print(resnet101(x)['x5'].size())
