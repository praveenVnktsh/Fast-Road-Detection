from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.optim.rmsprop import RMSprop


class Resnet(nn.Module):
    def __init__(self, pretrained=True, requires_grad=False, show_params=False):
        super(Resnet, self).__init__()

        model = models.resnet101(pretrained=pretrained)
        modules = list(model.children())[:-3]
        self.base = nn.Sequential(*modules)
        modules = list(model.children())[-3:-2]
        
        self.reducesize = nn.Conv2d(1024, 2048, 3, 2, 1)
        self.final = nn.Sequential(*modules)

        self.finalReduction = nn.Conv2d(2048, 512, 3, 1, 1)


        for p in self.final.parameters():
            p.requires_grad = requires_grad
        for p in self.base.parameters():
            p.requires_grad = requires_grad


        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        temp = self.base(x)
        output['final'] = self.finalReduction(self.final(temp))

        output['mid'] = self.finalReduction(self.reducesize(temp))

        return output




if __name__ == "__main__":
    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return round(pp/1e6, 1)

    resnet50_full = Resnet()
    
    x = torch.randn(1, 3, 160, 160)
    # print('full', get_n_params(resnet50_full))
    print(resnet50_full(x)['final'].size())
    print(resnet50_full(x)['mid'].size())
    # print(resnet101(x)['x5'].size())
