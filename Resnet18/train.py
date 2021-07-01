from __future__ import print_function
from models.litmodel import LitModel
from models.fcn32s import FCN32s

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG
from torch.optim.rmsprop import RMSprop


from dataloader import CustomDataset, lit_custom_data


import os
# os.environ["OPENBLAS_MAIN_FREE"] = '1'



def trainEpoch(model, loader):

    for index, batch in loader:
        optimizer.zero_grad()
        loss = model.training_step(batch, index)['loss']
        loss.backward()
        optimizer.step()

def valEpoch(model, loader):

    with torch.no_grad():
        for index, batch in loader:
            loss = model.training_step(batch, index)['loss']


if __name__ == '__main__':
    hparams = {
        'lr': 0.0019054607179632484
    }
    model = LitModel(hparams)

    dataset = lit_custom_data("/home/i1/lookingfastslow/lr_find_temp_model.ckpt")
    trainLoader = dataset.train_dataloader()
    valLoader = dataset.test_dataloader()
    optimizer = model.configure_optimizers()
    for epoch in range(120):
        trainEpoch(model, trainLoader)
        valEpoch(model,  valLoader)

