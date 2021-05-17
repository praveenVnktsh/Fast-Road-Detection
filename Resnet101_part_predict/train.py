from __future__ import print_function
from models.litmodel import LitModel
from models.fcn32s import FCN32s

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG
import pytorch_lightning as pl
from torch.optim.rmsprop import RMSprop


from dataloader import lit_custom_data
from pytorch_lightning import loggers
from configs import Configs


import os
if __name__ == '__main__':
    hparams = {
        'lr': 0.0019054607179632484
    }
    model = LitModel(hparams)

    dataset = lit_custom_data()
    dataset.setup()

    trainer = pl.Trainer(gpus=1, max_epochs=120)
    # trainer.tune(model, dataset)
    trainer.fit(model, dataset)
    print("hello")
