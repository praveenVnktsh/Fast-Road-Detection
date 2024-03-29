from __future__ import print_function
from models.litmodel import LitModel
from models.fcn32s import FCN32s
from models.vgg import VGGNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG
import pytorch_lightning as pl
from torch.optim.rmsprop import RMSprop


from dataloader import CustomDataset, lit_custom_data
from pytorch_lightning import loggers
from configs import Configs


hparams = {
    'lr': 0.01
}
model = LitModel(hparams)

dataset = lit_custom_data()

trainer = pl.Trainer(gpus=1)
trainer.fit(model, dataset)
