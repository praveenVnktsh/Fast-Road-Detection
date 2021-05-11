from models.convlstm import ConvLSTM
from models.resnet import Resnet
from models.fcn32s import FCN32s
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
import numpy as np


class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.learning_rate = hparams['lr']

        self.resnet101 = Resnet(model='resnet101', pretrained = True, requires_grad = False)
        self.resnet101.load_state_dict(torch.load('state_dicts/101.pt'))
        self.resnet101.eval()

        self.resnet18 = Resnet(model='resnet18', pretrained = True, requires_grad = True)

        self.deconv = FCN32s(n_class=1)
        self.deconv.load_state_dict(torch.load('state_dicts/deconv.pt'))
        self.deconv.eval()
        for p in self.deconv.parameters():
            p.requires_grad = False

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.resnet18.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, z):

        features = self.resnet101(z)['x5']
        out = self.deconv(features)

        return out

    def training_step(self, batch, index):
        image = batch['input']
        target = batch['target']

        eighteen = self.resnet18(image)['x5']
        maps = self.deconv(eighteen)
        loss = F.binary_cross_entropy(maps, target)

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        dic = {'loss': loss}
        return dic

    def validation_step(self, batch, index):

        image = batch['input']
        target = batch['target']
       
        with torch.no_grad():
            eighteen = self.resnet18(image)['x5']
            # one_o_one = self.resnet101(image)['x5']
            maps = self.deconv(eighteen)

        loss = F.binary_cross_entropy(maps, target)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        dic = {'loss': loss}
        return dic

    def test_step(self, batch, batch_idx):
        image = batch['input']
        target = batch['target']
        b, c, h, w = image.size()

        choose = np.random.randint(0, 2)
        if choose:
            features = torch.unsqueeze(self.resnet18(image)['x5'], dim=1)
        else:
            features = torch.unsqueeze(self.resnet101(image)['x5'], dim=1)

        prediction, self.hidden = self.convlstm(features, self.hidden)
        maps = self.deconv(prediction[-1][:, 0])

        return {"out": maps, "target": target}
