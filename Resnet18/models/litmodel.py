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

        self.resnet18 = Resnet(model='resnet18')
        self.convlstm = ConvLSTM(512, 128, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.deconv = FCN32s(n_class=1, in_channel= 128)
        self.hidden = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, z,):

        features = self.resnet18(z)['x5']
        out = self.deconv(features)

        return out

    def runEpoch(self, batch, index):
        image = batch['input']
        target = batch['target']
        b, s, c, h, w = image.size()

        hidden = self.convlstm._init_hidden(batch_size=b, image_size=(9, 11))
        for i in range(6):
            features =  torch.unsqueeze(self.resnet18(image[:, i])['x5'], dim=1)
            prediction, hidden = self.convlstm(features, hidden)
            
        maps = self.deconv(prediction[-1][:, 0])

        loss = F.binary_cross_entropy(maps, target[:, -1], reduction='mean')

        dic = {'loss': loss, 'maps' : maps}
        return dic

    def training_step(self, batch, index):
        # training_step defines the train loop. It is independent of forward.
        dic = self.runEpoch(batch, index)

        self.log('train_loss', dic['loss'], on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        dic = {'loss': dic['loss']}
        return dic

    def validation_step(self, batch, index):
        # training_step defines the train loop. It is independent of forward.
        dic = self.runEpoch(batch, index)
        maps = dic['maps']
        target = batch['target']
        image = batch['input']
        n_rows = 5

        outmap = torch.stack((maps, maps, maps), dim=1).float().squeeze()
        targetmap = torch.stack((target[:, -1], target[:, -1], target[:, -1]), dim=1).squeeze()

        data = torch.cat((image[:, -1].detach()[:n_rows].squeeze(0), outmap.float()[
            :n_rows], targetmap.detach()[:n_rows]), dim=2)

        self.log('val_loss', dic['loss'], on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_images(
            'validateImagesIndex0', data, self.current_epoch)

        dic = {'loss':  dic['loss']}
        return dic
