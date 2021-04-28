from models.convlstm import ConvLSTM
from models.fcn32s import FCN32s
from models.vgg import VGGNet
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

        self.vgg16 = VGGNet(model='vgg16')
        self.vgg11 = VGGNet(model='vgg11')
        self.extractors = [self.vgg11, self.vgg16]
        self.convlstm = ConvLSTM(512, 512, kernel_size=(
            3, 3), num_layers=1, batch_first=True)
        self.deconv = FCN32s(n_class=1)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, z):
        out = self.model.decode(z)
        return out

    def training_step(self, batch, index):
        # training_step defines the train loop. It is independent of forward.
        image = batch['input']
        target = batch['target']
        b, c, h, w = image.size()
        # print(image.size(), self.vgg11(image)['x5'].size())
        features = [
            torch.unsqueeze(self.vgg11(image)['x5'], dim=1),
            torch.unsqueeze(self.vgg16(image)['x5'], dim=1)
        ]

        b, _, c, h, w = features[0].size()
        # print(features[0].size())
        hidden = self.convlstm._init_hidden(batch_size=b, image_size=(h, w))
        for i in range(6):
            choose = np.random.randint(0, 2)
            prediction, hidden = self.convlstm(features[choose], hidden)
        maps = self.deconv(prediction[-1][:, 0])

        loss = F.binary_cross_entropy(maps, target)

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        dic = {'loss': loss}
        return dic

    def validation_step(self, batch, index):
        # training_step defines the train loop. It is independent of forward.
        image = batch['input']
        target = batch['target']
        b, c, h, w = image.size()
        # print(image.size(), self.vgg11(image)['x5'].size())
        features = [
            torch.unsqueeze(self.vgg11(image)['x5'], dim=1),
            torch.unsqueeze(self.vgg16(image)['x5'], dim=1)
        ]

        b, _, c, h, w = features[0].size()
        # print(features[0].size())
        hidden = self.convlstm._init_hidden(batch_size=b, image_size=(h, w))
        for i in range(6):
            # choose = np.random.randint(0, 2)
            prediction, hidden = self.convlstm(features[0], hidden)
        maps = self.deconv(prediction[-1][:, 0])

        loss = F.binary_cross_entropy(maps, target)

        outmap = torch.stack((maps, maps, maps), dim=1).float().squeeze()
        targetmap = torch.stack((target, target, target), dim=1).squeeze()

        n_rows = 5
        data = torch.cat((image.detach()[:n_rows], outmap.float()[
            :n_rows], targetmap.detach()[:n_rows]), dim=2)

        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_images(
            'validateImagesIndex0', data, self.current_epoch)

        dic = {'loss': loss}
        return dic
    #     image = batch['input']
    #     target = torch.cat((batch['target'], batch['depth']), dim = 1)

    #     intent = batch['intent']
    #     dist = batch['dist']

    #     out = self.model(image, intent, dist)

    #     loss = F.binary_cross_entropy(out['recon'], target)

    #     kld = -0.5 * torch.sum(1 + out['logvar'] - out['mu'].pow(2) - out['logvar'].exp())
    #     loss += kld

    #     self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #     if index == 0:
    #         outmap = out['recon']

    #         outmapdepth = outmap[:, 0, :, :].view(-1, 1, 128, 128)
    #         outmapseg = outmap[:, 1, :, :].view(-1, 1, 128, 128)

    #         targetmapseg = target[:, 0, :, :].view(-1, 1, 128, 128)
    #         targetmapdepth = target[:, 1, :, :].view(-1, 1, 128, 128)

    #         outmapdepth = torch.cat((outmapdepth, outmapdepth, outmapdepth), dim = 1).float().detach().cpu()
    #         outmapseg = torch.cat((outmapseg, outmapseg, outmapseg), dim = 1).float().detach().cpu()

    #         targetmapseg = torch.cat((targetmapseg, targetmapseg, targetmapseg), dim = 1).float().detach().cpu()
    #         targetmapdepth = torch.cat((targetmapdepth, targetmapdepth, targetmapdepth), dim = 1).float().detach().cpu()
    #         datatocat = (image.float().detach().cpu(), outmapdepth, targetmapdepth, outmapseg, targetmapseg)

    #         data = torch.cat((image.float().detach().cpu(), outmapdepth, targetmapdepth, outmapseg, targetmapseg), dim = 2)

    #         self.logger.experiment.add_images('valimages', data, self.current_epoch)

    #     dic = {'loss' : loss}

    #     return dic
