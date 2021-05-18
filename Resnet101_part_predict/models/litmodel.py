from configs import Configs
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

        self.resnet = Resnet()

        self.convlstm = ConvLSTM(1024, 128, kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.deconv = FCN32s(n_class=1)

        self.configs = Configs()

        self.hidden = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, z,):

        features = self.resnet(z)
        out = self.deconv(features)
        return out

    def runEpoch(self, batch, index):
        image = batch['input']
        target = batch['target']

        b = image.size(0)
        sLength = 6
        c = 512
        h = 5
        w = 5
        hidden = self.convlstm._init_hidden(batch_size=b, image_size=(h, w))
        loss = 0
        for i in range(sLength):

            features = self.resnet(image[:, i])

            if i == 0:
                currentLargeFeature = features['final']

            currentSmallFeature = features['mid']

            inputFeature = torch.unsqueeze(torch.cat((currentSmallFeature, currentLargeFeature), dim = 1), dim = 1)

            prediction, hidden = self.convlstm(inputFeature, hidden)

            maps = self.deconv(prediction[-1][:, 0])

            loss += F.binary_cross_entropy(maps, target[:, i])
        return loss, maps, target[:, i], image

    def training_step(self, batch, index):

        loss, maps, target, image = self.runEpoch(batch, index)

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        dic = {'loss': loss}
        return dic

    def validation_step(self, batch, index):
        # training_step defines the train loop. It is independent of forward.
        loss, maps, target, image = self.runEpoch(batch, index)

        outmap = torch.stack((maps, maps, maps), dim=1).float().squeeze()
        targetmap = torch.stack((target, target, target), dim=1).squeeze()

        n_rows = 5
        data = torch.cat((image[:, -1].detach()[:n_rows], outmap.float()[
            :n_rows], targetmap.detach()[:n_rows]), dim=2)

        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_images(
            'validateImagesIndex0', data, self.current_epoch)

        dic = {'loss': loss}
        return dic

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward.
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
