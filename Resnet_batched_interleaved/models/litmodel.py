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
    '''
    Batched Interleaved Model.
    We interleave two resnet feature extractors to detect features,
    these features are stored in the temporal memory of the convlstm.
    '''

    def __init__(self, hparams):
        super().__init__()
        self.learning_rate = hparams['lr']
        self.hidden = None
        self.index = 0

        # Resnet Feature Extractors
        self.resnet101 = Resnet(model='resnet101')
        self.resnet18 = Resnet(model='resnet18')
        self.extractors = [self.resnet18, self.resnet101]

        # Conv LSTM
        self.convlstm = ConvLSTM(512, 128, kernel_size=(
            3, 3), num_layers=1, batch_first=True)

        # Decoder Network
        self.deconv = FCN32s(n_class=1)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, z, clear=False, choose=0):
        if choose == 0:
            self.hidden = None
            features = torch.unsqueeze(self.resnet101(z)['x5'], dim=1)
            prediction, self.hidden = self.convlstm(features, self.hidden)
        features = torch.unsqueeze(self.resnet18(z)['x5'], dim=1)

        prediction, self.hidden = self.convlstm(features, self.hidden)
        out = self.deconv(prediction[-1][:, 0])
        self.index += 1

        return out

    def training_step(self, batch, index):
        # training_step defines the train loop. It is independent of forward.
        image = batch['input']
        target = batch['target']
        b, c, h, w = image.size()
        features = [
            torch.unsqueeze(self.resnet18(image)['x5'], dim=1),
            torch.unsqueeze(self.resnet101(image)['x5'], dim=1)
        ]

        b, _, c, h, w = features[0].size()

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
        features = [
            torch.unsqueeze(self.resnet18(image)['x5'], dim=1),
            torch.unsqueeze(self.resnet101(image)['x5'], dim=1)
        ]

        b, _, c, h, w = features[0].size()
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

    def test_step(self, batch, batch_idx):
        # test step defines the train loop. It is independent of forward.
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
