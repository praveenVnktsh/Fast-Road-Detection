from convlstm import ConvLSTM
from configs import Configs
import torch
import torch.nn as nn
import os
from torch.optim import RMSprop
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn


class TimeDistributed(nn.Module):
    '''
    A pytorch module to make any module time distributed. This model adds a dimension for time.
    '''

    def __init__(self, module, batch_first=False):
        '''
        module: Pytorch model to be TimeDistributed
        batchfirst: To specify if the batch dimension is first or not
        '''
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(2), x.size(3), x.size(4)
                                        )  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        y = y.contiguous().view(x.size(0), x.size(1), y.size(1), y.size(2),
                                y.size(3))  # (samples, timesteps, output_size)

        return y


class ConvBlock(nn.Module):
    '''
    Simple Convolution Block with Batch Norm and Relu6 Activation
    '''

    def __init__(self, in_size, out_size, kernel, stride, padding):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.net(x)


class ConvTBlock(nn.Module):
    '''
    Simple Transposed Convolution Block with Batch Norm and Relu6 Activation
    '''

    def __init__(self, in_size, out_size, kernel, stride, padding):
        super(ConvTBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel,
                               stride=stride, padding=padding,  bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.net(x)


class FCN(nn.Module):
    '''
    FCN Model with the ConvLSTM block in between the feature extraction network and the Decoder Network
    '''

    def __init__(self, configs: Configs):
        super(FCN, self).__init__()

        self.configs = configs

        # Time Distributed encoder for n sequential images
        self.encoder = TimeDistributed(nn.Sequential(
            ConvBlock(3, 16, 5, 3, 0),
            ConvBlock(16, 32, 5, 3, 1),
            ConvBlock(32, 64, 3, 1, 0),
        ))

        # Convolultional LSTM
        self.lstm = ConvLSTM(64, 64, kernel_size=(
            3, 3), num_layers=1, batch_first=True)

        # Decoder
        self.decoder = nn.Sequential(
            ConvTBlock(64, 32, 3, 1, 0),
            ConvTBlock(32, 16, 5, 3, 1),
            ConvTBlock(16, 3, 5, 3, 0),
            ConvTBlock(3, 3, 3, 1, 0),
        )

        # Predicting the segmentation mask
        self.final = nn.Sequential(
            nn.Conv2d(3, 2, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        # All N sequential images are passed through the Envoder
        out = self.encoder(x)
        outputs, _ = self.lstm(out)  # states for attention

        # Only the last image is taken for the Decoder
        out = self.decoder(outputs[-1][:, 0])

        # Getting the segmentation mask
        out = self.final(out)

        return out

    def save(self, optimizer: RMSprop, epoch, train_loss, val_loss):
        '''
        To save the model correctly
        '''
        dic = {
            'epoch':  epoch,
            'train_loss':  train_loss,
            'val_loss':  val_loss,
            'model':  self.state_dict(),
            'optim':  optimizer.state_dict()
        }
        path = self.configs.saveDir + 'models/model_epoch_%d.pth' % epoch
        torch.save(dic, path)
        print('Saved Checkpoint at', path)

    def load(self, startEpoch):
        """
        Load checkpoint from given path
        """

        loadPath = self.configs.saveDir + 'models/model_epoch_%d.pth' % startEpoch
        state = torch.load(loadPath)
        print('Loading Checkpoint at', loadPath)
        self.load_state_dict(state['model'])

        return state


if __name__ == '__main__':
    configs = Configs()
    m = FCN(configs).cuda()
    inp = torch.randn(2, 2, 3, 128, 128).cuda()
    print(inp.size())
    m(inp)
