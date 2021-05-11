from torch.nn.modules.conv import Conv2d
from configs import Configs
import torch
import torch.nn as nn
import os
from torch.optim import RMSprop
import numpy as np
import matplotlib.pyplot as plt

import layers as layers

WARNING = lambda x: print('\033[1;31;2mWARNING: ' + x + '\033[0m')
LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')

# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
        used for Semantic Segmentation on Cityscapes dataset
    """

    def __init__(self, configs : Configs):
        super(MobileNetv2_DeepLabv3, self).__init__()

        self.configs = configs


        # build network
        blocks = []

        ''' The network is made according the MobileNetV2 paper (https://arxiv.org/pdf/1801.04381v4.pdf), 
        The layers are follwed upto the 6th bottleneck operator Pg.5 Table 2'''
        # Initial 2D Convolution
        # All the parameters are set from config.py.
        # input channels - 3 (rgb image)
        # output channels - 32
        # kernel size - 3
        # strides - 2
        # Relu6 is used for robustness in low-prescision computing.

        blocks.append(nn.Sequential(nn.Conv2d(3, configs.c[0], 3, stride=configs.s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(configs.c[0]),
                                   # nn.Dropout2d(params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # Bottlneck Layers 
        for i in range(6):
            blocks.extend(layers.get_inverted_residual_block_arr(configs.c[i], configs.c[i+1],
                                                                t=configs.t[i+1], s=configs.s[i+1],
                                                                n=configs.n[i+1]))

        
        
        # torch.Size([2, 160, 32, 32])


        '''DeepLabv3 head https://arxiv.org/pdf/1706.05587.pdf'''
        # Dilated/Atrous convolutional layer 1-4
        # first dilation=rate, follows dilation=multi_grid*rate


        rate = configs.down_sample_rate // configs.output_stride # 32/16  = 2
        blocks.append(layers.InvertedResidual(configs.c[6], configs.c[6],
                                             t=configs.t[6], s=1, dilation=rate))

        # torch.Size([2, 160, 32, 32])

        for i in range(3):
            blocks.append(layers.InvertedResidual(configs.c[6], configs.c[6],
                                                 t=configs.t[6], s=1, dilation=rate*configs.multi_grid[i]))

        '''Atrous Spatial Pyramid Pooling ... refer Pg.7 https://arxiv.org/pdf/1706.05587.pdf'''
        # torch.Size([2, 160, 32, 32])


        self.encoder = nn.Sequential(*blocks).cuda()

        self.encoderConv = nn.Sequential(
            Conv2d(160, 40, kernel_size = 3, padding = 1, stride = 1),
            Conv2d(40, 20, kernel_size = 3, padding = 1, stride = 1),
            Conv2d(20, 10, kernel_size = 3, padding = 1, stride = 1)
        ).cuda()
        # torch.Size([2, 10, 32, 32])
        
        

        blocks.append(nn.LSTM(10, 10, 2, batch_first = True))


        self.decoderConv = nn.Sequential(
            Conv2d(10, 20, kernel_size = 3, padding = 1, stride = 1),
            Conv2d(20, 40, kernel_size = 3, padding = 1, stride = 1),
            Conv2d(40, 160, kernel_size = 3, padding = 1, stride = 1)
        ).cuda()

        
        # # # ASPP layer, architecture in layers.py
        # blocks.append(layers.ASPP_plus(configs))

        # # torch.Size([2, 256, 32, 32])

        # # # final conv layer
        # blocks.append(nn.Conv2d(256, configs.num_class, 1))

        # # torch.Size([2, 3, 32, 32])
        

        # # # bilinear upsample
        # blocks.append(nn.Upsample(scale_factor=configs.output_stride, mode='bilinear', align_corners=False))

        # self.network = nn.Sequential(*blocks).cuda()

        # initialize
        self.clearHidden()
        self.initialize()


    def clearHidden(self):
        self.hidden = (torch.zeros((2, self.configs.seqLength, 10)), torch.zeros((2,  self.configs.seqLength, 10)))

    def forward(self, x):
        out = self.encoder(x)
        conved = self.encoderConv(out)
        flattenedconv = torch.flatten(conved, 2)
        print(flattenedconv.shape)
        return out

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def save(self, optimizer : RMSprop, epoch, train_loss, val_loss):
        dic = {
            'epoch'        :  epoch,
            'train_loss'   :  train_loss,
            'val_loss'     :  val_loss,
            'model'   :  self.state_dict(),
            'optim'    :  optimizer.state_dict()
        }
        path = self.configs.saveDir + 'models/model_epoch_%d.pth' % epoch
        torch.save(dic, path)
        LOG('Saved Checkpoint at %s' % path)

    def load(self, startEpoch):
        """
        Load checkpoint from given path
        """

        loadPath = self.configs.saveDir + 'models/model_epoch_%d.pth' % startEpoch
        state = torch.load(loadPath)
        LOG('Loading Checkpoint at %s' % loadPath)
        self.network.load_state_dict(state['model'])

        return state

   



if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512)).cuda()
    m = MobileNetv2_DeepLabv3(Configs())
    m(x)