
import os
import json
import torch


class Configs():

    def __init__(self):

        self.device = torch.device('cuda')

        self.s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
        self.t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
        self.n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time

        # output channel of each conv stage
        self.c = [32, 16, 24, 32, 64, 96, 160]

        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)
        self.down_sample_rate = 32  # classic down sample rate

        # dataset parameters
        self.rescale_size = 600
        self.image_size = 160
        self.num_class = 4  # Change later 20  # 20 classes for training
        self.datasetPath = r"I:\dataset\cv\valset.pt"
        self.n_images = 4  # Number of images needed together
        self.dataloader_workers = 12
        self.shuffle = True
        self.train_batch = 10
        self.val_batch = 2
        self.test_batch = 1

        # model params
        self.startEpoch = 0
        self.trial = 0
        self.saveDir = 'results/trial' + str(self.trial) + '/'

        # train parameters
        self.nEpochs = 151
        self.batchSize = 16
        self.base_lr = 0.0002
        self.power = 0.9
        self.momentum = 0.9
        self.valSplit = 0.1
        self.weight_decay = 0.0005

        # os.makedirs(self.saveDir + '/models/', exist_ok=True)
        # os.makedirs(self.saveDir + '/images/', exist_ok= True)

    def dumpConfigs(self):
        dic = self.__dict__
        dic['device'] = 'cuda'
        return dic


if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()
