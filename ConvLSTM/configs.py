
import os
import json
import torch


class Configs():

    def __init__(self):

        self.device = torch.device('cuda')

        # dataset parameters
        self.datasetPath = 'dataset.pt'
        self.n_images = 4  # Number of images needed together

        # model params
        self.startEpoch = 0
        self.trial = 0
        self.saveDir = 'results/trial' + str(self.trial) + '/'

        # train parameters
        self.nEpochs = 151
        self.batchSize = 50
        self.base_lr = 0.0002
        self.momentum = 0.9
        self.valSplit = 0.1
        self.weight_decay = 0.0005

        os.makedirs(self.saveDir + '/models/', exist_ok=True)
        # os.makedirs(self.saveDir + '/images/', exist_ok= True)

    def dumpConfigs(self):
        dic = self.__dict__
        dic['device'] = 'cuda'
        return dic


if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()
