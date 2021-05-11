
import os
import json
import torch


class Configs():

    def __init__(self):

        self.device = torch.device('cuda')

        # dataset parameters
        self.image_size = 160
        self.datasetPath = "testset.pt"

        # train parameters
        self.batchSize = 16
        self.valSplit = 0.1

    def dumpConfigs(self):
        dic = self.__dict__
        dic['device'] = 'cuda'
        return dic


if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()
