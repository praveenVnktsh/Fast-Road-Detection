
import os
import json
import torch


class Configs():

    def __init__(self):

        self.device = torch.device('cuda')

        self.trainset = "/content/drive/Shareddrives/3DCV/Datasets/trainset.pt"
        self.valset = "/content/drive/Shareddrives/3DCV/Datasets/valset.pt"
        self.testset = "/content/drive/Shareddrives/3DCV/Datasets/valset.pt"
        self.image_size = 160
        # model params
        self.batchSize = 16
        self.valSplit = 0.9

    def dumpConfigs(self):
        dic = self.__dict__
        dic['device'] = 'cuda'
        return dic


if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()
