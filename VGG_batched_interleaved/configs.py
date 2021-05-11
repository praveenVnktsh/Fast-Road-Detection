
import os
import json
import torch


class Configs():

    def __init__(self):

        self.device = torch.device('cuda')

        self.datasetPath = r"testset.pt"

        self.shuffle = True

        # train parameters
        self.n_images = 4  # Number of images needed together

        self.batchSize = 50
        self.valSplit = 0.1

    def dumpConfigs(self):
        dic = self.__dict__
        dic['device'] = 'cuda'
        return dic


if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()
