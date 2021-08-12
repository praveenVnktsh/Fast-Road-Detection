
import os
import json
import torch


class Configs():

    def __init__(self):

        self.device = torch.device('cuda')

        self.trainset = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/ColorImage_road02/Resized_Data/"
        self.labelpath = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/Labels_road02/Resized_Data/"
        self.valset = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/ColorImage_road02/Resized_Data/"
        self.testset = "/home/rvp/Face-Lane-Detection/drive_download/apolloscape/ColorImage_road02/Resized_Data/"
        # self.image_size = 270
        self.image_size = (288,352)
        
        
        # model params
        self.batchSize = 32
        self.valSplit = 0.9

    def dumpConfigs(self):
        dic = self.__dict__
        dic['device'] = 'cuda'
        return dic


if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()
