
from torch import functional
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
import torchvision
import numpy as np

import pytorch_lightning as pl
from torchvision.transforms.functional import hflip

from configs import Configs

configs = Configs()


class TrainDataset(Dataset):

    def __init__(self, dataset):
        self.sequenceLength = 6
        self.device = configs.device
        self.size = (configs.image_size, configs.image_size)

        self.data = dataset

        self.length = (len(self.data) - self.sequenceLength) * 2

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.nonorm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])
        self.segmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        
        images = []
        segs = []
        flip = False
        if index >= self.length / 2:
            flip = True
            index = index % (self.length // 2)

        for i in range(index, index + 6):
            image = self.transform(self.data[i]['front'])
            seg = self.segmentation(self.data[i]['road']).bool().float()

            if flip:
                image = torchvision.transforms.functional.hflip(image)
                seg = torchvision.transforms.functional.hflip(seg)
            images.append(image)
            segs.append(seg)

        seg = torch.stack(tuple(segs), dim = 0)
        image = torch.stack(tuple(images), dim = 0)
        real = self.nonorm(self.data[i]['front'])

        return {'input': image, 'target': seg, 'real' : real}

    def __len__(self):
        return self.length



class TestDataset(Dataset):

    def __init__(self, dataset):
        
        self.sequenceLength = 6
        self.device = configs.device
        self.size = (configs.image_size, configs.image_size)
        
        self.data = dataset

        self.length = len(self.data)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.nonorm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])
        self.segmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        
        image = self.transform(self.data[index]['front'])
        seg = self.segmentation(self.data[index]['road']).bool().float()

        real = self.nonorm(self.data[index]['front'])
        

        return {'input': image, 'target': seg, 'real' : real}

    def __len__(self):
        return self.length


class lit_custom_data(pl.LightningDataModule):

    def setup(self, stage=None):

        self.configs = Configs()
        
        self.cpu = 0
        self.pin = True
        print('Loading dataset')
        

    def train_dataloader(self):
        dataset = TrainDataset(torch.load(self.configs.trainset))
        return DataLoader(dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler=SubsetRandomSampler(self.trainIndices), pin_memory=self.pin)

    def val_dataloader(self):
        dataset = TrainDataset(torch.load(self.configs.valset))
        return DataLoader(dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler=SubsetRandomSampler(self.valIndices), pin_memory=self.pin)

    def test_dataloader(self):
        dataset = TestDataset(torch.load(self.configs.testset))
        return DataLoader(dataset, batch_size=1,
                          num_workers=self.cpu, pin_memory=self.pin)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = "datasetsmall.pt"
    # cd = CustomDataset(configs)
    # print(cd[0]['input'].size())