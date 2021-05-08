
from torch import functional
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
import torchvision
import numpy as np

import pytorch_lightning as pl
from torchvision.transforms.functional import hflip

from configs import Configs

configs = Configs()


class CustomDataset(Dataset):

    def __init__(self, configs: Configs):

        self.device = configs.device
        self.path = configs.datasetPath
        self.size = (configs.image_size, configs.image_size)
        print('Loading dataset')
        self.data = torch.load(self.path)
        self.length = len(self.data) * 2

        print(self.length, 'images in', self.path)

        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.unormalized = transforms.Compose([
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
        flag = 0
        if index > self.length:
            raise Exception(
                f"Dataloader out of index. Max Index:{self.length - self.n_images}, Index asked:{index}.")
        if index >= self.length / 2:
            index = int(index - self.length // 2)
            flag = 1

        image = self.transform(self.data[index]['front'])
        seg = self.segmentation(self.data[index]['road']).bool().float()
        real = self.unormalized(self.data[index]['front'])
        # image.shape = [3,160,160]

        if flag:
            image = torchvision.transforms.functional.hflip(image)
            seg = torchvision.transforms.functional.hflip(seg)
            real = torchvision.transforms.functional.hflip(real)

        return {'input': image, 'target': seg, "real": real}

    def __len__(self):
        return self.length


class lit_custom_data(pl.LightningDataModule):

    def setup(self, stage=None):

        self.configs = Configs()
        self.dataset = CustomDataset(self.configs)
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.configs.valSplit * dataset_size))
        self.trainIndices, self.valIndices = indices[split:], indices[:split]
        self.cpu = 4
        self.pin = True

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler=SubsetRandomSampler(self.trainIndices), pin_memory=self.pin)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler=SubsetRandomSampler(self.valIndices), pin_memory=self.pin)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1,
                          num_workers=0, sampler=SequentialSampler(self.valIndices), pin_memory=self.pin)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = r"dataset.pt"
    cd = CustomDataset(configs)
    print(cd[0])
    data_module = lit_custom_data()
    print("python")
    # print(cd[0]['front']['seg'].shape)
