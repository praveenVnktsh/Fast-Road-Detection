
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop

import numpy as np

import pytorch_lightning as pl

from configs import Configs

configs = Configs()


class CustomDataset(Dataset):

    def __init__(self, configs: Configs):

        self.device = configs.device
        self.path = configs.datasetPath
        self.size = (configs.image_size, configs.image_size)

        self.data = torch.load(self.path)
        self.length = len(self.data)

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

        self.segmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        if index > self.length:
            raise Exception(
                f"Dataloader out of index. Max Index:{self.length - self.n_images}, Index asked:{index}.")

        image = self.transform(self.data[index]['front'])
        seg = self.segmentation(self.data[index]['road']).bool().float()
        # images.shape = [3,128,128]

        return {'input': image, 'target': seg}

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
        self.cpu = 0
        self.pin = True

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler=SubsetRandomSampler(self.trainIndices), pin_memory=self.pin)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler=SubsetRandomSampler(self.valIndices), pin_memory=self.pin)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1,
                          num_workers=self.cpu, pin_memory=self.pin)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = r"dataset.pt"
    cd = CustomDataset(configs)
    print(cd[0])
    data_module = lit_custom_data()
    print("python")
    # print(cd[0]['front']['seg'].shape)
