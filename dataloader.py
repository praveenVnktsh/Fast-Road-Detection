
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

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        if index > self.length:
            raise Exception(
                f"Dataloader out of index. Max Index:{self.length - self.n_images}, Index asked:{index}.")

        image = self.transform(self.data[index]['front']['rgb']).float()
        seg = self.transform(self.data[index]['front']['seg']).float()
        # images.shape = [3,128,128]

        return {'input': image, 'target': seg}

    def __len__(self):
        return self.length


class lit_custom_data(pl.LightningDataModule):

    def setup(self, stage):
    
        self.configs = Configs()
        self.dataset = CustomDataset(self.configs)
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.configs.valSplit * dataset_size))
        self.trainIndices, self.valIndices = indices[split:], indices[:split]



    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                         num_workers=0, sampler=SubsetRandomSampler(self.trainIndices))

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                         num_workers=0, sampler=SubsetRandomSampler(self.valIndices))




if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = r"dataset.pt"
    # cd = CustomDataset(config)
    # print(cd[0])
    data_module = lit_custom_data()
    print("python")
    # print(cd[0]['front']['seg'].shape)
