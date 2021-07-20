
from torch import functional
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
import torchvision
import numpy as np
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from torchvision.transforms.functional import hflip

from configs import Configs
import glob
import cv2
configs = Configs()




class TrainDataset(Dataset):

    def __init__(self,):
        self.sequenceLength = 6
        self.device = configs.device
        self.size = configs.image_size

        a = [sorted(glob.glob(path + "/Camera 5/*")) for path in glob.glob(configs.trainset+"*")]
        a = [[paths[i:i+self.sequenceLength] for i in range(len(paths[:-self.sequenceLength]))] for paths in a]
        self.data = [item for sublist in a for item in sublist]
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

        images = [cv2.cvtColor(cv2.imread(self.data[index][i]),cv2.COLOR_BGR2RGB) for i in range(self.sequenceLength)]
        segs = [cv2.inRange(cv2.imread(configs.labelpath + "/".join(self.data[index][i].split("/")[-3:]).split(".")[0]+"_bin.png")
                ,(179,129,69),(181,131,71)) for i in range(self.sequenceLength)]


        listsegs = []
        listimages = []
        for i in range(0, self.sequenceLength):
            image = self.transform(images[i]).float()
            seg = self.segmentation(segs[i]).float()

            listimages.append(image)
            listsegs.append(seg)

        listimages, listsegs = transformsForImages(listimages, listsegs)
        seg = torch.stack(tuple(listsegs), dim=0)
        images_final = torch.stack(tuple(listimages), dim=0)
        real = self.nonorm(listimages[-1]).float()

        return {'input': images_final, 'target': seg, 'real': real}

    def __len__(self):
        return self.length

        
def transformsForImages(images, segmentations):
    if np.random.rand() > 0.5:
        angle = np.random.randint(-30, 31)
        scale = np.random.randint(90, 111)/ 100.0
        shear = np.random.randint(-15, 16)
        newimages = []
        newsegs = []
        for image,segmentation in zip(images,segmentations):
            newseg = TF.affine(segmentation, angle = angle, translate = (0, 0),  scale = scale, shear = shear)
            newimage = TF.affine(image, angle = angle, translate = (0, 0),  scale = scale, shear = shear)
            newimages.append(newimage)
            newsegs.append(newseg)

    else:
        newimages = images
        newsegs=  segmentations

    return newimages, newsegs

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

        return {'input': image, 'target': seg, 'real': real}

    def __len__(self):
        return self.length


class lit_custom_data(pl.LightningDataModule):

    def setup(self, stage=None):

        self.configs = Configs()

        if stage is not None:
            self.configs.trainset = stage + self.configs.trainset
            self.configs.valset = stage + self.configs.valset
            self.configs.testset = stage + self.configs.testset

        self.cpu = 12
        self.pin = True
        self.dataset = TrainDataset()
        length = len(self.dataset)

        self.trainIndices = range(0, int(configs.valSplit *length))
        self.valIndices = range(int(configs.valSplit *length), length)
        print('Loading dataset')

    def train_dataloader(self):
        
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler =  SubsetRandomSampler(self.trainIndices), pin_memory=self.pin)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler =  SubsetRandomSampler(self.valIndices), pin_memory=self.pin)



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # path = "datasetsmall.pt"
    cd = TrainDataset()
    print(cd[0]['input'].size())
    print(cd[0]['target'].size())
    print(cd[0]['real'].size())
