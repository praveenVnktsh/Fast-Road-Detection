from libs.models.fastfcn import FastFCN
from dataloader import CustomDataset
from configs import Configs
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
import torch.nn as nn
from tqdm import tqdm

import torch
import numpy as np
import sys


configs = Configs()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = r"D:\Code\CV_project\lstm_fast_fcn\dataset.pt"


dataset = CustomDataset(configs)
dataset_size = len(dataset)
indices = list(range(dataset_size))

np.random.seed(0)
np.random.shuffle(indices)

split = int(np.floor(configs.valSplit * dataset_size))
trainIndices, valIndices = indices[split:], indices[:split]
trainLoader = DataLoader(dataset, batch_size=configs.batchSize,
                         num_workers=0, sampler=SubsetRandomSampler(trainIndices))
valLoader = DataLoader(dataset, batch_size=configs.batchSize,
                       num_workers=0, sampler=SubsetRandomSampler(valIndices))

model = FastFCN(n_classes=2)
model.to(device)

summary_writer = SummaryWriter(log_dir='run/' + str(configs.trial) + '/')
summary_writer.add_graph(model, torch.rand((1, 3, 224, 224)).cuda())

criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=configs.base_lr,
                    momentum=configs.momentum, weight_decay=configs.weight_decay)


def runEpoch(loader, epoch, train=False):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0
    for i, batch in tqdm(enumerate(loader), total=len(loader)):

        img = batch["input"].to(configs.device)
        target = batch["target"].to(configs.device)
        out = model(img)
        # out = torch.argmax(out, axis=1)

        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if not train and i == 0:

            outmap = torch.argmax(out.detach().cpu(), axis=1).float()
            outmap = torch.stack((outmap, outmap, outmap), dim=1)

            targetmap = torch.stack((target, target, target), dim=1).float()
            n_rows = 5
            data = torch.cat((img.detach().cpu()[:n_rows], outmap.float()[
                             :n_rows], targetmap.detach().cpu()[:n_rows]), dim=2)
            summary_writer.add_images('validateImagesIndex0', data, epoch)

    return running_loss


if __name__ == '__main__':

    for epoch in range(configs.startEpoch, configs.nEpochs):

        trainLoss = runEpoch(trainLoader, epoch, train=True)
        valLoss = runEpoch(valLoader, epoch, train=False)

        # model.save(optimizer, epoch, trainLoss, valLoss)

        summary_writer.add_scalar('trainloss', trainLoss, epoch)
        summary_writer.add_scalar('valloss', valLoss, epoch)

        if epoch % 10 == 0:
            model.save(optimizer, epoch, trainLoss, valLoss)

    summary_writer.flush()
    summary_writer.close()
