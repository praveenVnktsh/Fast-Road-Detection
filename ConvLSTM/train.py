from model import FCN
from torch import nn
from torch.optim.rmsprop import RMSprop
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import CustomDataset
from configs import Configs
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter


configs = Configs()

model = FCN(configs = configs).cuda()
dataset = CustomDataset(configs)

dataset_size = len(dataset)
indices = list(range(dataset_size))

np.random.seed(0)
np.random.shuffle(indices)

split = int(np.floor(configs.valSplit * dataset_size))

trainIndices, valIndices = indices[split:], indices[:split]

trainLoader = DataLoader(dataset, batch_size = configs.batchSize , num_workers = 0, sampler = SubsetRandomSampler(trainIndices))
valLoader = DataLoader(dataset, batch_size = configs.batchSize , num_workers = 0, sampler = SubsetRandomSampler(valIndices))


criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=configs.base_lr, momentum=configs.momentum, weight_decay=configs.weight_decay)


summary_writer = SummaryWriter(logdir = 'runs/' + str(configs.trial) + '/')
# summary_writer.add_graph(model, torch.rand((2, 3, 512, 512)).cuda())
summary_writer.add_hparams(configs.dumpConfigs(), {})

if configs.startEpoch != 0:
    state = model.load(configs.startEpoch)
    optimizer.load_state_dict(state['optim'])


def runEpoch(loader, epoch, train = False):
    runningLoss = 0.0
    if train:
        model.train()
    else:
        model.eval()

    for index, batch in tqdm(enumerate(loader), total = len(loader)):

        image = batch['input'].to(configs.device)
        target = batch['target'].to(configs.device)

        out = model(image)

        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

        if not train and index == 0:
            
            outmap = torch.argmax(out.detach().cpu(), axis = 1).float()
            outmap = torch.stack((outmap, outmap, outmap), dim = 1)

            targetmap = torch.stack((target, target, target), dim = 1).float()
            n_rows = 10
            data = torch.cat((image.detach().cpu()[:n_rows, 0], outmap.float()[:n_rows], targetmap.detach().cpu()[:n_rows]), dim = 2)
            summary_writer.add_images('validateImagesIndex0', data , epoch)

    return runningLoss / (len(loader) * configs.batchSize)


for epoch in range(configs.startEpoch, configs.nEpochs):
    trainLoss = runEpoch(trainLoader, epoch, train = True)
    valLoss = runEpoch(valLoader, epoch, train = False)
    
    
    summary_writer.add_scalar('trainloss', trainLoss , epoch)
    summary_writer.add_scalar('valloss', valLoss , epoch)
    if epoch % 10 == 0:
        model.save(optimizer, epoch, trainLoss, valLoss)