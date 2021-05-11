from torch import nn
from torch.optim.rmsprop import RMSprop
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import CustomDataset
from configs import Configs
from models.network import MobileNetv2_DeepLabv3
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

torch.cuda.empty_cache()
configs = Configs()

model = MobileNetv2_DeepLabv3(configs = configs)
dataset = CustomDataset(configs)

indices = list(range(len(dataset)))
# indices = list(range(100))

np.random.seed(0)
np.random.shuffle(indices)

split = int(np.floor(configs.valSplit * len(indices)))
trainIndices, valIndices = indices[split:], indices[:split]

trainLoader = DataLoader(dataset, batch_size = configs.batchSize , num_workers = 0, sampler = SubsetRandomSampler(trainIndices))
valLoader = DataLoader(dataset, batch_size = configs.batchSize , num_workers = 0, sampler = SubsetRandomSampler(valIndices))

print('# Training images = ', len(trainLoader) * configs.batchSize)
print('# Validation images = ', len(valLoader)* configs.batchSize)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = RMSprop(model.parameters(), lr=configs.base_lr, momentum=configs.momentum, weight_decay=configs.weight_decay)



summary_writer = SummaryWriter(logdir = 'tensorboardx/trial' + str(configs.trial) + '/')
summary_writer.add_graph(model, torch.rand((2, 3, 512, 512)))

if configs.startEpoch != 0:
    state = model.load(configs.startEpoch)
    optimizer.load_state_dict(state['optim'])


def runEpoch(loader, epoch, train = False):
    runningLoss = 0.0
    if train:
        model.train()
    else:
        model.eval()

    for index, batch in tqdm(enumerate(loader), total= len(trainLoader), desc = 'TRAINING = ' + str(train) + ' Epoch ' + str(epoch)):

        optimizer.zero_grad()
        
        image = batch['input'].to(configs.device)
        target = batch['target'].to(configs.device)

        out = model(image)
        # loss = criterion((out * 255).long(), (target*255).long())
        loss = criterion(out, target)

        
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

        if not train and index == 0:
            
            outmap = (torch.argmax(out.detach().cpu(), axis = 1).float() / 2.0)
            outmap = torch.stack((outmap, outmap, outmap), dim = 1)

            targetmap = (torch.stack((target, target, target), dim = 1).float() / 2.0)
            n_rows = 5
            data = torch.cat((image.detach().cpu()[:n_rows].float(), outmap[:n_rows], targetmap.detach().cpu()[:n_rows].float()), dim = 2)
            summary_writer.add_images('validateImagesIndex0', data , epoch)
    return runningLoss / (len(loader) * configs.batchSize)


def adjust_lr(epoch):

    learning_rate = configs.base_lr * (1 - float(epoch) / configs.nEpochs) ** configs.power
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('Learning at', learning_rate, 'units')


for epoch in range(configs.startEpoch, configs.nEpochs):
    trainLoss = runEpoch(trainLoader, epoch, train = True)
    valLoss = runEpoch(valLoader, epoch, train = False)
    
    summary_writer.add_scalar('trainloss', trainLoss , epoch)
    summary_writer.add_scalar('valloss', valLoss , epoch)

    adjust_lr(epoch)

    model.save(optimizer, epoch, trainLoss, valLoss)

    print('Train loss epoch', epoch, '=', trainLoss)
    print('Val loss epoch', epoch, '=', valLoss)
    print('------------------------')