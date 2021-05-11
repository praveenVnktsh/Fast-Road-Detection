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


# criterion = nn.CrossEntropyLoss()
# optimizer = RMSprop(model.parameters(), lr=configs.base_lr, momentum=configs.momentum, weight_decay=configs.weight_decay)


summary_writer = SummaryWriter(logdir = 'metrics/' + str(configs.trial) + '/')
# summary_writer.add_graph(model, torch.rand((2, 3, 512, 512)).cuda())
summary_writer.add_hparams(configs.dumpConfigs(), {})

# if configs.startEpoch != 0:
    # state = model.load(configs.startEpoch)
    # optimizer.load_state_dict(state['optim'])


SMOOTH = 1e-6
def sim(y_hat,y):

    return torch.sum(y == y_hat)/torch.sum(torch.ones(y.shape))

    # intersection = (y_hat & y).float().sum((1, 2))
    # union = (y_hat | y).float().sum((1, 2))  
    # sim = (intersection + SMOOTH) / (union + SMOOTH)
    # thresholded = torch.clamp(20 * (sim - 0.5), 0, 10).ceil() / 10

    # return thresholded
    


def runEpoch(loader,epoch):
    runningLoss = 0.0
    model.eval()

    similarities = []

    for index, batch in tqdm(enumerate(loader), total = len(loader)):

        image = batch['input'].to(configs.device)
        target = batch['target'].to(configs.device)
        out = model(image)

        similarities.append(sim(torch.argmax(out,axis=1).detach().cpu(),target.detach().cpu()))

    print(f"Epoch:{epoch} sim:{np.mean(similarities)}")
    return np.mean(similarities)

for epoch in range(0,151,10):
    model.load(epoch)
    # trainLoss = runEpoch(trainLoader,epoch)
    similarity = runEpoch(valLoader, epoch)
    
    
    # summary_writer.add_scalar('trainloss', trainLoss , epoch)
    summary_writer.add_scalar('similarity', similarity , epoch)
    # if epoch % 10 == 0:
    #     model.save(optimizer, epoch, trainLoss, valLoss)