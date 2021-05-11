
import os
import json
import torch
class Configs():
    
    def __init__(self):
        
        self.device = torch.device('cuda')

        self.s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
        self.t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
        self.n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time

        # output channel of each conv stage
        self.c = [32, 16, 24, 32, 64, 96, 160]

        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)
        self.down_sample_rate = 32  # classic down sample rate
        self.seqLength = 50

        # dataset parameters
        self.num_class = 3  
        self.datasetPath = 'dataset/' 

        # model params
        self.startEpoch = 0
        self.trial = 2
        self.saveDir = 'results/trial' + str(self.trial) + '/'
        
        # train parameters
        self.nEpochs = 1000
        self.batchSize = 5
        self.base_lr = 0.0002
        self.power = 0.9
        self.momentum = 0.9
        self.valSplit = 0.1
        self.weight_decay = 0.0005
        
        


        os.makedirs(self.saveDir + '/models/', exist_ok= True)
        # os.makedirs(self.saveDir + '/images/', exist_ok= True)
        self.dumpConfigs()
        
    def dumpConfigs(self):
        with open(self.saveDir + 'configs.json', 'w') as f:
            dic = self.__dict__
            dic['device'] = 'cuda'
            json.dump(dic, f,  sort_keys=True, indent=4)

if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()

