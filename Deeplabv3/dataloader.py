
from configs import Configs
from torch.utils.data.dataset import Dataset 
import torch
from torchvision.transforms import transforms
import cv2
import glob

def convertSegToGray(im):
    roadmask = getMask(im, 150)
    im[roadmask == 255] = 2
    carMask = getMask(im, 116)
    im[carMask == 255] = 1
    
    im[~((carMask == 255) + (roadmask == 255))] = 0

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

def getMask(im, color, dist = 20):
    frame_HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (color - dist, 0, 0), (color + dist, 255, 255))
    return frame_threshold



class CustomDataset(Dataset):

    def __init__(self, configs : Configs):

        self.device = configs.device
        self.path = configs.datasetPath
        
        self.rgbpaths = sorted(glob.glob(self.path + 'front/rgb/*'))
        self.segpaths = sorted(glob.glob(self.path + 'front/seg/*'))

        self.length = len(self.rgbpaths)

        print(self.length, 'images in', self.path)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, i):

        im = cv2.resize(cv2.imread(self.rgbpaths[i]), (512, 512), interpolation = cv2.INTER_NEAREST)
        image = self.transform(im).float()
        seg1 = cv2.resize(convertSegToGray(cv2.imread(self.segpaths[i])), (512, 512), interpolation = cv2.INTER_NEAREST)
        seg = torch.tensor(seg1).long()

        return {'input' : image, 'target' : seg}

    def __len__(self): 
        return self.length


if __name__ == "__main__":
    c = Configs()
    
    cd = CustomDataset(c)
    print(cd[0])
    # print(cd[0]['front']['seg'].shape)