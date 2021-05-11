
from configs import Configs
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class CustomDataset(Dataset):

    def __init__(self, configs: Configs):

        self.device = configs.device
        self.path = configs.datasetPath
        self.n_images = configs.n_images

        self.data = torch.load(self.path)
        self.length = len(self.data) - self.n_images

        print(self.length, 'images in', self.path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        if index > self.length :
            raise Exception(
                f"Dataloader out of index. Max Index:{self.length - self.n_images}, Index asked:{index}.")
        images = []
        for i in range(self.n_images):
            images.append(self.transform(
                self.data[(self.n_images-1) + index-i]['front']['rgb']).float())

        images = torch.stack(images)
        # images.shape = [n_images,3,128,128]
        seg = torch.tensor(
            self.data[(self.n_images-1) + index]['front']['seg']).long()

        return {'input': images, 'target': seg}

    def __len__(self):
        return self.length


if __name__ == "__main__":
    c = Configs()

    cd = CustomDataset(c)
    print(cd[0])
    # print(cd[0]['front']['seg'].shape)
