import torch.nn as nn
import torch


class FCN32s(nn.Module):
    '''
    A reminant of the fcn32s, where we have removed the feature 
    extractor part that is directedly replaced by the pretrained 
    resnet feature extractors.

    This is only the Decoder Part
    '''

    def __init__(self, n_class):
        super(FCN32s, self).__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            128, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Sequential(
            nn.Conv2d(32, n_class, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x5):
        # output = self.pretrained_net(x)
        # x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(self.relu(self.deconv2(score)))
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)
