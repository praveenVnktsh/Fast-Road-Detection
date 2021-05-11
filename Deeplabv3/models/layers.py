import torch.nn as nn
import torch

'''
Various layers are defined here to use direclty later on
'''


class InvertedResidual(nn.Module):
    '''
    implementing basic Inverted residual as suggested here: https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5
    '''

    def __init__(self, input, output, t=6, s=1, dilation=1):
        '''
        Initailizing Inverted Residual Layer
        :param input: number of input channels
        :param output: number of output channels
        :param t: expansion factor of block
        :param stride: stride of first convolution
        :param dilation: dilation rate of 3*3 depthwise conv
        '''
        super(InvertedResidual, self).__init__()

        self.in_ = input
        self.out_ = output
        self.t = t
        self.s = s
        self.dilation = dilation
        self.inverted_residual_block()

    def inverted_residual_block(self):
        '''
        Building Inverted Residual Block
        '''

        block = []

        # conv1*1
        block.append(nn.Conv2d(self.in_, self.in_*self.t, 1, bias=False))
        block.append(nn.BatchNorm2d(self.in_*self.t))
        block.append(nn.ReLU6())

        # conv3*3 depthwise
        block.append(nn.Conv2d(self.in_*self.t, self.in_*self.t, kernel_size=3,
                               stride=self.s,
                               padding=self.dilation,
                               groups=self.t,
                               dilation=self.dilation))
        block.append(nn.BatchNorm2d(self.in_*self.t))
        block.append(nn.ReLU6())

        # conv 1*1 linear
        block.append(nn.Conv2d(self.in_*self.t, self.out_, 1, bias=False))
        block.append(nn.BatchNorm2d(self.out_))

        self.block = nn.Sequential(*block)

        # conv residual connections
        if self.in_ != self.out_ and self.s != 2:
            self.res_conv = nn.Sequential(
                nn.Conv2d(self.in_, self.out_, 1, bias=False),
                nn.BatchNorm2d(self.out_))

        else:
            self.res_conv = None

    def forward(self, x):
        if self.s == 1:
            # use residual connections
            if self.res_conv is None:
                out = x + self.block(x)
            else:
                out = self.res_conv(x) + self.block(x)
        else:
            # plain block, no residual
            out = self.block(x)

        return out


def get_inverted_residual_block_arr(in_, out_, t=6, s=1, n=1):
    '''
    Function to get n-blocks of inverted residual layers

    :params in_: number of input channel
    :params out_: number of output channel
    :params t: expansion size
    :params s: stride
    :params n: number of serial blocks
    '''
    block = []
    block.append(InvertedResidual(in_, out_, t=t, s=s))
    for _ in range(n-1):
        block.append(InvertedResidual(out_, out_, t=t, s=1))

    return nn.Sequential(*block)


class ASPP_plus(nn.Module):
    '''
    More about this here https://www.programmersought.com/article/65703173978/
    '''

    def __init__(self, params):
        super(ASPP_plus, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 1, bias=False),
                                    nn.BatchNorm2d(256))
        self.conv33_1 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[0],
                                                dilation=params.aspp[0],
                                                bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_2 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[1],
                                                dilation=params.aspp[1],
                                                bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_3 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[2],
                                                dilation=params.aspp[2],
                                                bias=False),
                                      nn.BatchNorm2d(256))

        self.concate_conv = nn.Sequential(
            nn.Conv2d(256*5, 256, 1, bias=False), nn.BatchNorm2d(256))

    def forward(self, x):

        # making data
        conv11 = self.conv11(x)
        conv33_1 = self.conv33_1(x)
        conv33_2 = self.conv33_2(x)
        conv33_3 = self.conv33_3(x)

        # image pool and upsampling
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        image_pool = self.conv11(image_pool)
        upsample = nn.Upsample(
            size=x.size()[2:], mode="bilinear", align_corners=True)
        upsample = upsample(image_pool)

        # Concating all the images
        concate = torch.cat(
            [conv11, conv33_1, conv33_2, conv33_3, upsample], dim=1)

        return self.concate_conv(concate)


if __name__ == '__main__':

    # rand_input = torch.randn((600, 500, 512))
    ir = InvertedResidual(512, 256)
    print(ir)
