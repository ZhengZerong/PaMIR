'''A CycleGAN Encoder'''

import torch.nn as nn
import math

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initializes network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Residual(BaseNetwork):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.GroupNorm(32, self.numIn)  ###?????????
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=3, stride=1,
                               padding=1)
        self.bn1 = nn.GroupNorm(32, self.numOut)
        self.conv2 = nn.Conv2d(self.numOut, self.numOut, bias=True, kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.GroupNorm(32, self.numOut)
        self.conv3 = nn.Conv2d(self.numOut, self.numOut, bias=True, kernel_size=3, stride=1,
                               padding=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)
        self.init_weights()

    def forward(self, x):
        residual = x
        # out = self.bn(x)
        # out = self.relu(out)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.conv3(out)
        # out = self.relu(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


class CycleGANEncoder(BaseNetwork):
    """CycleGan Encoder"""

    def __init__(self, num_in=3, num_out=256):
        super(CycleGANEncoder, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.num_in, self.num_out // 4, bias=True, kernel_size=7,
                               stride=1, padding=3, padding_mode='circular')
        self.bn1 = nn.GroupNorm(32, self.num_out // 4)
        self.conv2 = nn.Conv2d(self.num_out // 4, self.num_out // 2, bias=True, kernel_size=3,
                               stride=2, padding=1)
        self.bn2 = nn.GroupNorm(32, self.num_out // 2)
        self.conv3 = nn.Conv2d(self.num_out // 2, self.num_out, bias=True, kernel_size=3,
                               stride=2, padding=1)
        self.bn3 = nn.GroupNorm(32, self.num_out)

        self.r1 = Residual(self.num_out, self.num_out)
        self.r2 = Residual(self.num_out, self.num_out)
        self.r3 = Residual(self.num_out, self.num_out)
        self.r4 = Residual(self.num_out, self.num_out)
        self.r5 = Residual(self.num_out, self.num_out)
        self.r6 = Residual(self.num_out, self.num_out)

        self.init_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.r1(out)
        out = self.r2(out)
        out = self.r3(out)
        out = self.r4(out)
        out = self.r5(out)
        out = self.r6(out)

        return out


