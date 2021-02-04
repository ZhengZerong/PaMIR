"""
Implementation of Stacked Hourglass network
code borrowed from: https://github.com/xingyizhou/pytorch-pose-hg-3d
"""

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
        self.bn = nn.GroupNorm(4, self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=3, stride=1,
                               padding=1)
        # self.bn1 = nn.BatchNorm2d(self.numOut / 2)
        # self.conv2 = nn.Conv2d(self.numOut / 2, self.numOut / 2, bias=True, kernel_size=3, stride=1,
        #                        padding=1)
        self.bn2 = nn.GroupNorm(4, self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=3, stride=1,
                               padding=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)
        self.init_weights()

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


class Hourglass(BaseNetwork):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.init_weights()

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2


class HourglassNet(BaseNetwork):
    def __init__(self, nStack, nModules, nFeats, numOutput):
        super(HourglassNet, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.numOutput = numOutput

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.GroupNorm(4, 64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(
                nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                nn.GroupNorm(4, self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(
                nn.Conv2d(self.nFeats, self.numOutput, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(
                    nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)
        self.init_weights()

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.pool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_

        return out