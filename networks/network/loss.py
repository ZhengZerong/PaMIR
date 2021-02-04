"""
Losses
borrowed from: https://github.com/knazeri/edge-connect/blob/master/src/loss.py
"""

from __future__ import print_function, absolute_import, division
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2 as cv


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = self.real_label if is_real else self.fake_label
            labels = labels.expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']),
                                     self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']),
                                     self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']),
                                     self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']),
                                     self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class StyleLoss2(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss2, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, style_2, style_3, style_4, style_5):
        # Compute features
        x_vgg = self.vgg(x)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), style_2)
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), style_3)
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), style_4)
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), style_5)
        return style_loss


class FirstOrderLoss(nn.Module):
    def __init__(self, channel):
        super(FirstOrderLoss, self).__init__()
        dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        winx = torch.from_numpy(dx).unsqueeze(0).unsqueeze(0) / 8
        winy = torch.from_numpy(dy).unsqueeze(0).unsqueeze(0) / 8
        winx = winx.expand((channel, -1, -1, -1))
        winy = winy.expand((channel, -1, -1, -1))
        self.register_buffer('winx', winx)
        self.register_buffer('winy', winy)
        self.criterion = torch.nn.MSELoss()

    def __call__(self, input, target):
        ch_num = input.size()[1]  # channel number
        didx = F.conv2d(input, self.winx, padding=0, groups=ch_num)
        didy = F.conv2d(input, self.winy, padding=0, groups=ch_num)
        dtdx = F.conv2d(target, self.winx, padding=0, groups=ch_num)
        dtdy = F.conv2d(target, self.winy, padding=0, groups=ch_num)
        return self.criterion(didx, dtdx) + self.criterion(didy, dtdy)


class SSIM(nn.Module):
    def __init__(self, window_size=11, channel=3, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in
             range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(
            _2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def __call__(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        self.window = window
        self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=(1.0 / 1.6, 1.0 / 2.3, 1.0 / 1.8, 1.0 / 2.8, 1.0 / 0.8)):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.w = weights
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape([1, -1, 1, 1])
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape([1, -1, 1, 1])
        mean = torch.from_numpy(mean)
        std = torch.from_numpy(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def __call__(self, x, y):
        # Compute features
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.w[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.w[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.w[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.w[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.w[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class BackwardConsistencyLoss(nn.Module):
    def __init__(self, batch_size, height, width):
        super(BackwardConsistencyLoss, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.criterion = torch.nn.L1Loss()
        h_grid = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        h_grid = h_grid.expand(batch_size, -1, height, -1)
        v_grid = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
        v_grid = v_grid.expand(batch_size, -1, -1, width)
        self.register_buffer('backward_grid',
                             torch.cat([h_grid, v_grid], 1))

    def __call__(self, l, r, flow_l2r, mask=None):
        flow_l2r_ = torch.cat([flow_l2r[:, 0:1, :, :] / (self.width - 1.0) * 2,
                               flow_l2r[:, 1:2, :, :] / (self.height - 1.0) * 2], 1)
        grid = (self.backward_grid - flow_l2r_).permute(0, 2, 3, 1)
        l_ = F.grid_sample(input=r, grid=grid, mode='bilinear', padding_mode='border')

        """testing code"""
        # l_host = l.detach().cpu().numpy().transpose((0, 2, 3, 1))
        # l__host = l_.detach().cpu().numpy().transpose((0, 2, 3, 1))
        # msk_host = mask.detach().cpu().numpy().transpose((0, 2, 3, 1))
        # for bi in range(self.batch_size):
        #     cv.imwrite('./debug/test_f_%d_1.png' % bi,
        #                cv.cvtColor(np.uint8(l_host[bi] * msk_host[bi] * 255), cv.COLOR_RGB2BGR))
        #     cv.imwrite('./debug/test_f_%d_2.png' % bi,
        #                cv.cvtColor(np.uint8(l__host[bi] * msk_host[bi] * 255), cv.COLOR_RGB2BGR))
        # import pdb
        # pdb.set_trace()

        if mask is not None:
            return self.criterion(mask * l, mask * l_)
        else:
            return self.criterion(l, l_)


class EdgeBackwardConsistencyLoss(nn.Module):
    def __init__(self, batch_size, height, width, channel=3):
        super(EdgeBackwardConsistencyLoss, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.criterion = torch.nn.L1Loss()
        h_grid = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        h_grid = h_grid.expand(batch_size, -1, height, -1)
        v_grid = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
        v_grid = v_grid.expand(batch_size, -1, -1, width)
        self.register_buffer('backward_grid',
                             torch.cat([h_grid, v_grid], 1))
        dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        winx = torch.from_numpy(dx).unsqueeze(0).unsqueeze(0) / 8
        winy = torch.from_numpy(dy).unsqueeze(0).unsqueeze(0) / 8
        winx = winx.expand((channel, -1, -1, -1))
        winy = winy.expand((channel, -1, -1, -1))
        self.register_buffer('winx', winx)
        self.register_buffer('winy', winy)

    def __call__(self, l, r, flow_l2r, mask=None):
        ch_num = l.size()[1]  # channel number
        dldx = F.conv2d(l, self.winx, padding=1, groups=ch_num)
        dldy = F.conv2d(l, self.winy, padding=1, groups=ch_num)
        drdx = F.conv2d(r, self.winx, padding=1, groups=ch_num)
        drdy = F.conv2d(r, self.winy, padding=1, groups=ch_num)

        flow_l2r_ = torch.cat([flow_l2r[:, 0:1, :, :] / (self.width - 1.0) * 2,
                               flow_l2r[:, 1:2, :, :] / (self.height - 1.0) * 2], 1)
        grid = (self.backward_grid + flow_l2r_).permute(0, 2, 3, 1)
        dldx_ = F.grid_sample(input=drdx, grid=grid, mode='bilinear', padding_mode='border')
        dldy_ = F.grid_sample(input=drdy, grid=grid, mode='bilinear', padding_mode='border')
        if mask is not None:
            return self.criterion(mask * dldx_, mask * dldx) \
                   + self.criterion(mask * dldy_, mask * dldy)
        else:
            return self.criterion(dldx_, dldx) + self.criterion(dldy_, dldy)


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
