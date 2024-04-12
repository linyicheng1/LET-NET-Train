import math
import cv2
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from torchvision.models import resnet
from typing import Optional, Callable
import torch.nn.functional as F
import time
from copy import deepcopy
import logging
from torchvision.transforms import ToTensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class BaseNet(LightningModule):
    def __init__(self, ):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class LETNet(BaseNet):
    def __init__(self, c1: int = 8, c2: int = 16, grayscale: bool = False):
        super().__init__()
        self.gate = nn.ReLU(inplace=True)
        # ================================== feature encoder
        if grayscale:
            self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
        else:
            self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)
        self.conv1 = resnet.conv1x1(c1, c2)
        # ================================== detector and descriptor head
        if grayscale:
            self.conv_head = resnet.conv1x1(c2, 2)
        else:
            self.conv_head = resnet.conv1x1(c2, 4)

    def forward(self, x: torch.Tensor):
        # ================================== feature encoder
        block = self.block1(x)
        x1 = self.gate(self.conv1(block))
        # ================================== detector and descriptor head
        head = self.conv_head(x1)
        score_map = torch.sigmoid(head[:, -1, :, :]).unsqueeze(1)
        descriptor = torch.sigmoid(head[:, :-1, :, :])
        return score_map, descriptor


class LETNetTrain(BaseNet):
    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 grayscale: bool = False):
        super().__init__()
        self.let_net = LETNet(c1, dim // 4, grayscale)
        self.gray = grayscale
        self.gate = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block2 = ResBlock(inplanes=c1, planes=c2, stride=1,
                               downsample=nn.Conv2d(c1, c2, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=c2, planes=c3, stride=1,
                               downsample=nn.Conv2d(c2, c3, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=c3, planes=c4, stride=1,
                               downsample=nn.Conv2d(c3, c4, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)

        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)

        # ================================== feature encoder
        if grayscale:
            self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
        else:
            self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)

        # ================================== detector and descriptor head

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.convhead2 = resnet.conv1x1(dim, dim)

    def forward(self, x: torch.Tensor):
        x1 = self.let_net.block1(x)
        x2 = self.pool2(x1)  # B x c2 x H/2 x W/2
        x2 = self.block2(x2)
        x3 = self.pool4(x2)  # B x c3 x H/8 x W/8
        x3 = self.block3(x3)
        x4 = self.pool4(x3)  # B x c4 x H/32 x W/32
        x4 = self.block4(x4)

        x1 = self.gate(self.let_net.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))

        x2_up = self.upsample2(x2)
        x3_up = self.upsample8(x3)
        x4_up = self.upsample32(x4)
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        desc_map = self.convhead2(x1234)
        head = self.let_net.conv_head(x1)
        score_map = torch.sigmoid(head[:, -1, :, :]).unsqueeze(1)
        local_desc = torch.sigmoid(head[:, :-1, :, :])
        return score_map, local_desc, desc_map

    def extract_dense_map(self, image: torch.Tensor, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        score_map, local_desc, desc_map = self.forward(image)

        # ====================================================
        if h_ != h or w_ != w:
            desc_map = desc_map[:, :, :h, :w]
            score_map = score_map[:, :, :h, :w]  # Bx1xHxW
            local_desc = local_desc[:, :, :h, :w]
        # ====================================================

        # BxCxHxW
        desc_map = torch.nn.functional.normalize(desc_map, p=2, dim=1)

        if ret_dict:
            return {'descriptor_map': desc_map, 'scores_map': score_map, 'local_desc': local_desc}
        else:
            return desc_map, score_map, local_desc


if __name__ == '__main__':
    net = LETNetTrain()
    x = torch.rand(1, 3, 640, 640)
    scores_map, local_descriptor, descriptor_map = net(x)
    print(scores_map.shape, local_descriptor.shape, descriptor_map.shape)
    print(net)

