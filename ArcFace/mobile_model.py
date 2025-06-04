import math

import torch
from torch import nn
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import functional as F

class Conv_block(Module):
    def __init__(self,in_c,out_c,kernel=[3,3],stride=[2,2],padding=(1,1),groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups,bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.prelu(x)
        return x

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class mobileFaceNet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(mobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x

class Depth_separable_conv(Module):
    def __init__(self,in_c,out_c,kernel=[3,3],stride=[2,2],padding=(1,1),expansion=1):
        super(Depth_separable_conv, self).__init__()
        self.depth_wise_conv = nn.Conv2d(in_c,in_c,kernel_size=kernel,stride=stride,padding=padding,groups=in_c,bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(in_c)
        self.prelu = nn.PReLU(in_c)
        self.point_wise_conv = nn.Conv2d(in_c,out_c,kernel_size=[1,1],stride=[1,1],padding=0,bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.batchnorm_1(x)
        x = self.prelu(x)
        x = self.point_wise_conv(x)
        x = self.batchnorm_2(x)
        return x

class BottleNeck(Module):
    def __init__(self,in_c,out_c,kernel=[3,3],stride=[1,1],padding=(1,1),groups=1):
        super(BottleNeck,self).__init__()
        self.stride = stride
        self.conv1 = Conv_block(in_c,groups,kernel=[1,1],stride=[1,1],padding=0)
        self.d_s_conv = Depth_separable_conv(groups,out_c,kernel=kernel,stride=stride,padding=padding)
    def forward(self,x):
        out = self.conv1(x)
        out = self.d_s_conv(out)
        if self.stride[0] == 1:
           return x+out
        else:
            return out

