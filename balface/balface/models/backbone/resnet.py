import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BN(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BN(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = BN(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BN(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, use_norm=False):
        super(ResNet, self).__init__()

        global BN
        BN = nn.BatchNorm2d

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer_one = self.conv1
        self.bn1 = BN(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.output_dim = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.layer_one_out = self.conv1(x)
        self.layer_one_out.requires_grad_()
        self.layer_one_out.retain_grad()
        out = F.relu(self.bn1(self.layer_one_out))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        encoding = out.view(out.size(0), -1)

        return encoding


class TorchResNet34(nn.Module):
    def __init__(self):
        super(TorchResNet34, self).__init__()
        resnet34 = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet34.children())[:-1]
        self.resnet34 = nn.Sequential(*modules)

    def forward(self, x):
        return self.resnet34(x).squeeze(2).squeeze(2)

class HeadTorchResnet34(nn.Module):
    def __init__(self):
        super(HeadTorchResnet34, self).__init__()
        resnet34 = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet34.children())[:-1]
        self.resnet34 = nn.Sequential(*modules)
        self.head = nn.Linear(512, 512)

    def forward(self, x):
        return self.head(self.resnet34(x).squeeze(2).squeeze(2))

class RSCTorchResnet34(nn.Module):
    def __init__(self):
        super(RSCTorchResnet34, self).__init__()
        resnet34 = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet34.children())[:-2]
        self.resnet34 = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.resnet34(x)
        return x

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # return x

def torch_resnet34():
    # 21,284,672
    return TorchResNet34()

def head_torch_resnet34():
    return HeadTorchResnet34()

def rsc_torch_resnet34():
    return RSCTorchResnet34()


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    net = resnet50()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
