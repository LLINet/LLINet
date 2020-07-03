import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Linear_transform(nn.Module):
    def __init__(self,args):
        super(Linear_transform,self).__init__()
        self.fc = nn.Linear(1024,1024)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        # x = nn.functional.normalize(x, p=2, dim=1)     
        return x

def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),
        nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class Resnet(nn.Module):
    def __init__(self,args):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = make_layer(64, 128, 2)
        self.layer2 = make_layer(128, 512, 2, stride=2)
        #self.layer3 = make_layer(512, 1024, 2, stride=2)
        #self.layer4 = make_layer(512, 1024, 2, stride=2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512,1024)

    def forward(self, x,l):
        x = x.unsqueeze(1)
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        #out = self.att(x)
        x = self.fc1(x)
        out = nn.functional.normalize(x, p=2, dim=1)
        return out
