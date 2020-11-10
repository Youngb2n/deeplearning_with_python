import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.attention import attach_attention_module

class _Bottleneck(nn.Module):
    def __init__(self, inplanes,growth_rate,use_attention=None):
        super(_Bottleneck,self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes,4*growth_rate,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3,padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_layer = attach_attention_module(channel=4*growth_rate, attention_module=use_attention)

    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.use_attention:
            out = self.attention_layer(out)
        out = torch.cat((x, out), dim=1)
        return out

class _Transition(nn.Module):
    def __init__(self, inplanes,outplanes):
        super(_Transition,self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv = nn.Conv2d(inplanes,outplanes,kernel_size=1,bias=False)
        self.relu =nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.conv(self.relu(self.bn(x)))
        out = F.avg_pool2d(out,2)
        return out


class DenseNet(nn.Module):
    def __init__(self,num_blocks, growth_rate =32, reduction=0.5, num_classes=2, use_attention=None):
        super(DenseNet,self).__init__()
        num_planes =64
        self.growth_rate = growth_rate
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(3,num_planes,7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layers(_Bottleneck, num_blocks[0], num_planes)
        num_planes += num_blocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.tran1 = _Transition(num_planes,out_planes)
        num_planes = out_planes 

        self.layer2 = self._make_layers(_Bottleneck, num_blocks[1], num_planes)
        num_planes += num_blocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.tran2 = _Transition(num_planes,out_planes)
        num_planes = out_planes
        
        self.layer3 = self._make_layers(_Bottleneck, num_blocks[2], num_planes)
        num_planes += num_blocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.tran3 = _Transition(num_planes,out_planes)
        num_planes = out_planes

        self.layer4 = self._make_layers(_Bottleneck, num_blocks[3], num_planes)
        num_planes += num_blocks[3]*growth_rate

        self.bn2 = nn.BatchNorm2d(num_planes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(num_planes, num_classes)

    def forward(self,x):
        out = nn.ReLU(inplace= True)(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.tran1(self.layer1(out))
        out = self.tran2(self.layer2(out))
        out = self.tran3(self.layer3(out))
        out = nn.ReLU(inplace= True)(self.bn2(self.layer4(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)

        return out

    def _make_layers(self,block,num_blocks,inplanes):
        layers = []
        for i in range(num_blocks):
            layers.append(block(inplanes,self.growth_rate, use_attention=self.use_attention))
            inplanes += self.growth_rate
        return nn.Sequential(*layers)


def DenseNet121(num_classes= 2, use_attention=None):
    return DenseNet([6,12,24,16], growth_rate=32,num_classes=num_classes, use_attention=use_attention)

def DenseNet169(num_classes =2, use_attention=None):
    return DenseNet([6,12,32,32], growth_rate=32,num_classes=num_classes, use_attention=use_attention)

def DenseNet201(num_classes =2, use_attention=None):
    return DenseNet([6,12,48,32], growth_rate=32,num_classes=num_classes, use_attention=use_attention)

def DenseNet161(num_classes =2, use_attention=None):
    return DenseNet([6,12,36,24], growth_rate=48,num_classes=num_classes, use_attention=use_attention)
