import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import attach_attention_module

        
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_attention=None):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_layer = attach_attention_module(channel=self.expansion*out_channels, attention_module=use_attention)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.residual_function(x)
        if self.use_attention:
            out = self.attention_layer(out)
        return nn.ReLU(inplace=True)(out + self.shortcut(x))

class _Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1, use_attention=None):
        super(_Bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_layer = attach_attention_module(channel=self.expansion*planes, attention_module=use_attention)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.use_attention:
            out = self.attention_layer(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=2, use_attention=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,num_blocks[0],64,stride=1,use_attention=use_attention)
        self.layer2 = self._make_layer(block,num_blocks[1],128,stride=2,use_attention=use_attention)
        self.layer3 = self._make_layer(block,num_blocks[2],256,stride=2,use_attention=use_attention)
        self.layer4 = self._make_layer(block,num_blocks[3],512,stride=2,use_attention=use_attention)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,num_blocks,planes,stride,use_attention):
        layers = []
        layers+=[block(self.inplanes,planes,stride=stride,use_attention=use_attention)]
        self.inplanes = planes*block.expansion
        for i in range(1,num_blocks):
            layers+=[block(self.inplanes,planes,use_attention=use_attention)]
            self.inplanes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes = 2, use_attention=None):
    return ResNet(_BasicBlock, [2,2,2,2], num_classes=num_classes, use_attention=use_attention)

def ResNet34(num_classes = 2, use_attention=None):
    return ResNet(_BasicBlock, [3,4,6,3], num_classes=num_classes, use_attention=use_attention)

def ResNet50(num_classes = 2, use_attention=None):
    return ResNet(_Bottleneck, [3,4,6,3], num_classes=num_classes, use_attention=use_attention)

def ResNet101(num_classes = 2, use_attention=None):
    return ResNet(_Bottleneck, [3,4,23,3], num_classes=num_classes, use_attention=use_attention)

def ResNet152(num_classes = 2, use_attention=None):
    return ResNet(_Bottleneck, [3,8,36,3], num_classes=num_classes, use_attention=use_attention)
