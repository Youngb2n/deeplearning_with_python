import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import attach_attention_module

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,num_classes=1000, use_attention=None, wavename=None):
        super(VGG,self).__init__()
        self.vgg_name = vgg_name
        self.use_attention = use_attention
        self.wavename = wavename
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self,x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return F.sigmoid(out)
    
    def _make_layers(self, cfg):
        layers=  []
        in_channels = 3
        for x in cfg:
            if x == 'M' and self.wavename:
                print(Downsample(wavename=self.wavename))
                layers += [Downsample(wavename=self.wavename)]#, nn.BatchNorm2d(in_channels)]
        #    elif x == 'M' and not self.wavename:
        #        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else :
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=x ,kernel_size=(3,3),stride =1, padding=1,bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)  # inplace 메모리 감소
                           ]
                if self.use_attention:
                    layers += [attach_attention_module(channel=x, attention_module=self.use_attention)]

                in_channels = x
        return nn.Sequential(*layers)


def VGG11(num_classes= 2, use_attention=None, wavename=None):
    return VGG('VGG11',num_classes=num_classes, use_attention=use_attention, wavename=wavename)
def VGG13(num_classes =2, use_attention=None, wavename=None):
    return VGG('VGG13',num_classes=num_classes, use_attention=use_attention, wavename=wavename)
def VGG16(num_classes = 2, use_attention=None, wavename=None):
    return VGG('VGG16',num_classes=num_classes, use_attention=use_attention, wavename=wavename)
def VGG19(num_classes =2, use_attention=None, wavename=None):
    return VGG('VGG19',num_classes=num_classes, use_attention=use_attention, wavename=wavename)
