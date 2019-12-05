# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:23:17 2019

@author: asus
"""

import torch as t 
import torch.nn as nn 
import torch.nn.functional as F
import os



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=True)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=62):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        self.fc5 = nn.Linear(512, num_classes)
        # print(num_classes)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1)
        # print(x.shape)  
        x = x.view(-1, 512)
        
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        y5 = self.fc5(x)
        return y1, y2, y3, y4, y5

    def save(self, circle):
        name = "./checkpoint/captcha/resNet" + str(circle) + ".pth"
        t.save(self.state_dict(), name)
        name2 = "./checkpoint/captcha/resNet_new.pth"
        t.save(self.state_dict(), name2)

    def loadIfExist(self):
        fileList = os.listdir("./checkpoint/captcha/")
        # print(fileList)
        if "resNet_new.pth" in fileList:
            name = "./checkpoint/captcha/resNet_new.pth"
            self.load_state_dict(t.load(name))
            print("the latest model has been load")

