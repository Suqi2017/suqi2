# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:24:03 2019

@author: asus
"""

from train import *
from ResNet import *
from dataset import *

if __name__ == '__main__':
    net = ResNet(ResidualBlock)
    train(net)