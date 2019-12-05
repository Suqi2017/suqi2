# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:32:08 2019

@author: asus
"""

import torch as t
from torch import optim
from torch import nn
from dataset import *
from torch.utils.data import DataLoader
# from torchnet import meter

def train(model):
    avgLoss =0.0
    
    if t.cuda.is_available():
        model =model.cuda()
    
    trainDataset = Captcha(r'C:/Users/25447/Desktop/第三次竞赛/train',train=True)
    trainLoader = DataLoader(trainDataset,batch_size=32,shuffle=True)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    # loss_meter = meter.AverageValueMeter()
    
    for epoch in range(200):
        print(epoch)
        for circle ,inputs in enumerate(trainLoader):
            x,label = inputs
            if t.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            # print(x.shape)    
            label = label.long()
            label1, label2, label3, label4, label5 = label[:, 0], label[:, 1], label[:, 2], label[:, 3], label[:,4]
            
            optimizer.zero_grad()
            
            y1, y2, y3, y4, y5 = model(x)
            # print(y1.shape)
            loss1, loss2, loss3, loss4, loss5 = criterion(y1, label1), criterion(y2, label2), criterion(y3, label3), criterion(y4, label4), criterion(y5, label5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            
            # loss_meter.add(loss.item())
            # print(loss)
            avgLoss += loss.item()
            loss.backward()
            optimizer.step()
            
            if circle % 3 == 1:
                print("after %d circle,the train loss is %.5f" %(circle, avgLoss /3))
                avgLoss = 0
        if epoch % 10 == 9:
            print('Save checkpoint...')
            t.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},'./checkpoint')



            