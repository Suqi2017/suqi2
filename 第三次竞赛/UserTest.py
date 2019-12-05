# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:19:58 2019

@author: asus
"""

import torch
from torchvision import transforms as T
from dataset import *
from ResNet import *
from PIL import Image
lable1 = []
m  = 0
for n in range(0,200):
    label = []
    
    for i in range(0,5):
        m = m+20
        for j in range(m-20,m):
            img_path = r'C:/Users/25447/Desktop/第三次竞赛/test/'+str(j)+'.jpg'
            label.append(img_path)
    lable1.append(label)


label1 = []
model_path ='./checkpoint'
for i in range(200): 
    for j in range(100):
        img_path = lable1[i][j]
        
        img = Image.open(img_path)
        img_tensor = T.Compose([
                    T.Resize((130, 64)), 
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
        inputs = img_tensor(img)
        inputs = torch.unsqueeze(inputs,dim=0)
        cnn = ResNet(ResidualBlock)
        cnn.eval()

        cnn.load_state_dict(torch.load(model_path,map_location='cpu')['model_state_dict'])

        y1, y2, y3, y4 ,y5 = cnn(inputs)
        y1, y2, y3, y4 ,y5 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1), y5.topk(1, dim=1)[1].view(1, 1)
        y = t.cat((y1, y2, y3, y4, y5), dim=1)

        label1.append(LabeltoStr(y[0]))
import pandas as pd


length = len(label1)
test = pd.DataFrame(data=label1,index = range(length),columns=['y'])
test.index.name = 'id'
test.to_csv('pd5.csv')    
income = pd.read_csv(r"pd5.csv")
print(income)      
    
    
    
    
    