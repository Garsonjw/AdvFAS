'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

class Conv2d_basic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_basic, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        return out_normal

class Depthnet(nn.Module):

    def __init__(self, basic_conv=Conv2d_basic, map_size=16, multi_heads=False, input_channel=3):   
        super(Depthnet, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),    
            
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  
            
            basic_conv(128, int(128*1.6), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128*1.6)),
            nn.ReLU(inplace=True),  
            basic_conv(int(128*1.6), 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#éçť´
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(inplace=True),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  
            basic_conv(128, int(128*1.4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128*1.4)),
            nn.ReLU(inplace=True),  
            basic_conv(int(128*1.4), 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(inplace=True),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Original

        self.downconv1 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            basic_conv(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.downconv2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   
        )

        self.decoder1 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),    
        )
        
        self.decoder2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),    
        )

        self.decoder3 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),    
        )
        

        self.average = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spoofer = nn.Linear(map_size * map_size, 2)
 
    def forward(self, x):	    	# x [3, 112, 112]

        x = self.conv1(x)		   

        x_Block1 = self.Block1(x)#13*128*64*64

        x_Block1_32x32 = self.downconv1(x_Block1)#13*128*16*16

        x_Block2 = self.Block2(x_Block1)#13*128*32*32
     
        x_Block2_32x32 = self.downconv2(x_Block2)#13*128*16*16  
     
        x_Block3 = self.Block3(x_Block2)#13*128*16*16	    
   
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3), dim=1) #13*384*16*16   
    
        #pdb.set_trace()
        
        embedding = self.lastconv1(x_concat)#13*128*16*16

        map_x = self.decoder1(embedding)#13*1*16*16
 
        map_x = map_x.squeeze(1)#13*16*16
   
        embedding_1 = self.average(embedding)
        map_x_1 = self.decoder2(embedding_1)
        map_x_1 = map_x_1.squeeze(1)

        embedding_2 = self.average(embedding_1)
        map_x_2 = self.decoder3(embedding_2)
        map_x_2 = map_x_2.squeeze(1)

        
        output = map_x.view(map_x.size(0), -1)
        
        spoof_output = self.spoofer(output)
        #spoof_output = torch.sigmoid(spoof_output)
        
        ####adv      
        
        return map_x, map_x_1, map_x_2, spoof_output