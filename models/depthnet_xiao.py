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
import sys
sys.path.insert(0,"/data/home/scv7305/run/chenjiawei/adv_branch/CDCN")
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        
        self.decoder_adv = nn.Sequential(
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
        self.dense = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            )

        self.average = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spoofer = nn.Linear(map_size * map_size, 1)
 
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

        map_adv = self.decoder_adv(embedding)
        map_adv = map_adv.squeeze(1)
   
        embedding_1 = self.average(embedding)
        map_x_1 = self.decoder2(embedding_1)
        map_x_1 = map_x_1.squeeze(1)

        embedding_2 = self.average(embedding_1)
        map_x_2 = self.decoder3(embedding_2)
        map_x_2 = map_x_2.squeeze(1)

        
        output = map_x.view(map_x.size(0), -1)
        adv_out = map_adv.view(map_adv.size(0), -1)
        map_adv = self.dense(adv_out)
        
        spoof_output = self.spoofer(output)
        spoof_output = torch.sigmoid(spoof_output)
        adv_out = torch.sigmoid(map_adv)     
        
        return map_x, spoof_output, map_x_1, map_x_2, adv_out
    
if __name__ == '__main__':
    model = Depthnet().cuda()
    a=torch.load('../checkpoint/train_adv_Depthnet()_couple_patch_1_last.pt')
    # new_state_dict = {}
    # for k,v in a.items():
    #     new_state_dict[k[7:]] = v

    model.load_state_dict(a)

    with torch.no_grad():
        model.eval()
        test_data = MyDataset('/data/home/scv7305/run/chenjiawei/fasdata/facepath.txt', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean                                =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))
        dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        
        n = 0
        for images in dataloader_test:
                # get the inputs
            images = images.cuda()

            map_x, spoof_output, map_x_1, map_x_2, adv_out = model(images)
            map_score = (torch.clamp(torch.mean(map_x, dim=(1,2)),0,1)+torch.clamp(torch.mean(map_x_1, dim=(1,2)),0,1)
                            +torch.clamp(torch.mean(map_x_2, dim=(1,2)),0,1)+spoof_output)/4
            
            socre_all = map_score.squeeze(-1).float()*adv_out.squeeze(-1).float()
            if socre_all < 0.5:
                n = n+1
        
        print(n)
            #print(spoof_out)

