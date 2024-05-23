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
from dataset import MyDataset
from models.depthnet_ensemble import Depthnet
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from attacks import PGD_adaptive
import numpy as np
import copy
from tqdm import tqdm

if __name__ == '__main__':
    model = Depthnet().cuda()
    #depthnet PGD thre=0.6903
    a=torch.load('./checkpoint_PGD/train_adv_Depthnet()_couple_PGDepoch6.pt')

    model.load_state_dict(a)


    model.eval()
    test_data = MyDataset('/data/home/scv7305/run/yanghao/fasdata/WMCA/WMCA_eval.txt', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean                                =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))
    dataloader_test = DataLoader(test_data, batch_size=70, shuffle=False, num_workers=1)
    num_adv=0
    num_adv_suc = 0
    for images, lables in tqdm(dataloader_test):
            # get the inputs
        images, lables = images.cuda(), lables.cuda()

        for X in range(len(lables)):
            if lables[X] == 1:
                lables[X] = 0   #fake
            elif lables[X] == 0:
                lables[X] = 1   #true

        attack = PGD_adaptive.PGD(model, np.inf)
        adv_images = copy.deepcopy(images)
        adv_lables = copy.deepcopy(lables)
        #print(lables)
        for Y in range(len(adv_lables)):
            if adv_lables[Y] ==0:
                
                num_adv = num_adv+1
                #adv_imgs = attack.forward(adv_images, adv_lables)
                adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                images = torch.cat((images, adv_imgs), dim=0)
                adv_lables[Y] = 2
                lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)

        map_x, spoof_out, map_x_1, map_x_2, adv_out =  model(images)
            
        map_1_score = (torch.clamp(torch.mean(map_x, dim=(1,2)),0,1)+torch.clamp(torch.mean(map_x_1, dim=(1,2)),0,1)
        +torch.clamp(torch.mean(map_x_2, dim=(1,2)),0,1))
        spoof_out = spoof_out.squeeze(1)
        map_score = (map_1_score+spoof_out)/4
        socre_all = map_score.squeeze(-1).float()*adv_out.squeeze(-1).float()
        for S in range(len(socre_all)):
            if socre_all[S] >=0.6903:
                num_adv_suc = num_adv_suc + 1 
    
    print(num_adv)
    print(num_adv_suc)
    print(num_adv_suc/num_adv)

