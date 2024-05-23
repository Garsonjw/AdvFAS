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

from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize
from models.RGBDmh import RGBDMH
from models.CMFL import CMFL

from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pdb
from dataset import MyDataset

from utils import AvgrageMeter, accuracy, performances
from tqdm import tqdm
from attacks import PGD_rgbdmh

import copy

# Dataset roo
train_list = '/data/home/scv7305/run/chenjiawei/fasdata/WMCA/WMCA_train.txt'
val_list = '/data/home/scv7305/run/chenjiawei/fasdata/WMCA/WMCA_dev.txt'
test_list =  '/data/home/scv7305/run/chenjiawei/fasdata/WMCA/WMCA_eval.txt'

train_depth_list = '/data/home/scv7305/run/chenjiawei/fasdata/WMCA/WMCA_depth_train.txt'
val_depth_list = '/data/home/scv7305/run/chenjiawei/fasdata/WMCA/WMCA_depth_dev.txt'
test_depth_list =  '/data/home/scv7305/run/chenjiawei/fasdata/WMCA/WMCA_depth_eval.txt'


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log_P1.txt', 'w')
    
    echo_batches = args.echo_batches

    print("WMCA, P1:\n ")

    log_file.write('WMCA, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?

    print('train from scratch!\n')
    log_file.write('train from scratch!\n')
    log_file.flush()    
        
    model = RGBDMH()
    model = model.cuda()

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)

    print(model) 

    ACER_save = 1.0
    best_val_acc = -1

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_feature = AvgrageMeter()
        loss_lables = AvgrageMeter() 
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        g=torch.Generator()
        g.manual_seed(0)
        
        train_RGB_data = MyDataset(train_list, transform=transforms.Compose([ transforms.ToTensor(),transforms.Resize([224,224]), transforms.RandomErasing(),            transforms.RandomHorizontalFlip(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])]))
        dataloader_train_RGB = DataLoader(train_RGB_data, batch_size=args.batchsize, shuffle=True, num_workers=4, generator=g)

        train_depth_data = MyDataset(train_depth_list, transform=transforms.Compose([ transforms.ToTensor(),transforms.Resize([224,224]),transforms.RandomErasing(), transforms.RandomHorizontalFlip(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])]))
        dataloader_train_depth = DataLoader(train_depth_data, batch_size=args.batchsize, shuffle=True, num_workers=4, generator=g)
    
        criterion_cmfl = CMFL(alpha=1, gamma= 3, binary= False, multiplier=2)
        i = 0
        for RGB_data, depth_data in zip(dataloader_train_RGB, dataloader_train_depth):
            i = i+1
            images = torch.cat((RGB_data[0], depth_data[0]), 1).cuda()
            lables = RGB_data[1].cuda()
            # index = np.arange(len(lables))
            # np.random.shuffle(index)
            # images = images[index]
            # lables = lables[index]
            optimizer.zero_grad()

            gap, op, op_rgb, op_d =  model(images)
            # print(op)
            # print(lables)
            # print(op)
            # print(lables)
            # print(op[0].item())
            
            for X in range(len(lables)):
                if lables[X] == 1:
                    lables[X] = 0   #fake
                elif lables[X] == 0:
                    lables[X] = 1   #true
            
            loss_cmfl = criterion_cmfl(op_rgb.squeeze(-1).float(), op_d.squeeze(-1).float(), lables.float())
            loss_bce = nn.BCELoss()(op.squeeze(-1).float(), lables.float())

            loss = 0.5*loss_bce+0.5*loss_cmfl
            loss.backward()
            optimizer.step()
            #scheduler.step()   

            loss_lables.update(loss.data, images.size(0))# images.size(0)=100

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches

                #print(torch.argmax(spoof_out, dim=1))
                print('epoch:%d, mini-batch:%3d, lr=%f, lable_loss= %.4f\n' % (epoch + 1, i + 1, lr,  loss_lables.avg))
        
        # whole epoch average
        print('epoch:%d, Train: lables_loss= %.4f\n' % (epoch + 1, loss_lables.avg))
        log_file.write('epoch:%d, Train: lables_loss= %.4f \n' % (epoch + 1, loss_lables.avg))
        log_file.flush()
                 
        #### validation/test
        if epoch <100:
             epoch_test = 1   
        else:
            epoch_test = 10   
        #epoch_test = 1

        if epoch % epoch_test == epoch_test-1:    # test every 5 epochs  
            model.eval()
            ###########################################
            '''                val             '''
            ###########################################
            # val for threshold
            val_RGB_data = MyDataset(val_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224]), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))
            dataloader_val_RGB = DataLoader(val_RGB_data, batch_size=1, shuffle=False, num_workers=4)

            val_depth_data = MyDataset(val_depth_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224]), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))
            dataloader_val_depth = DataLoader(val_depth_data, batch_size=1, shuffle=False, num_workers=4)

            map_score_list = []
            for RGB_data, depth_data in zip(dataloader_val_RGB, dataloader_val_depth):
                # get the inputs
                images = torch.cat((RGB_data[0], depth_data[0]), 1).cuda()
                lables = RGB_data[1].cuda()
                
                #将原始标签换成常规的标签
                for X in range(len(lables)):
                    if lables[X] == 1:
                        lables[X] =0#fake
                    elif lables[X] ==0:
                        lables[X] =1#real

                optimizer.zero_grad()
                gap, op, op_rgb, op_d = model(images)
                score = op [0]
                map_score_list.append('{} {}\n'.format(score.item(), lables[0].item()))
                
            score_spoof_val_filename = args.log+'/'+args.log+'score_spoof_val.txt'
            with open(score_spoof_val_filename, 'w') as file:
                file.writelines(map_score_list) 

            ###########################################
            '''                test             '''
            ##########################################
            # test for ACC
            test_RGB_data = MyDataset(test_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224]), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))
            dataloader_test_RGB = DataLoader(test_RGB_data, batch_size=1, shuffle=False, num_workers=4)

            test_depth_data = MyDataset(test_depth_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224]), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))
            dataloader_test_depth = DataLoader(test_depth_data, batch_size=1, shuffle=False, num_workers=4)
            
            map_score_list = []
            adv_score_list = []
            for RGB_data, depth_data in zip(dataloader_test_RGB, dataloader_test_depth):
                # get the inputs
                images = torch.cat((RGB_data[0], depth_data[0]), 1).cuda()
                lables = RGB_data[1].cuda()

                #将原始标签替换成常用标签
                for X in range(len(lables)):
                    if lables[X] == 1:
                        lables[X] =0
                    elif lables[X] ==0:
                        lables[X] =1
                
                #对抗攻击
                if args.attack == 'PGD':
                    attack = PGD_rgbdmh.PGD(model, np.inf)
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)

                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:

                            adv_imgs = attack.forward(adv_images, torch.tensor([0]).cuda())
                            images = torch.cat((images, adv_imgs), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
   
                model.eval()
                optimizer.zero_grad()
                
                gap, op, op_rgb, op_d  =  model(images)

                score_spoof = op[0]
                map_score_list.append('{} {}\n'.format(score_spoof.item(), lables[0].item()))

                if lables[0] ==0:
                    score_adv = op[1]
                    adv_score_list.append('{} {}\n'.format(score_adv.item(), lables[1].item()))

            score_spoof_test_filename = args.log+'/'+args.log+'score_spoof_test.txt'
            with open(score_spoof_test_filename, 'w') as file:
                file.writelines(map_score_list)

            adv_test_filename = args.log+'/'+args.log+'adv_test.txt'
            with open(adv_test_filename, 'w') as file:
                file.writelines(adv_score_list) 

                # log written
            
            #############################################################     
            #       performance measurement both val and test
            #############################################################     
            
            val_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER = performances(score_spoof_val_filename, score_spoof_test_filename)
            
            adv_val_threshold, adv_val_ACC, adv_val_ACER, adv_test_ACC, adv_test_APCER, adv_test_BPCER, adv_test_ACER = performances(score_spoof_val_filename, adv_test_filename)
            
            if val_ACC>best_val_acc:
                best_val_acc=val_ACC
                torch.save(model.state_dict(),'./checkpoint/train_no_adv_'+args.model+'_testadv_best_'+args.attack+'.pt')
                print('Best test accuracy achieved on epoch: %d'%(epoch + 1))
            torch.save(model.state_dict(),'./checkpoint/train_no_adv_'+args.model+'_testadv_last_'+args.attack+'.pt')
            
            print('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (epoch + 1, val_threshold, val_ACC, val_ACER))
            log_file.write('\n epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % (epoch + 1, val_threshold, val_ACC, val_ACER))
            
            print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
            #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
            #log_file.write('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f \n\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            
            print('epoch:%d, Test:  adv_ACC= %.4f, adv_APCER= %.4f, adv_BPCER= %.4f, adv_ACER= %.4f' % (epoch + 1, adv_test_ACC, adv_test_APCER, adv_test_BPCER, adv_test_ACER))
            #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, adv_test_ACC, adv_test_APCER, adv_test_BPCER, adv_test_ACER))
            #log_file.write('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f \n\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.flush()

                         
        #if epoch <1:    
        # save the model until the next improvement     
        #    torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pth' % (epoch + 1))

    print('Finished Training')
    log_file.close()
  

  
 

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=80, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=50, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    parser.add_argument('--attack', type=str, default='PGD', help='attack')#PGD patch
    parser.add_argument('--model', type=str, default='RGBDmh', help='model')#Depthnet Resnet18 Resnet50
    parser.add_argument('--log', type=str, default="RGBDmh_no_test_adv_PGD", help='log and save model name')

    args = parser.parse_args()
    train_test()
    
