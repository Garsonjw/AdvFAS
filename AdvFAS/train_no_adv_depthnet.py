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

from models.depthnet import Depthnet
from models.CDCNs import Conv2d_cd, CDCN, CDCNpp


from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
from dataset import MyDataset

from utils import AvgrageMeter, accuracy, performances
from tqdm import tqdm

# Dataset root
train_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Train_images/'        
val_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Dev_images/'     
test_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Test_images/'   

train_list = '/data/home/scv7305/run/yanghao/fasdata/WMCA/WMCA_train.txt'
val_list = '/data/home/scv7305/run/yanghao/fasdata/WMCA/WMCA_dev.txt'
test_list =  '/data/home/scv7305/run/yanghao/fasdata/WMCA/WMCA_eval.txt'


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
        
    model = Depthnet()
    #model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    #criterion_absolute_loss = nn.MSELoss().cuda()

    print(model) 
    
    
    #bandpass_filter_numpy = build_bandpass_filter_numpy(30, 30)  # fs, order  # 61, 64 

    ACER_save = 1.0
    best_val_acc = -1

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        #scheduler.step() #注意scheduler的位置
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_feature = AvgrageMeter()
        loss_lables = AvgrageMeter() 
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = MyDataset(train_list, transform=transforms.Compose([transforms.ToTensor(),transforms.RandomErasing(), transforms.RandomHorizontalFlip(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)
        i = 0

        for images, lables in dataloader_train:
            i = i+1
            # get the inputs
            images, lables = images.cuda(), lables.cuda()

            optimizer.zero_grad()

            
            #pdb.set_trace()
            
            # forward + backward + optimize
            score = []
            map_x, map_x_1, map_x_2, spoof_out =  model(images)
            #score = map_x.view(7,-1)
            #score = torch.nn.Sigmoid()(torch.mean(map_x, dim=(1,2)))   #meanćšćsum
            # print(score)
            # print(lables)
            #map_x = torch.nn.Sigmoid()(map_x)

            for X in range(len(lables)):
                if lables[X] == 1:
                    lables[X] =0
                else:
                    lables[X] =1

            map_label = []
            map_1_label = []
            map_2_label = []
            for X in range(len(lables)):
                if lables[X] == 0:
                    map_label.append(np.zeros([16, 16]))
                    map_1_label.append(np.zeros([8, 8]))
                    map_2_label.append(np.zeros([4, 4]))
                else:
                    map_label.append(np.ones([16, 16]))
                    map_1_label.append(np.ones([8, 8]))
                    map_2_label.append(np.ones([4, 4]))
            
            map_label = torch.tensor(map_label, dtype=torch.float32).cuda()
            map_1_label = torch.tensor(map_1_label, dtype=torch.float32).cuda()
            map_2_label = torch.tensor(map_2_label, dtype=torch.float32).cuda()
            
            lables = torch.tensor(lables, dtype=torch.float32)

            absolute_loss = nn.MSELoss()(map_x.float(), map_label)
            absolute_loss_1 = nn.MSELoss()(map_x_1.float(), map_1_label)
            absolute_loss_2 = nn.MSELoss()(map_x_2.float(), map_2_label)
            absolute_loss = absolute_loss + absolute_loss_1 + absolute_loss_2

            spoof_loss = nn.BCELoss()(spoof_out.squeeze(-1).float(), lables.float())

            #spoof_loss = nn.BCELoss()(binary_score.squeeze(-1).float(), lables.float())

            #lables_loss = absolute_loss + 3*spoof_loss
            lables_loss = absolute_loss + 3*spoof_loss
            #lables_loss = spoof_loss
            #feature_loss = nn.MSELoss(reduction='sum')(map_x, feature_map)
            #print(lables_loss)
            #lables_loss = lables_loss.sum(dim=(0))
            #lables_loss = feature_loss

            lables_loss.backward()
            optimizer.step()  

            #loss_feature.update(feature_loss.data, images.size(0))
            loss_lables.update(lables_loss.data, images.size(0))

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, lable_loss= %.4f\n' % (epoch + 1, i + 1, lr,  loss_lables.avg))
        
        # whole epoch average
        print('epoch:%d, Train: lables_loss= %.4f\n' % (epoch + 1, loss_lables.avg))
        log_file.write('epoch:%d, Train: lables_loss= %.4f \n' % (epoch + 1, loss_lables.avg))
        log_file.flush()
                 
        #### validation/test
        if epoch <100:
             epoch_test = 10   
        else:
            epoch_test = 10   
        #epoch_test = 1

        if epoch % epoch_test == epoch_test-1:    # test every 5 epochs  
            model.eval()
            
            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                # val for threshold
                val_data = MyDataset(val_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])]))
                dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for images, lables in dataloader_val:
                    # get the inputs
                    images, lables = images.cuda(), lables.cuda()
                    
                    for X in range(len(lables)):
                        if lables[X] == 1:
                            lables[X] =0
                        else:
                            lables[X] =1
        
                    optimizer.zero_grad()
                    
                    map_x, map_x_1, map_x_2, spoof_out =  model(images)
                    #print(spoof_out)
                    # map_score = (torch.clamp(torch.mean(map_x, dim=(1,2)),0,1)+torch.clamp(torch.mean(map_x_1, dim=(1,2)),0,1)
                    # +torch.clamp(torch.mean(map_x_2, dim=(1,2)),0,1)+spoof_out)/4
                    map_score = spoof_out
                        
                    map_score_list.append('{} {}\n'.format(map_score.item(), lables.item()))
                    #pdb.set_trace()
                map_score_val_filename = args.log+'/'+ args.log+'_map_score_val.txt'
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)                
                
                ###########################################
                '''                test             '''
                ##########################################
                # test for ACC
                test_data = MyDataset(test_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])]))
                dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for images, lables in dataloader_test:
                    # get the inputs
                    images, lables = images.cuda(), lables.cuda()

                    for X in range(len(lables)):
                        if lables[X] == 1:
                            lables[X] =0
                        else:
                            lables[X] =1
        
                    optimizer.zero_grad()
                    

                    map_x, map_x_1, map_x_2, spoof_out  =  model(images)
                    # map_score = (torch.clamp(torch.mean(map_x, dim=(1,2)),0,1)+torch.clamp(torch.mean(map_x_1, dim=(1,2)),0,1)
                    # +torch.clamp(torch.mean(map_x_2, dim=(1,2)),0,1)+spoof_out)/4
                    map_score = spoof_out
                        
                    map_score_list.append('{} {}\n'.format(map_score.item(), lables.item()))
                
                map_score_test_filename = args.log+'/'+ args.log+'_map_score_test.txt'
                with open(map_score_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                #############################################################     
                #       performance measurement both val and test
                #############################################################     
                val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(map_score_val_filename, map_score_test_filename)

                if val_ACC>best_val_acc:
                    best_val_acc=val_ACC
                    torch.save(model.state_dict(),'./checkpoint/depthnet_best_test_2.pt')
                    print('Best test accuracy achieved on epoch: %d'%(epoch + 1))
                torch.save(model.state_dict(),'./checkpoint/depthnet_spoofout_test_2.pt')
                
                print('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (epoch + 1, val_threshold, val_ACC, val_ACER))
                log_file.write('\n epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % (epoch + 1, val_threshold, val_ACC, val_ACER))
              
                print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
                #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
                log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
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
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  #lr 0.0001应该也可以
    parser.add_argument('--batchsize', type=int, default=100, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=50, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--log', type=str, default="depthnet_test_2", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
    
