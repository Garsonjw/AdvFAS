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
from attacks import PGD
import copy

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
    model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3])
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
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_feature = AvgrageMeter()
        loss_lables = AvgrageMeter() 
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = MyDataset(train_list, transform=transforms.Compose([ transforms.ToTensor(),transforms.RandomErasing(), transforms.RandomHorizontalFlip(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)
        i = 0

        for images, lables in dataloader_train:
            i = i+1
            images, lables = images.cuda(), lables.cuda()
            # get the inputs
            attack = PGD.PGD(model, 2)
            adv_images = copy.deepcopy(images)
            adv_lables = copy.deepcopy(lables)
            #print(lables)
            for Y in range(len(adv_lables)):
                if adv_lables[Y] ==1:

                    adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), adv_lables[Y].unsqueeze(0))
                    adv_lables[Y] = 2
                    images = torch.cat((images, adv_imgs), dim=0)
                    lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
            
            #print(lables)

            # print(images.shape)
            # print(lables.shape)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            score = []

            map_x, map_x_1, map_x_2, spoof_out =  model(images)
            

            for X in range(len(lables)):
                if lables[X] == 1:
                    lables[X] = 0   #fake
                elif lables[X] == 0:
                    lables[X] = 1   #true
            
            # loss = nn.NLLLoss()
            # spoof_loss = loss(torch.log(spoof_out), lables)
            spoof_loss = nn.CrossEntropyLoss(weight = torch.tensor([1., 1., 1.]).cuda())(spoof_out, lables)

            spoof_loss.backward()
            optimizer.step()
            scheduler.step()   

            loss_lables.update(spoof_loss.data, images.size(0))# images.size(0)=100

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches

                # log written
                #print(torch.equal(torch.argmax(torch.sigmoid((spoof_out)),dim=1),lables).mean())
                pre_lables = torch.argmax(torch.sigmoid((spoof_out)),dim=1)
                right = torch.eq(pre_lables, lables).sum().item()
                acc = right/len(lables)
                # log written
                print(acc)
            
                itm_lable_1 =0
                itm_lable_right_1 = 0
                itm_lable_0 =0
                itm_lable_right_0 = 0
                itm_lable_2 =0
                itm_lable_right_2 = 0
                right_1 = 0
                right_2 = 0
                right_0 = 0

                for itm_lable in range(len(lables)):
                    if lables[itm_lable] == 1:
                        itm_lable_1 = itm_lable_1 + 1
                        if pre_lables[itm_lable] ==1:
                            itm_lable_right_1 = itm_lable_right_1 + 1#计算两者都等于1的个数
                    elif lables[itm_lable] == 0:
                        itm_lable_0 = itm_lable_0 + 1
                        if pre_lables[itm_lable] ==0:
                            itm_lable_right_0 = itm_lable_right_0 + 1#计算两者都等于0的个数
                    elif lables[itm_lable] == 2:
                        itm_lable_2 = itm_lable_2 + 1
                        if pre_lables[itm_lable] ==2:
                            itm_lable_right_2 = itm_lable_right_2 + 1#计算两者都等于2的个数
                        
                right_1 = itm_lable_right_1/itm_lable_1#正确判别为1的概率
                print(right_1)

                right_0 = itm_lable_right_0/itm_lable_0#正确判别为0的概率
                print(right_0)

                right_2 = itm_lable_right_2/itm_lable_2#正确判别为2的概率
                print(right_2)
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
            val_data = MyDataset(val_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])]))
            dataloader_val = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)
            
            standard_acc_list = []
            robust_acc_list = []
            total_acc_list = []

            itm_lable_1 =0
            itm_lable_right_1 = 0
            itm_lable_0 =0
            itm_lable_right_0 = 0
            itm_lable_2 =0
            itm_lable_right_2 = 0
            right_1 = 0
            right_2 = 0
            right_0 = 0
            acc_list = []

            for images, lables in dataloader_val:
                # get the inputs
                images, lables = images.cuda(), lables.cuda()
                
                #对抗攻击
                attack = PGD.PGD(model, 2)
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)

                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==1:

                        adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), adv_lables[Y].unsqueeze(0))
                        adv_lables[Y] = 2
                        images = torch.cat((images, adv_imgs), dim=0)
                        lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)

                #将原始标签换成常规的标签
                for X in range(len(lables)):
                    if lables[X] == 1:
                        lables[X] =0
                    elif lables[X] ==0:
                        lables[X] =1
    
                optimizer.zero_grad()
                
                map_x, map_x_1, map_x_2, spoof_out =  model(images)
                #print(spoof_out)
                
                pre_lables = torch.argmax(torch.sigmoid((spoof_out)),dim=1)
                right = torch.eq(pre_lables, lables).sum().item()
                
                acc_list.append(right/len(lables))
                # log written
                
                for itm_lable in range(len(lables)):
                    if lables[itm_lable] == 1:
                        itm_lable_1 = itm_lable_1 + 1
                        if pre_lables[itm_lable] ==1:
                            itm_lable_right_1 = itm_lable_right_1 + 1#计算两者都等于1的个数
                    elif lables[itm_lable] == 0:
                        itm_lable_0 = itm_lable_0 + 1
                        if pre_lables[itm_lable] ==0:
                            itm_lable_right_0 = itm_lable_right_0 + 1#计算两者都等于0的个数
                    elif lables[itm_lable] == 2:
                        itm_lable_2 = itm_lable_2 + 1
                        if pre_lables[itm_lable] ==2:
                            itm_lable_right_2 = itm_lable_right_2 + 1#计算两者都等于2的个数
                
            
            right_1 = itm_lable_right_1/itm_lable_1#正确判别为1的概率 真
            print('right_1: =%.4f' % (right_1))
            right_0 = itm_lable_right_0/itm_lable_0#正确判别为0的概率
            print('right_0: =%.4f' % (right_0))
            right_2 = itm_lable_right_2/itm_lable_2#正确判别为2的概率
            print('right_2: =%.4f' % (right_2))
            #map_score = spoof_out
            val_acc = np.mean(acc_list)#总的acc
                #pdb.set_trace()
            val_standard_acc = args.log+'/'+ args.log+'_val_standard_acc.txt'
            val_standard_acc_file = open(val_standard_acc, 'w')
            val_standard_acc_file.write(str((right_0+right_1)/2))
            val_standard_acc_file.flush()

            val_robust_acc = args.log+'/'+ args.log+'_val_robust_acc.txt'
            val_robust_acc_file = open(val_robust_acc, 'w')
            val_robust_acc_file.write(str(right_2))
            val_robust_acc_file.flush()

            val_total_acc = args.log+'/'+ args.log+'_val_total_acc.txt'
            val_total_acc_file = open(val_total_acc, 'w')
            val_total_acc_file.write(str(val_acc))
            val_total_acc_file.flush()  
            
            ###########################################
            '''                test             '''
            ##########################################
            # test for ACC
            test_data = MyDataset(test_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])]))
            dataloader_test = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
            
            map_score_list = []
            itm_lable_1 =0
            itm_lable_right_1 = 0
            itm_lable_0 =0
            itm_lable_right_0 = 0
            itm_lable_2 =0
            itm_lable_right_2 = 0
            right_1 = 0
            right_2 = 0
            right_0 = 0
            acc_list = []

            for images, lables in dataloader_test:
                # get the inputs
                images, lables = images.cuda(), lables.cuda()
                
                #对抗攻击
                attack = PGD.PGD(model, 2)
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)

                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==1:

                        adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), adv_lables[Y].unsqueeze(0))
                        adv_lables[Y] = 2
                        images = torch.cat((images, adv_imgs), dim=0)
                        lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)

                #将原始标签替换成常用标签
                for X in range(len(lables)):
                    if lables[X] == 1:
                        lables[X] =0
                    elif lables[X] ==0:
                        lables[X] =1
    
                optimizer.zero_grad()
                

                map_x, map_x_1, map_x_2, spoof_out  =  model(images)
                #map_score = spoof_out
                pre_lables = torch.argmax(torch.sigmoid((spoof_out)),dim=1)
                right = torch.eq(pre_lables, lables).sum().item()
                acc_list.append(right/len(lables))
                # log written

                for itm_lable in range(len(lables)):
                    if lables[itm_lable] == 1:
                        itm_lable_1 = itm_lable_1 + 1
                        if pre_lables[itm_lable] ==1:
                            itm_lable_right_1 = itm_lable_right_1 + 1#计算两者都等于1的个数
                    elif lables[itm_lable] == 0:
                        itm_lable_0 = itm_lable_0 + 1
                        if pre_lables[itm_lable] ==0:
                            itm_lable_right_0 = itm_lable_right_0 + 1#计算两者都等于0的个数
                    elif lables[itm_lable] == 2:
                        itm_lable_2 = itm_lable_2 + 1
                        if pre_lables[itm_lable] ==2:
                            itm_lable_right_2 = itm_lable_right_2 + 1#计算两者都等于2的个数
                    
            right_1 = itm_lable_right_1/itm_lable_1#正确判别为1的概率 真
            print('right_1: =%.4f' % (right_1))
            right_0 = itm_lable_right_0/itm_lable_0#正确判别为0的概率
            print('right_0: =%.4f' % (right_0))
            right_2 = itm_lable_right_2/itm_lable_2#正确判别为2的概率
            print('right_2: =%.4f' % (right_2))
            
            test_acc = np.mean(acc_list)
            
                #pdb.set_trace()
            test_standard_acc = args.log+'/'+ args.log+'_test_standard_acc.txt'
            test_standard_acc_file = open(test_standard_acc, 'w')
            test_standard_acc_file.write(str((right_0+right_1)/2))
            test_standard_acc_file.flush()

            test_robust_acc = args.log+'/'+ args.log+'_test_robust_acc.txt'
            test_robust_acc_file = open(test_robust_acc, 'w')
            test_robust_acc_file.write(str(right_2))
            test_robust_acc_file.flush()

            test_total_acc = args.log+'/'+ args.log+'_test_total_acc.txt'
            test_total_acc_file = open(test_total_acc, 'w')
            test_total_acc_file.write(str(test_acc))
            test_total_acc_file.flush() 
            
            #############################################################     
            #       performance measurement both val and test
            #############################################################     

            if val_acc>best_val_acc:
                best_val_acc=val_acc
                torch.save(model.state_dict(),'./checkpoint/train_adv_depthnet_best.pt')
                print('Best test accuracy achieved on epoch: %d'%(epoch + 1))
            torch.save(model.state_dict(),'./checkpoint/train_adv_depthnet_last.pt')
            
            print('epoch:%d, Val: val_ACC= %.4f \n' % (epoch + 1, val_acc))
            log_file.write('\n epoch:%d, Val:  val_ACC= %.4f \n' % (epoch + 1, val_acc))
            
            print('epoch:%d, Test:  ACC= %.4f' % (epoch + 1, test_acc))
            #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.write('epoch:%d, Test:  ACC= %.4f \n' % (epoch + 1, test_acc))
            #log_file.write('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f \n\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.flush()

                         
        #if epoch <1:    
        # save the model until the next improvement     
        #    torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pth' % (epoch + 1))

    print('Finished Training')
    log_file.close()
    val_standard_acc_file.close()
    val_robust_acc_file.close()
    val_total_acc_file.close()
    test_standard_acc_file.close()
    test_robust_acc_file.close()
    test_total_acc_file.close()
  

  
 

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=128, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=10, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--log', type=str, default="depthnet_adv", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
    
