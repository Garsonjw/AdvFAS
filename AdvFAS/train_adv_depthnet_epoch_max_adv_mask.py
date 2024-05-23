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

from utils import AvgrageMeter, accuracy, performances, get_threshold
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
    #model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    model = model.cuda()

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

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

            optimizer.zero_grad()
            
            # forward + backward + optimize
            score = []
            
            for X in range(len(lables)):
                if lables[X] == 1:
                    lables[X] = 0   #fake
                elif lables[X] == 0:
                    lables[X] = 1   #true

            attack = PGD.PGD(model, np.inf)
            adv_images = copy.deepcopy(images)
            adv_lables = copy.deepcopy(lables)
            #print(lables)
            for Y in range(len(adv_lables)):
                if adv_lables[Y] ==0:

                    #adv_imgs = attack.forward(adv_images, adv_lables)
                    adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([1]).cuda())
                    adv_lables[Y] = 2
                    images = torch.cat((images, adv_imgs), dim=0)
                    lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
            
            #shuffle
            index = np.arange(len(lables))
            np.random.shuffle(index)
            images = images[index]
            lables = lables[index]

            map_x, map_x_1, map_x_2, spoof_out =  model(images)

            map_label = []
            mask = []
            for X in range(len(lables)):
                if lables[X] == 0:
                    map_label.append(np.zeros([16, 16]))
                    mask.append(1)
                elif lables[X] == 1: 
                    map_label.append(np.ones([16, 16]))
                    mask.append(1)
                elif lables[X] ==2:
                    map_label.append(np.ones([16, 16]))
                    mask.append(0)

            map_label = torch.tensor(map_label, dtype=torch.float32).cuda()
            mask = torch.tensor(mask, dtype=torch.float32).cuda()
            #print(mask)
            absolute_loss = nn.MSELoss(reduce=False)(map_x.float(), map_label)
            absolute_loss = (absolute_loss.mean(dim=(1,2))*mask).mean()
            #print(absolute_loss)
            spoof_loss = nn.CrossEntropyLoss()(spoof_out, lables)

            spoof_loss = 3*spoof_loss + absolute_loss

            spoof_loss.backward()
            optimizer.step()
            #scheduler.step()   

            loss_lables.update(spoof_loss.data, images.size(0))# images.size(0)=100

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
            val_data = MyDataset(val_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])]))
            dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

            map_score_list = []
            adv_score_list = []
            n = 0
            for images, lables in dataloader_val:
                # get the inputs
                images, lables = images.cuda(), lables.cuda()
                
                #将原始标签换成常规的标签
                for X in range(len(lables)):
                    if lables[X] == 1:
                        lables[X] =0#fake
                    elif lables[X] ==0:
                        lables[X] =1#real
                #对抗攻击
                attack = PGD.PGD(model, np.inf, step_size = 0.05019,epsilon=16/255, step=40)
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)

                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==0:

                        adv_imgs = attack.forward(adv_images, torch.tensor([1]).cuda())
                        images = torch.cat((images, adv_imgs), dim=0)
                        lables = torch.cat((lables, adv_lables), dim=0)

                optimizer.zero_grad()
                
                map_x, map_x_1, map_x_2, spoof_out =  model(images)

                if torch.argmax(spoof_out[0], dim=0).item() !=2:
                ###保存分数和adv分数、原始标签
                    map_score_list.append('{} {}\n'.format(torch.softmax(spoof_out, dim=1)[0][1].item(), lables[0].item()))
                else:
                    adv_score_list.append('{}\n'.format(lables[0].item()))
                
                if lables[0] ==0:
                    if torch.argmax(spoof_out[1], dim=0).item() !=2:
                        map_score_list.append('{} {}\n'.format(torch.softmax(spoof_out, dim=1)[1][1].item(), lables[1].item()))
                    else:
                         adv_score_list.append('{}\n'.format(lables[1].item())) 

                print('wait') 

            adv_filename = args.log+'/'+args.log +'adv_val.txt'
            with open(adv_filename, 'w') as file:
                file.writelines(adv_score_list)    

            score_spoof_val_filename = args.log+'/'+args.log+'score_spoof_val.txt'
            with open(score_spoof_val_filename, 'w') as file:
                file.writelines(map_score_list)

            ###########################################
            '''                test             '''
            ###########################################
            # test for ACC
            test_data = MyDataset(test_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])]))
            dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
            
            map_score_list = []
            adv_score_list = []
            for images, lables in dataloader_test:
                # get the inputs
                images, lables = images.cuda(), lables.cuda()

                #将原始标签替换成常用标签
                for X in range(len(lables)):
                    if lables[X] == 1:
                        lables[X] =0
                    elif lables[X] ==0:
                        lables[X] =1
                
                #对抗攻击
                attack = PGD.PGD(model, np.inf, step_size = 0.05019,epsilon=16/255, step=40)
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)

                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==0:

                        adv_imgs = attack.forward(adv_images, torch.tensor([1]).cuda())
                        images = torch.cat((images, adv_imgs), dim=0)
                        lables = torch.cat((lables, adv_lables), dim=0)
    
                optimizer.zero_grad()
                
                map_x, map_x_1, map_x_2, spoof_out  =  model(images)
                #map_score = spoof_out
                if torch.argmax(spoof_out[0], dim=0).item() !=2:
                ###保存分数和adv分数、原始标签
                    map_score_list.append('{} {}\n'.format(torch.softmax(spoof_out, dim=1)[0][1].item(), lables[0].item()))
                else:
                    adv_score_list.append('{}\n'.format(lables[0].item()))
                
                if lables[0] ==0:
                    if torch.argmax(spoof_out[1], dim=0).item() !=2:
                        map_score_list.append('{} {}\n'.format(torch.softmax(spoof_out, dim=1)[1][1].item(), lables[1].item()))
                    else:
                         adv_score_list.append('{}\n'.format(lables[1].item())) 
                
                print('wait')
            adv_filename = args.log+'/'+args.log +'adv_test.txt'
            with open(adv_filename, 'w') as file:
                file.writelines(adv_score_list)  

            score_spoof_test_filename = args.log+'/'+args.log+'score_spoof_test.txt'
            with open(score_spoof_test_filename, 'w') as file:
                file.writelines(map_score_list)    
                
                # log written
            
            #############################################################     
            #       performance measurement both val and test
            #############################################################     
            
            val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(score_spoof_val_filename, score_spoof_test_filename)
            
            if val_ACC>best_val_acc:
                best_val_acc=val_ACC
                torch.save(model.state_dict(),'./checkpoint/train_adv_depthnet_epoch_max_adv_best.pt')
                print('Best test accuracy achieved on epoch: %d'%(epoch + 1))
            torch.save(model.state_dict(),'./checkpoint/train_adv_depthnet_epoch_max_adv_last.pt')
            
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
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=100, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=50, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--log', type=str, default="depthnet_adv-epoch_max_adv_mask_test3", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
    
