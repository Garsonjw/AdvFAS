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
import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize

from models.depthnet_ensemble import Depthnet
from models.CDCNs import Conv2d_cd, CDCN, CDCNpp
from models.resnet_ensemble import Resnet18, Resnet50
#多卡
import argparse
from torch.utils.data.distributed import DistributedSampler

from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
from dataset import MyDataset

from utils import AvgrageMeter, accuracy, performances, get_threshold, performances_my, performances_my_adv
from tqdm import tqdm
from attacks import PGD, MIFGSM, DIFGSM
from attacks.facesec.attack import utgt_occlusion
from attacks.facesec.utils import load_mask

import copy

# Dataset root
train_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Train_images/'        
val_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Dev_images/'     
test_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Test_images/'   

# train_list = '/data/home/scv7305/run/yanghao/fasdata/WMCA/WMCA_train.txt'
# val_list = '/data/home/scv7305/run/yanghao/fasdata/WMCA/WMCA_dev.txt'
# test_list =  '/data/home/scv7305/run/yanghao/fasdata/WMCA/WMCA_eval.txt'

train_list = '/data/home/scv7305/run/chenjiawei/fasdata/crop_casia/train.txt'
val_list = '/data/home/scv7305/run/chenjiawei/fasdata/crop_casia/dev.txt'
test_list =  '/data/home/scv7305/run/chenjiawei/fasdata/crop_casia/eval.txt'

# def del_tensor_ele(arr,index):
#     arr1 = arr[0:index]
#     arr2 = arr[index+1:]
#     return torch.cat((arr1,arr2),dim=0)

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

    #单卡 
    model = Depthnet()
    #model = Resnet18()
    #model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
    #model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    model = model.cuda()
    #多卡
    # if args.local_rank != -1:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=5400))
    # model = Depthnet()#创建模型
    # model.to(device)
    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    #     #logger.info('use {} gpus!'.format(num_gpus))
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True,
    #                                             output_device=args.local_rank)


    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    #criterion_absolute_loss = nn.MSELoss().cuda()

    print(model) 
    
    
    #bandpass_filter_numpy = build_bandpass_filter_numpy(30, 30)  # fs, order  # 61, 64 

    ACER_save = 1.0
    best_adv_acc = -1

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

        ##多卡
        # train_sampler = DistributedSampler(train_data)
        # dataloader_train = DataLoader(train_data, sampler=train_sampler, batch_size=args.batchsize, num_workers=4)
        # train_sampler.set_epoch(epoch)

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

            if args.attack == 'patch':
            #attack eyeglasses
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)
                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==0:
                        #adv_imgs = attack.forward(adv_images, adv_lables)
                        mask = load_mask('eyeglass').cuda()
                        torch_size = Resize([128, 128])
                        mask = torch_size(mask)
                        delta = utgt_occlusion('closed', model, adv_images[Y].unsqueeze(0), torch.tensor([0]), mask, univ=False)
                        adv_images[Y] = adv_images[Y]*(1-mask) + delta
                        adv_lables[Y] = 2
                        images = torch.cat((images, adv_images[Y].unsqueeze(0)), dim=0)
                        lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)  

            if args.attack == 'PGD':
                attack = PGD.PGD(model, np.inf)
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)
                #print(lables)
                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==0:

                        #adv_imgs = attack.forward(adv_images, adv_lables)
                        adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                        adv_lables[Y] = 2
                        images = torch.cat((images, adv_imgs), dim=0)
                        lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
            
            if args.attack == 'MI-FGSM':
                attack = MIFGSM.MIFGSM(model, eps=8/255, steps=10, decay =1.0)
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)
                #print(lables)
                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==0:

                        #adv_imgs = attack.forward(adv_images, adv_lables)
                        adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                        adv_lables[Y] = 2
                        images = torch.cat((images, adv_imgs), dim=0)
                        lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
            
            if args.attack == 'DI-FGSM':
                attack = DIFGSM.DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
                adv_images = copy.deepcopy(images)
                adv_lables = copy.deepcopy(lables)
                #print(lables)
                for Y in range(len(adv_lables)):
                    if adv_lables[Y] ==0:

                        #adv_imgs = attack.forward(adv_images, adv_lables)
                        adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                        adv_lables[Y] = 2
                        images = torch.cat((images, adv_imgs), dim=0)
                        lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
            
            #shuffle
            index = np.arange(len(lables))
            np.random.shuffle(index)
            images = images[index]
            lables = lables[index]

            map_x, spoof_out, map_x_1, map_x_2, adv_out =  model(images)

            map_label = []
            map_1_label = []
            map_2_label = []
            lables_live = []
            #spoof_map
            no_adv_num = 0
            mask = []
            for X in range(len(lables)):
                if lables[X] == 0:
                    map_label.append(np.zeros([16, 16]))
                    map_1_label.append(np.zeros([8, 8]))
                    map_2_label.append(np.zeros([4, 4]))
                    lables_live.append(0)
                    no_adv_num = no_adv_num +1
                    mask.append(1)

                elif lables[X] == 1: 
                    map_label.append(np.ones([16, 16]))
                    map_1_label.append(np.ones([8, 8]))
                    map_2_label.append(np.ones([4, 4]))
                    lables_live.append(1)
                    no_adv_num = no_adv_num +1
                    mask.append(1)

                elif lables[X] == 2:
                    map_label.append(np.zeros([16, 16]))
                    map_1_label.append(np.zeros([8, 8]))
                    map_2_label.append(np.zeros([4, 4]))
                    lables_live.append(0)
                    mask.append(0)
                        #spoof_map
            #adv的lable

            map_label = torch.tensor(map_label, dtype=torch.float32).cuda()
            map_1_label = torch.tensor(map_1_label, dtype=torch.float32).cuda()
            map_2_label = torch.tensor(map_2_label, dtype=torch.float32).cuda()
            mask = torch.tensor(mask, dtype=torch.float32).cuda()

            lables_live = torch.tensor(lables_live, dtype=torch.float32).cuda()

            map_1_score = (torch.clamp(torch.mean(map_x, dim=(1,2)),0,1)+torch.clamp(torch.mean(map_x_1, dim=(1,2)),0,1)
            +torch.clamp(torch.mean(map_x_2, dim=(1,2)),0,1))
            spoof_out = spoof_out.squeeze(1)
            map_score = (map_1_score+spoof_out)/4
            # print(map_score)
            # print(lables_live)

            score_xiu = map_score.clone() 
            score_xiu = score_xiu.detach()
            mask_divide = score_xiu.ge(0.5)
            #print(score_xiu)

            correct_index = torch.where(mask_divide == lables_live)[0]
            error_index = torch.where((~mask_divide) ==lables_live)[0]
            
            #print(mask_divide)
            #print(lables_live)
            #删除lable为1的error_index
            error_index_lable1 = []
            #print(error_index)
            for index in error_index:
                if lables_live[index] == 1:
                    error_index_lable1.append(index.item())
            
            error_index_lable1 = torch.tensor(error_index_lable1, dtype=torch.int64)
            error_index = error_index[~np.isin(error_index.cpu(), error_index_lable1)]
            error_index_lable1 = error_index_lable1.cuda()

            #print(lables[error_index])
            #print(error_index_lable1)

            map_score[correct_index] = map_score[correct_index].detach()

            #没有用1拉
            # map_score[error_index_lable1] = map_score[error_index_lable1].detach()

            # score_xiu[error_index] = 0

            #用1拉
            score_xiu[error_index] = 0
            score_xiu[error_index_lable1] = 1

            loss_adv = nn.BCELoss()(map_score.squeeze(-1).float()*adv_out.squeeze(-1).float(), score_xiu.float()) 

            #absolute_loss = nn.MSELoss()(map_x.float(), map_label)
            absolute_loss = nn.MSELoss(reduce = False, reduction='none')(map_x.float(), map_label)

            absolute_loss = (absolute_loss.mean(dim=(1,2))*mask).sum()/float(no_adv_num)

            absolute_loss_1 = nn.MSELoss(reduce = False, reduction='none')(map_x_1.float(), map_1_label)
            absolute_loss_1 = (absolute_loss_1.mean(dim=(1,2))*mask).sum()/float(no_adv_num)
            
            absolute_loss_2 = nn.MSELoss(reduce = False, reduction='none')(map_x_2.float(), map_2_label)
            absolute_loss_2 = (absolute_loss_2.mean(dim=(1,2))*mask).sum()/float(no_adv_num)

            absolute_loss = absolute_loss + absolute_loss_1 + absolute_loss_2
            
            #absolute_loss = (absolute_loss.mean(dim = (1,2))*mask).sum()/float(live_num)
           
            # print(lables)
            # print(torch.clamp(torch.mean(map_adv, dim=(1,2)),0,1))

            spoof_loss = nn.BCELoss(reduce = False, reduction='none')(spoof_out.squeeze(-1).float(), lables_live.float())
            # print(map_score)
            # print(map_score.squeeze(-1).float()*adv_out.squeeze(-1).float())
            # print(lables)
            spoof_loss = (spoof_loss*mask).sum()/float(no_adv_num)

            spoof_loss = 3*spoof_loss + absolute_loss + loss_adv

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

            #多卡
            # val_sampler = DistributedSampler(val_data)
            # dataloader_val = DataLoader(val_data, sampler=val_sampler, batch_size=1, num_workers=4)

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
                    elif lables[X] == 0:
                        lables[X] =1#real
                #对抗攻击
                if args.attack == 'patch':
                #attack eyeglasses
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:
                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            mask = load_mask('eyeglass').cuda()
                            torch_size = Resize([128, 128])
                            mask = torch_size(mask)
                            delta = utgt_occlusion('closed', model, adv_images[Y].unsqueeze(0), torch.tensor([0]), mask, univ=False)
                            adv_images[Y] = adv_images[Y]*(1-mask) + delta
                            images = torch.cat((images, adv_images[Y].unsqueeze(0)), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)  

                if args.attack == 'PGD':
                    attack = PGD.PGD(model, np.inf)
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    #print(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:

                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                            images = torch.cat((images, adv_imgs), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
                            
                if args.attack == 'MI-FGSM':
                    attack = MIFGSM.MIFGSM(model, eps=8/255, steps=10, decay =1.0)
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    #print(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:

                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                            images = torch.cat((images, adv_imgs), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)

                if args.attack == 'DI-FGSM':
                    attack = DIFGSM.DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    #print(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:

                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                            images = torch.cat((images, adv_imgs), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)

                optimizer.zero_grad()
                
                map_x, spoof_out, map_x_1, map_x_2, adv_out =  model(images)
                #print(adv_out)
                
                map_1_score = (torch.clamp(torch.mean(map_x, dim=(1,2)),0,1)+torch.clamp(torch.mean(map_x_1, dim=(1,2)),0,1)
                +torch.clamp(torch.mean(map_x_2, dim=(1,2)),0,1))
                spoof_out = spoof_out.squeeze(1)
                map_score = (map_1_score+spoof_out)/4
                socre_all = map_score.squeeze(-1).float()*adv_out.squeeze(-1).float()

            ###保存分数和adv分数、原始标签
                map_score_list.append('{} {}\n'.format(socre_all[0].item(), lables[0].item()))

                if lables[0] ==0:
                ###保存分数和adv分数、原始标签
                    map_score_list.append('{} {}\n'.format(socre_all[1].item(), lables[1].item()))

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

            ##多卡
            # test_sampler = DistributedSampler(test_data)
            # dataloader_test = DataLoader(test_data, sampler=test_sampler, batch_size=1, num_workers=4)
            
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
                if args.attack == 'patch':
                #attack eyeglasses
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:
                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            mask = load_mask('eyeglass').cuda()
                            torch_size = Resize([128, 128])
                            mask = torch_size(mask)
                            delta = utgt_occlusion('closed', model, adv_images[Y].unsqueeze(0), torch.tensor([0]), mask, univ=False)
                            adv_images[Y] = adv_images[Y]*(1-mask) + delta
                            images = torch.cat((images, adv_images[Y].unsqueeze(0)), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)  

                if args.attack == 'PGD':
                    attack = PGD.PGD(model, np.inf)
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    #print(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:
                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                            images = torch.cat((images, adv_imgs), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)

                if args.attack == 'MI-FGSM':
                    attack = MIFGSM.MIFGSM(model, eps=8/255, steps=10, decay =1.0)
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    #print(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:

                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                            images = torch.cat((images, adv_imgs), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)

                if args.attack == 'DI-FGSM':
                    attack = DIFGSM.DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
                    adv_images = copy.deepcopy(images)
                    adv_lables = copy.deepcopy(lables)
                    #print(lables)
                    for Y in range(len(adv_lables)):
                        if adv_lables[Y] ==0:

                            #adv_imgs = attack.forward(adv_images, adv_lables)
                            adv_imgs = attack.forward(adv_images[Y].unsqueeze(0), torch.tensor([0]).cuda())
                            images = torch.cat((images, adv_imgs), dim=0)
                            lables = torch.cat((lables, adv_lables[Y].unsqueeze(0)), dim=0)
    
                optimizer.zero_grad()
                
                map_x, spoof_out, map_x_1, map_x_2, adv_out =  model(images)
                map_1_score = (torch.clamp(torch.mean(map_x, dim=(1,2)),0,1)+torch.clamp(torch.mean(map_x_1, dim=(1,2)),0,1)
                +torch.clamp(torch.mean(map_x_2, dim=(1,2)),0,1))
                spoof_out = spoof_out.squeeze(1)
                map_score = (map_1_score+spoof_out)/4

                socre_all = map_score.squeeze(-1).float()*adv_out.squeeze(-1).float()

            ###保存分数和adv分数、原始标签
                map_score_list.append('{} {}\n'.format(socre_all[0].item(), lables[0].item()))

                if lables[0] ==0:
                ###保存分数和adv分数、原始标签
                    adv_score_list.append('{} {}\n'.format(socre_all[1].item(), lables[1].item()))

            adv_filename_test = args.log+'/'+args.log+'adv_test.txt'
            with open(adv_filename_test, 'w') as file:
                file.writelines(adv_score_list)  

            score_spoof_test_filename = args.log+'/'+args.log+'score_spoof_test.txt'
            with open(score_spoof_test_filename, 'w') as file:
                file.writelines(map_score_list)    
                
                # log written
            
            #############################################################     
            #       performance measurement both val and test
            #############################################################     
            
            val_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER = performances(score_spoof_val_filename, score_spoof_test_filename)

            val_threshold_adv, val_ACC_adv, val_ACER_adv, test_ACC_adv, test_APCER_adv, test_BPCER_adv, test_ACER_adv = performances(score_spoof_val_filename, adv_filename_test)
            
            if test_ACC_adv>best_adv_acc:
                best_adv_acc=test_ACC_adv
                torch.save(model.state_dict(),'./checkpoint/train_adv_'+args.model+'_couple_'+args.attack+'_1_best_casia.pt')
                print('Best test accuracy achieved on epoch: %d'%(epoch + 1))
            torch.save(model.state_dict(),'./checkpoint/train_adv_'+args.model+'_couple_'+args.attack+'_1_last_casia.pt')
            #存每一轮的checkpoint
            #torch.save(model.state_dict(),'./checkpoint_PGD/train_adv_'+args.model+'_couple_'+args.attack+'epoch'+str(epoch+1)+'.pt')
            
            print('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (epoch + 1, val_threshold, val_ACC, val_ACER))
            log_file.write('\n epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % (epoch + 1, val_threshold, val_ACC, val_ACER))
            
            print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
            #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))

            print('epoch:%d, Adv_Val:  val_threshold_adv= %.4f, val_ACC_adv= %.4f, val_ACER_adv= %.4f' % (epoch + 1, val_threshold_adv, val_ACC_adv, val_ACER_adv))
            log_file.write('\n epoch:%d, Adv_Val:  val_threshold_adv= %.4f, val_ACC_adv= %.4f, val_ACER_adv= %.4f \n' % (epoch + 1, val_threshold_adv, val_ACC_adv, val_ACER_adv))
            
            print('epoch:%d, Test:  adv_ACC= %.4f, adv_APCER= %.4f, adv_BPCER= %.4f, ACER= %.4f' % (epoch + 1, test_ACC_adv, test_APCER_adv, test_BPCER_adv, test_ACER_adv))
            #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC_adv, test_APCER_adv, test_BPCER_adv, test_ACER_adv))
            #log_file.write('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f \n\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
            log_file.flush()

                         
        #if epoch <1:    
        # save the model until the next improvement     
        #    torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pth' % (epoch + 1))

    print('Finished Training')
    log_file.close()
  

  
 

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=100, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=50, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    #parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)#多卡

    parser.add_argument('--model', type=str, default='Depthnet()', help='attack')#Depthnet() Resnet18() Resnet50()
    parser.add_argument('--attack', type=str, default='PGD', help='model')#PGD   patch MI-FGSM DI-FGSM
    parser.add_argument('--log', type=str, default="test", help='log and save model name')

    args = parser.parse_args()
    train_test()
    
