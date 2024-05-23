"""adversary.py"""
# from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import copy

class PGD(object):
    '''Projected Gradient Descent'''
    #def __init__(self, net, norm, step_size = 2/255,epsilon=8/255, step=8):  #test: self, net, norm, step_size = 0.05019,epsilon=16/255, step=40   step_size=2*eps/step
    def __init__(self, net, norm, step_size = 1/255,epsilon=6/255, step=12): 
        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = step
        self.p = norm
        self.net = copy.deepcopy(net)
        self.net.eval()
  
    def forward(self, image, label):
        image, label = image.cuda(), label.cuda()

        batchsize = image.shape[0]
        # random start
        delta = torch.rand_like(image)*2*self.epsilon-self.epsilon
        if self.p!=np.inf: # projected into feasible set if needed
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1) #self
            mask = normVal<=self.epsilon
            scaling = self.epsilon/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = image+delta
        

        for i in range(self.steps):
            advimage = advimage.clone().detach().requires_grad_(True) # clone the advimage as the next iteration input
            
            #_, netOut, _,_ = self.net(advimage)
            map_x, map_x_1, map_x_2, netOut = self.net(advimage)
            #map_x, netOut, map_x_1, map_x_2, map_adv, map_adv_1, map_adv_2, adv_out =  self.net(advimage)#for double_final
            #map_x, netOut, map_x_1, map_x_2, adv_out =  self.net(advimage)#for double_ensemble

            map_label = []
            map_1_label = []
            map_2_label = []

            map_label.append(np.zeros([16, 16]))
            map_1_label.append(np.zeros([8, 8]))
            map_2_label.append(np.zeros([4, 4]))
            
            map_label = torch.tensor(map_label, dtype=torch.float32).cuda()
            map_1_label = torch.tensor(map_1_label, dtype=torch.float32).cuda()
            map_2_label = torch.tensor(map_2_label, dtype=torch.float32).cuda()

            absolute_loss = nn.MSELoss()(map_x.float(), map_label)
            absolute_loss_1 = nn.MSELoss()(map_x_1.float(), map_1_label)
            absolute_loss_2 = nn.MSELoss()(map_x_2.float(), map_2_label)
            
            loss_spoof = nn.BCELoss()(netOut.squeeze(-1).float(), label.float())
            loss = loss_spoof + absolute_loss + absolute_loss_1 + absolute_loss_2
            # _, _, _,netOut = self.net(advimage)
            # loss = -nn.CrossEntropyLoss()(netOut, label)
            
            updates = torch.autograd.grad(loss, [advimage])[0].detach()
            if self.p==np.inf:
                updates = updates.sign()
            else:
                normVal = torch.norm(updates.view(batchsize, -1), self.p, 1)
                updates = updates/normVal.view(batchsize, 1, 1, 1)
            updates = updates * self.step_size
            advimage = advimage+updates
            # project the disturbed image to feasible set if needed
            delta = advimage-image
            if self.p==np.inf:
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:
                normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(batchsize, 1, 1, 1)
            advimage = image+delta
            
            advimage = torch.clamp(advimage, -1, 1) # cifar10(-1,1)
            
        return advimage