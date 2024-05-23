"""adversary.py"""
# from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import copy

class PGD(object):
    '''Projected Gradient Descent'''
    def __init__(self, net, norm, step_size = 2/255,epsilon=8/255, step=8):  #test: self, net, norm, step_size = 0.05019,epsilon=16/255, step=40   step_size=2*eps/step
    #def __init__(self, net, norm, step_size = 4/255,epsilon=16/255, step=8): 
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
            #map_x, map_x_1, map_x_2, netOut = self.net(advimage)
            #map_x, netOut, map_x_1, map_x_2, map_adv, map_adv_1, map_adv_2, adv_out =  self.net(advimage)#for double_final
            #gap, op, op_rgb, op_d= self.net(advimage)
            gap, op, op_rgb, op_d, op_adv= self.net(advimage)#for double_ensemble
            op = nn.Sigmoid()(op)

            loss = nn.BCELoss()(op.squeeze(-1).float(), label.float())
            
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