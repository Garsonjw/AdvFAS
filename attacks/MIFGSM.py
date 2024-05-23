import torch
import torch.nn as nn
import numpy as np

class MIFGSM(object):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.net = model

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        momentum = torch.zeros_like(images).detach().cuda()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            #_, netOut, _,_ = self.net(advimage)
            #map_x, map_x_1, map_x_2, netOut = self.net(adv_images)
            map_x, netOut, map_x_1, map_x_2, adv_out =  self.net(adv_images)#for double_ensemble

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
            
            loss_spoof = nn.BCELoss()(netOut.squeeze(-1).float(), labels.float())
            loss = loss_spoof + absolute_loss + absolute_loss_1 + absolute_loss_2

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
