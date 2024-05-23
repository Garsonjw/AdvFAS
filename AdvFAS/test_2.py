import torch

a = torch.load('/data/home/scv7305/run/chenjiawei/adv_branch/CDCN/checkpoint/train_adv_depthnet_couple_0_bestPGD.pt')
# print(type(a[0]))
for k, v in a.items():
        print(f"{k}: {v.shape}")
        #if epoch <1:    
        # save the model until the next improvement     
        #    torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pth' % (epoch + 1))

