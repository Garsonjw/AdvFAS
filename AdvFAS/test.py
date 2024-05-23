

from utils import performances

score_spoof_val_filename = '/data/home/scv7305/run/chenjiawei/adv_branch/CDCN/Depthnet_adv_couple_1_casia_PGD/Depthnet_adv_couple_1_casia_PGDscore_spoof_val.txt'
#score_spoof_test_filename = '/data/home/scv7305/run/chenjiawei/adv_branch/CDCN/Depthnet_eps=_two_class_PGD-AT_vail/Depthnet_eps=_two_class_PGD-AT_vailscore_spoof_test.txt'
score_spoof_test_filename = '/data/home/scv7305/run/chenjiawei/adv_branch/CDCN/Depthnet_adv_couple_1_casia_PGD/AUC.txt'

            
val_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER = performances(score_spoof_val_filename, score_spoof_test_filename)

print('val_threshold:  val_threshold= %.4f' % (val_threshold))          
            
print('Val:  val_ACC= %.4f' % (val_ACC))

print('Test:  ACC= %.4f' % (test_ACC))

print('val_ACER:  val_ACER= %.4f' % (val_ACER))

print('test_APCER:  test_APCER= %.4f' % (test_APCER))

print('test_BPCER:  test_BPCER= %.4f' % (test_BPCER))
                         
        #if epoch <1:    
        # save the model until the next improvement     
        #    torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pth' % (epoch + 1))

