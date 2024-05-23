#!/bin/bash
# officehome
#data_dir=
#CUDA_VISIBLE_DEVICES=0 python train_all.py OH0 --dataset OfficeHome --deterministic \
#--trial_seed 0 --steps 3000 --checkpoint_freq 300 --data_dir $data_dir
module load anaconda/2020.11
source activate SSDG

# PACS
#data_dir=
python -u train_adv_depthnet_two_class_epoch=10_div_real_adv.py --attack MI-FGSM-AT --model Depthnet_MI-FGSM-AT_casia --log Depthnet__two_class_MI-FGSM-AT_casia
# TerraIncognita
#data_dir=
#CUDA_VISIBLE_DEVICES=0 python train_all.py TR0 --dataset TerraIncognita --deterministic \
#--trial_seed 0 --checkpoint_freq 1000 --steps 5000 --data_dir $data_dir
