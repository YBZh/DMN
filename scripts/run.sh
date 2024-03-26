#!/bin/bash

# data_root='/home/notebook/data/group/yabin/clip_tuning_datasets/'
# testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
# num_important_channel=0 ## following APE, 0 denotes using all feature channels without selection.
# lambda_ape=0.3 ## not activated when num_important_channel=0
# lr=0.0001/0.001 ## not activated when --use_searched_param=True
# epoch=20/100  ## not activated when --use_searched_param=True
# arch=ViT-B/16
# bs=32           ## number of views for each test samples following TPT
# selection_p=0.1 ## ratio of selected views to calculate pseudo labels. 0.1 denotes selecting bs*0.1 samples. 
# ctx_init=a_photo_of_a ## not activated in our method

# for nshot in 1 2 4 8 16
# do
#   python ./dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#   -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_searched_vit  \
#   --gpu 0 --n_shot ${nshot} --n_augview 0   --beta 5.5  --use_searched_param \
#   --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
# done


data_root='/home/notebook/data/group/yabin/clip_tuning_datasets/'
testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
num_important_channel=250/400/0 ## not activated when num_important_channel=0
lambda_ape=0.3/0.7 ## not activated when num_important_channel=0
lr=0.0001 ## not activated when num_important_channel=0
epoch=20 ## not activated when num_important_channel=0
#arch=RN50
arch=ViT-B/16
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

for nshot in 0 1 2 4 8 16
do
  python ./dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_tf_searched_vit \
  --gpu 0 --n_shot ${nshot} --n_augview 0   --beta 5.5   --use_searched_param  \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

data_root='/home/notebook/data/group/yabin/clip_tuning_datasets/'
testsets=A/V/R/K
num_important_channel=250/400/0 ## not activated when num_important_channel=0
lambda_ape=0.3/0.7 ## not activated when num_important_channel=0
lr=0.0001 ## not activated when num_important_channel=0
epoch=20 ## not activated when num_important_channel=0
#arch=RN50
arch=ViT-B/16
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

for nshot in 0
do
  python ./dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_tf_shift \
  --gpu 0 --n_shot ${nshot} --n_augview 0   --beta 5.5  --use_searched_param \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done