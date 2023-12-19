#!/bin/bash


data_root='/data3/tuning/'
testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
num_important_channel=0
lambda_ape=1.0
lr=0.0001
epoch=20/100
#arch=RN50
arch=ViT-B/16
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

for nshot in 8 16
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tr_classifier_weights_rerun3_32 \
  --gpu 5 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1  \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
done




#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#num_important_channel=200/350/500
#lambda_ape=0.3/0.7
#lr=0.0001
#epoch=20
##arch=RN50
#arch=ViT-B/16
#bs=1
#selection_p=1
#ctx_init=a_photo_of_a
#
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf_various_channel_lambda_1_1 \
#  --gpu 5 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done

#data_root='/data3/tuning/'
#testsets=I
#num_important_channel=0
#lambda_ape=0.3
#lr=0.0001
#epoch=20
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_vit_tf_onlydynamic_3_32 \
#  --gpu 5 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --mute_static
#done





#data_root='/data3/tuning/'
##testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#arch=RN50
##arch=ViT-B/16
#bs=16
#selection_p=0.2
#ctx_init=a_photo_of_a
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 5 --n_shot 0 --epoch 20 --lr 1e-2 --wd 1e-3 --optimizer adamw --mapping bias --n_augview 16
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 5 --n_shot 1 --epoch 20 --lr 1e-2 --wd 1e-3 --optimizer adamw --mapping bias --n_augview 10
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 5 --n_shot 2 --epoch 20 --lr 1e-2 --wd 1e-3 --optimizer adamw --mapping bias --n_augview 20
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 5 --n_shot 4 --epoch 20 --lr 1e-2 --wd 1e-3 --optimizer adamw --mapping bias --n_augview 40
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 5 --n_shot 8 --epoch 20 --lr 1e-2 --wd 1e-3 --optimizer adamw --mapping bias --n_augview 80
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 5 --n_shot 16 --epoch 20 --lr 1e-2 --wd 1e-3 --optimizer adamw --mapping bias --n_augview 160


