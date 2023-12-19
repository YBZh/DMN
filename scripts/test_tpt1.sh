#!/bin/bash

data_root='/data3/tuning/'
testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
num_important_channel=0
lambda_ape=1.0
lr=0.0001
epoch=20
#arch=RN50
arch=ViT-B/16
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

for nshot in 4 8 16
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf_classifier_weights_rerun_3_32 \
  --gpu 1 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done






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


#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 1 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position output  \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done

#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 1 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position vo  \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done

#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 1 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position qkv  --shared_param \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 1 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position all --shared_param \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done






#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#num_important_channel=0
#lambda_ape=0.3
#lr=0.0001/0.001
#epoch=20/100
##arch=RN50
#arch=ViT-B/16
#bs=1
#selection_p=1
#ctx_init=a_photo_of_a
#
#for nshot in 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tr_various_channel_lambda_1_1 \
#  --gpu 1 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done


#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
##testsets=Flower102/DTD
#num_important_channel=0
#lambda_ape=0.3
#lr=0.0001/0.001
#epoch=20/100
#arch=RN50
##arch=ViT-B/16
#bs=1
#selection_p=1.0
#ctx_init=a_photo_of_a
#
#for nshot in 16 4 1
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log debug_res50_tr_nolocal_lambda_1_1 \
#  --gpu 1 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done


#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
##testsets=Flower102/DTD
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_ft_beta55_memW0_3_32_lr1e3 \
#  --gpu 1 --n_shot ${nshot} --n_augview 0  --ft --static_mem_weight 0.0 --beta 5.5 --lr 1e-3 --epoch 20
#done







