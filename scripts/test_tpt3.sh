#!/bin/bash

data_root='/data3/tuning/'
testsets=I
#testsets=Flower102
num_important_channel=0
lambda_ape=1.0
lr=0.0001
epoch=20
#arch=RN50
arch=ViT-B/16
bs=32
selection_p=0.1
ctx_init=a_photo_of_a

#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf16shot_softmaxresults_3_32 \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 10 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done
#
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf16shot_softmaxresults_3_32 \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 30 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf16shot_softmaxresults_3_32 \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 100 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done


for nshot in 16
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf16shot_softmaxresults_3_32 \
  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 300 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done



for nshot in 0
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf0shot_softmaxresults_3_32 \
  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 0.3 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 0
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf0shot_softmaxresults_3_32 \
  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 1 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 0
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf0shot_softmaxresults_3_32 \
  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 3 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 0
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf0shot_softmaxresults_3_32 \
  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 10 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 0
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf0shot_softmaxresults_3_32 \
  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 30 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 0
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf0shot_softmaxresults_3_32 \
  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 100 --local_top 1 \
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
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_beta \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position all --shared_param \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_beta \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 7.5 --local_top 1 --position all --shared_param \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_beta \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 10.5 --local_top 1 --position all --shared_param \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done

#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#num_important_channel=0
#lambda_ape=0.3
#lr=0.0001/0.001
#epoch=20/100
#arch=RN50
##arch=ViT-B/16
#bs=1
#selection_p=1
#ctx_init=a_photo_of_a
#
#for nshot in 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log res50_tr_various_channel_lambda_1_1 \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
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
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log debug_res50_tr_nolocal_lambda_1_1_shared_param \
#  --gpu 3 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --shared_param \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done



#data_root='/data3/tuning/'
##testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=eurosat
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_noft_beta55_memW0_3_32 \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5
#done


#data_root='/data3/tuning/'
##testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=eurosat
##arch=RN50
#arch=ViT-B/16
#bs=1
#selection_p=1
#ctx_init=a_photo_of_a
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log debug \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --lr 1e-4 --epoch 200 --ft
#done
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log debug \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --lr 1e-3 --epoch 200 --ft
#done


###### zero shot;
## simple: 42.22; 55.96
## full:   47.73; 59.44
# tip:     48.35; 66.98
#tip_cupl: 55.73; 60.14 (text only, ensembled.)

###### 16 shot: tf, tip, 82.93
###### 16 shot: tr, tip, 20 epoch: 84.28
###### 16 shot: tr, tip, 200 epoch:  83.98; 87.42
###### 16 shot: tr, tip, 1e-3, 200 epoch: 85.85; 87.85

###### 16 shot: tr, tip_cupl, 200 epoch: 83.63; 86.95
###### 16 shot: tr, tip_cupl, 1e-3, 200 epoch: 85.85, 87.23