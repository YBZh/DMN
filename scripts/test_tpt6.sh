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

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf4shot_softmaxresults_3_32 \
  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 1 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf4shot_softmaxresults_3_32 \
  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 3 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf4shot_softmaxresults_3_32 \
  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 10 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf4shot_softmaxresults_3_32 \
  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 30 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf4shot_softmaxresults_3_32 \
  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 100 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done


#data_root='/data3/tuning/'
#testsets=I
#num_important_channel=500/700/900
#lambda_ape=0.3/0.7
#lr=0.0001
#epoch=20
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 0
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 1 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done
#
#data_root='/data3/tuning/'
#testsets=I
#num_important_channel=500/700/900
#lambda_ape=0.3/0.7
#lr=0.0001
#epoch=20
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 0
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 3 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done
#
#data_root='/data3/tuning/'
#testsets=I
#num_important_channel=500/700/900
#lambda_ape=0.3/0.7
#lr=0.0001
#epoch=20
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 0
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 5 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done

#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#num_important_channel=500/700/900
#lambda_ape=0.3/0.7
#lr=0.0001
#epoch=20
#arch=RN50
##arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log res50_tf_various_channel_lambda_3_32 \
#  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done


############### 后面写代码搜一下不同 的 channel number: 500, 600, 700, 800, 900, 1000; 以及不同的lambda: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
###### 把代码结构改一下，为每个setting 单独搜，并把结果最好的额保留下来。！！ 去实现，在testsets 里再套几个循环，然后把最好的结果记录下来。

#testsets=Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=I/A/V/R/K

#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
##testsets=Flower102/DTD
#arch=RN50
##arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log noft_beta55_memW005_3_32 \
#  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.05 --beta 5.5
#done
#
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log noft_beta55_memW01_3_32 \
#  --gpu 6 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.1 --beta 5.5
#done



#data_root='/data3/tuning/'
#testsets=R
#arch=RN50
##arch=ViT-B/16
#bs=64
#ctx_init=a_photo_of_a
#
#python ./tpt_classification_memory_bank_tfree_localc.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 6 --memory_size 50



#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 6 --weight_maxent 0.1 --log logv
#
#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 6 --weight_maxent 0.01 --log logv
#
#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 6 --weight_maxent 0.001 --log logv
#
#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 6 --weight_maxent 0.0001 --log logv

#data_root='/data3/tuning/'
#testsets=I
#arch=RN50
## arch=ViT-B/16
#bs=64
#ctx_init=a_photo_of_a
#
#python ./tpt_classification_v3.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --gpu 3 \
#--tpt --ctx_init ${ctx_init}