#!/bin/bash

data_root='/data3/tuning/'
testsets=Food101
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
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log debug_food \
  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
done

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt simple  --log debug_food \
  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
done

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip  --log debug_food \
  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
done

for nshot in 4
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt full  --log debug_food \
  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
done



#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#num_important_channel=0
#lambda_ape=1.0
#lr=0.0001
#epoch=20
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 0 1 2
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf_classifier_weights_rerun_3_32 \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
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
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position query \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done

#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position key  \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position value  \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done
#
#for nshot in 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_projection.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_position_qkvaffine \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 --position qkv  \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done



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
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 70 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
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
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 100 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done

#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#num_important_channel=0
#lambda_ape=0.3
#lr=0.0001/0.001
#epoch=20/100
##arch=RN50
#arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tr_various_channel_lambda_3_32 \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr} --ft
#done


#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
##testsets=Flower102/DTD
#arch=RN50
##arch=ViT-B/16
#bs=32
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#for nshot in 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ft_beta55_memW01_3_32 \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --ft --static_mem_weight 0.1 --beta 5.5
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
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_ft_beta55_memW0_3_32_lr1e4 \
#  --gpu 0 --n_shot ${nshot} --n_augview 0  --ft --static_mem_weight 0.0 --beta 5.5 --lr 1e-4 --epoch 20
#done





#data_root='/data3/tuning/'
#testsets=Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#arch=RN50
#bs=64
#selection_p=0.1
#ctx_init=a_photo_of_a
#
#python ./fixclip_memory_bank_tfree_fewshot.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 1.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 0 --weight_maxent 1.0 --memory_size 50 --text_prompt tip

### beta 越小，曲线越平滑； beta 越大，曲线越

#data_root='/data3/tuning/'
#testsets=I/A/V/R/K
#arch=RN50
##arch=ViT-B/16
#bs=64
#ctx_init=a_photo_of_a
#
#python ./tpt_classification_v1.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 0





#data_root='/data3/tuning/'
#testsets=Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#arch=RN50
##arch=ViT-B/16
#bs=64
#ctx_init=a_photo_of_a
#
#python ./tpt_classification_memory_bank_tfree.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 0 --weight_maxent 1.0 --memory_size 50 --log few_tfree_RN50



#data_root='/data3/tuning/'
#coop_weight='/data3/tuning/to_gdrive/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
#testsets=I/A/V/R/K
#arch=RN50
##arch=ViT-B/16
#bs=64
#ctx_init=a_photo_of_a
#
#python ./tpt_classification_memory_bank_tfree.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 0 --load ${coop_weight} --memory_size 50 --log imagenet_tfree_RN50_coco



#data_root='/data3/tuning/'
#testsets=A
#arch=RN50
## arch=ViT-B/16
#bs=64
#ctx_init=a_photo_of_a
#
#python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --gpu 0 \
#--tpt --ctx_init ${ctx_init}


#data_root='/data3/tuning/'
#testsets=I
#arch=RN50
## arch=ViT-B/16
#bs=64
#ctx_init=a_photo_of_a
#
#python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --gpu 0 \
#--tpt --ctx_init ${ctx_init}