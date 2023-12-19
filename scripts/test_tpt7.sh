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

for nshot in 1
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf1shot_softmaxresults_3_32 \
  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 1 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 1
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf1shot_softmaxresults_3_32 \
  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 3 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 1
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf1shot_softmaxresults_3_32 \
  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 10 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 1
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf1shot_softmaxresults_3_32 \
  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 30 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done

for nshot in 1
do
  python ./fixclip_memory_bank_ft_fewshot_ape_textfull_nolocal_softmax.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log vit_tf1shot_softmaxresults_3_32 \
  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 100 --local_top 1 \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done


##!/bin/bash
#
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
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 10 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
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
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 30 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
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
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log ablation_vit_tf_memorysize_3_32 \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
#done


#testsets=Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=I/A/V/R/K
#testsets=Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#!/bin/bash

#testsets=Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=I/A/V/R/K


#data_root='/data3/tuning/'
#testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#num_important_channel=500/700/900
#lambda_ape=0.3/0.7
#lr=0.0001
#epoch=20
#arch=RN50
##arch=ViT-B/16
#bs=1
#selection_p=1
#ctx_init=a_photo_of_a
#
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log res50_tf_various_channel_lambda_1_1 \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
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
#for nshot in 16 8
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape_textfull.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log res50_tf_various_channel_lambda_3_32_temp \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.0 --beta 5.5 --local_top 1 \
#  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
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
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log noft_beta55_memW02_3_32 \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.2 --beta 5.5
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
#for nshot in 0 1 2 4 8 16
#do
#  python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
#  -a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log noft_beta55_memW03_3_32 \
#  --gpu 7 --n_shot ${nshot} --n_augview 0  --static_mem_weight 0.3 --beta 5.5
#done

#data_root='/data3/tuning/'
##testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
#testsets=Food101
#arch=RN50
##arch=ViT-B/16
#bs=1
#selection_p=1
#ctx_init=a_photo_of_a
#
##python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
##-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
##--gpu 0 --n_shot 1 --epoch 20 --lr 1e-2 --wd 1e-3 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 0 --shared_param
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 7 --n_shot 0 --epoch 20 --lr 1e-3 --wd 1e-3 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 800 --lambda_ape 0.7 --shared_param


#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 0 --n_shot 1 --epoch 20 --lr 1e-4 --wd 1e-3 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 0 --shared_param
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 0 --n_shot 1 --epoch 20 --lr 1e-5 --wd 1e-3 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 0 --shared_param
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 0 --n_shot 1 --epoch 20 --lr 1e-4 --wd 1e-1 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 0 --shared_param
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 0 --n_shot 1 --epoch 20 --lr 1e-4 --wd 1e-2 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 0 --shared_param
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 0 --n_shot 1 --epoch 20 --lr 1e-4 --wd 1e-3 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 0 --shared_param
#
#python ./fixclip_memory_bank_ft_fewshot_ape.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p} --beta 5.5 \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl \
#--gpu 0 --n_shot 1 --epoch 20 --lr 1e-4 --wd 1e-4 --optimizer adamw --ft --mapping bias --n_augview 16 --num_important_channel 0 --shared_param







#python ./tpt_classification_memory_bank_tfree_localc.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 7 --weight_maxent 1.0 --memory_size 50


## 观察大一些的 memory filter score 对结果的影响？ 不是把所有样本都放进memory 中

#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 7 --weight_maxent 0.1
#
#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --gpu 7 --weight_maxent 0.01 \
#--tpt --ctx_init ${ctx_init}
#
#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --gpu 7 --weight_maxent 0.001 \
#--tpt --ctx_init ${ctx_init}
#
#python ./tpt_classification_max_average_entropy.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --gpu 7 --weight_maxent 0.0001 \
#--tpt --ctx_init ${ctx_init}

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