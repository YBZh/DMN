#!/bin/bash

#data_root='/path/to/your/data/root'
#coop_weight='/path/to/pretrained/coop/weight.pth'
#testsets=$1
#arch=RN50
## arch=ViT-B/16
#bs=64
#
#python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
#-a ${arch} -b ${bs} --gpu 0 \
#--tpt --load ${coop_weight}


data_root='/data3/tuning/'
coop_weight='/data3/tuning/to_gdrive/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
testsets=I
arch=RN50
#arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --tpt --ctx_init ${ctx_init} --gpu 4 --load ${coop_weight}