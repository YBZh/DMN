######### verified the effectiveness of the 7 handcrafted prompt;
######### the memorized image prediction, do not work here.

import argparse
import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import torch.nn as nn
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.fix_clip import get_fixed_clip

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, WeakStrongAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import ipdb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] ## pick the min entropy
    idx_confused = torch.argsort(batch_entropy, descending=False)[int(batch_entropy.size()[0] * top):] ## pick the max entropy
    return logits[idx], idx, logits[idx_confused], idx_confused

def avg_entropy(outputs):
    ## N*Class
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def sharpen(p, T=0.5):
    pt = p ** (1 / T)
    targets_u = pt / pt.sum()
    targets_u = targets_u.detach()  ## pseudo labels
    return targets_u


def align_entropy_confused(outputs, confused_output, strong_output, target, args, model, selected_idx, average_predcition):
    ## relation_matrix: nclass*nclass, max100 logit.
    all_loss = 0

    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0, keepdim=True) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]; log(mean_prob)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    confidence_entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    all_loss = all_loss + confidence_entropy

    # ipdb.set_trace()
    prediction_softmax = torch.exp(avg_logits)
    num_cla = average_predcition.shape[0]
    alpha = 1. / num_cla
    new_average_predcition = prediction_softmax * alpha + average_predcition.detach() * (1-alpha)
    max_entropy_loss = (new_average_predcition * torch.log(new_average_predcition + min_real)).sum(dim=-1)

    all_loss = all_loss + max_entropy_loss * 0.01


    return all_loss, new_average_predcition

class CLIPTTA(nn.Module):
    def __init__(self, beta=5.5):
        super(CLIPTTA, self).__init__()
        self.beta = beta

    def update_memory_bank(self, model, outputs, target):
        # outputs: batch * class.
        prob = outputs.softmax(dim=1)
        mean_prob = prob.mean(0, keepdim=True)
        value, indice = mean_prob.max(1)
        pseudo_label = indice.item()
        # print(value, indice, target)
        memory_write = 'local_global'  ## global_feat | local_att | local_global

        if memory_write == 'global_feat':
            if value.item() > 0.0:
                image_features = model.image_features_global
                selected_image_features = image_features[0].unsqueeze(0)  ## N*D -> 1*D  ##### 因为正式测试用的这种crop, 所以feature memory 里也存这样的。
                selected_image_features = selected_image_features / selected_image_features.norm(dim=-1, keepdim=True)
                # print(model.image_feature_count[:,0])  ## 全都分到了某些特定类别，类别很不均衡；加一个约束，强制要求类别均衡一些。
                if model.image_feature_count[pseudo_label] == model.memory_size:
                    ###### find the sample with the max entropy, and replace it with the new one; if the new one is low entropy.
                    current_instance_entropy = avg_entropy(outputs)
                    # ipdb.set_trace()
                    if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                        pass ## the current entropy is very large.
                    else:
                        _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                        to_replace_indice = indice[-1] ## with max entropy, ascending.
                        model.image_feature_memory[pseudo_label][to_replace_indice] = selected_image_features
                        model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                        model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
                else:
                    model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = selected_image_features
                    model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]
                    model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = -(mean_prob * torch.log(mean_prob + 1e-10)).sum(1)
                    model.image_feature_count[pseudo_label] += 1

        elif memory_write == 'local_att':
            ### get the attented feature using text feature as Q, and spatial image feat as K, and spatial image feat as V.
            if value.item() > 0.0:
                image_features = model.image_features_vanilla
                text_features = model.text_features[pseudo_label] ## 512
                selected_image_features = image_features[0].unsqueeze(0)  ## N*L*D -> 1*L*D  ##### 因为正式测试用的这种crop, 所以feature memory 里也存这样的。
                selected_image_features = selected_image_features / selected_image_features.norm(dim=-1, keepdim=True) ## 1*197*512
                ##### calculate the similarity between target text feat and selected image features, get the attented feat.
                cos_sim = (selected_image_features * text_features).sum(-1)  ## between 0.2-0.3, very close.
                ipdb.set_trace()
                weight_prob = (cos_sim * 100).softmax(-1) ## 1*197  ## max about 6% - 60%.
                attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features).sum(1)[0] ## 512
                attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 512

                if model.image_feature_count[pseudo_label] == model.memory_size:
                    ###### find the sample with the max entropy, and replace it with the new one; if the new one is low entropy.
                    current_instance_entropy = avg_entropy(outputs)
                    # ipdb.set_trace()
                    if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                        pass ## the current entropy is very large.
                    else:
                        _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                        to_replace_indice = indice[-1] ## with max entropy, ascending.
                        model.image_feature_memory[pseudo_label][to_replace_indice] = attented_feat
                        model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                        model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
                else:
                    model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = attented_feat
                    model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]
                    model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = -(mean_prob * torch.log(mean_prob + 1e-10)).sum(1)
                    model.image_feature_count[pseudo_label] += 1

        elif memory_write == 'local_global':
            ###
            ### get the attented feature using text feature as Q, and spatial image feat as K, and spatial image feat as V.
            if value.item() > 0.0:
                text_features = model.text_feat[pseudo_label]  ## 512
                selected_image_features_global = model.image_features_global[:1]
                selected_image_features_local = model.image_features_local[:1]
                # selected_image_features = image_features[0].unsqueeze(0)  ## N*L*D -> 1*L*D  ##### 因为正式测试用的这种crop, 所以feature memory 里也存这样的。
                # selected_image_features = selected_image_features / selected_image_features.norm(dim=-1, keepdim=True)  ## 1*197*512
                # selected_image_features_global = selected_image_features[0,0]
                # selected_image_features_local = selected_image_features[:,1:]
                ##### calculate the similarity between target text feat and selected image features, get the attented feat.
                cos_sim = (selected_image_features_local * text_features).sum(-1)  ## between 0.2-0.3, very close.
                # ipdb.set_trace()
                weight_prob = (cos_sim * 100).softmax(-1)  ## 1*197  ## max about 6% - 60%.
                attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features_local).sum(1)[0]  ## 512

                attented_feat = attented_feat  ## merge local & global features.
                attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 512

                if model.image_feature_count[pseudo_label] == model.memory_size:
                    ###### find the sample with the max entropy, and replace it with the new one; if the new one is low entropy.
                    current_instance_entropy = avg_entropy(outputs)
                    # ipdb.set_trace()
                    if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                        pass  ## the current entropy is very large.
                    else:
                        _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        model.image_feature_memory[pseudo_label][to_replace_indice] = selected_image_features_global
                        model.local_feature_memory[pseudo_label][to_replace_indice] = attented_feat
                        model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                        model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
                else:
                    model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = selected_image_features_global
                    model.local_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = attented_feat

                    model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]
                    model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = -(
                                mean_prob * torch.log(mean_prob + 1e-10)).sum(1)
                    model.image_feature_count[pseudo_label] += 1
        else:
            raise NotImplementedError

    def get_image_pred(self, model):
        img_feat = model.image_features_global[:1]  # 1*1024
        count_image_feat = model.image_feature_count.clone()
        num_class = model.image_feature_memory.shape[0]
        image_classifier = 'similarity_weighted'  ## category_center | weighted_center | similarity_weighted

        ############ use the mix of new text feature and image feature as image feature.
        ## the online updated text feature.
        # merged_image_feat = torch.cat((model.image_feature_memory, model.text_features.unsqueeze(1)), dim=1) ## 200*11*1024
        ## fixed init text features.
        merged_image_feat = torch.cat((model.image_feature_memory, model.text_feat.unsqueeze(1)), dim=1)  ## 200*11*1024
        if image_classifier == 'weighted_center':
            ############## weighted combine the memorized feature as the final classifier.
            merged_entropy = torch.cat((model.image_entropy_mem,  torch.zeros(num_class,1).to(merged_image_feat.device)), dim=1) ## 200*11
            filled_image_feat = (merged_image_feat * (- merged_entropy - math.log(1./ num_class)).unsqueeze(-1)).sum(1)  ## 根据entropy 进行加权
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        elif image_classifier == 'category_center':
            ############### assign each feature with equal weights.
            filled_image_feat = merged_image_feat.sum(1) / (count_image_feat + 1)  ### no zero. 200*1024
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        elif image_classifier == 'similarity_weighted':  ## this is an instance adaptative method.
            ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized probability.
            # merged_image_prediction = torch.cat((model.image_prediction_mem, torch.eye(num_class).unsqueeze(1).to(merged_image_feat.device)), dim=1)
            ##  200*11*200；
            # ipdb.set_trace()
            similarity_matrix = (img_feat * merged_image_feat).sum(-1) ## 200*11  理应-1~1, 但是实际都是 0.1~0.2， 可能和训练方式有关
            ####### 这里要修改吗？ 这里实际上还是类似对所有的样本做加权组合？ 只是权重是adaptative 的； 但是目前的权重基本都是相同的差异性没有拉开
            ########### image-image 的相似度是比较高的，0.5-0.6 左右，所以image 贡献了比较多的权重。
            ## 但是最开始image 可能是不准确的； 所以最好开始给 text 较大的权重:发现结果正好相反，开始要给text 比较小的权重，否则结果很差
            # similarity_matrix[:,-1] = 0.01 ## put more weight to the text features.
            # ipdb.set_trace()  ##
            filled_image_feat = (merged_image_feat * similarity_matrix.unsqueeze(-1)).sum(1) ## 根据同image feat similarity 进行加权, 得到new classifier.
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        else:
            raise NotImplementedError

    ######## 基于pseudo label, 给一个判断，然后
    def get_image_pred_local(self, model):
        img_feat = model.image_features_local[:1].mean(1)  # 1*1024
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        merged_image_feat = torch.cat((model.local_feature_memory, model.text_feat.unsqueeze(1)), dim=1)  ## 200*11*1024

        similarity_matrix = (img_feat * merged_image_feat).sum(-1)  ## 200*11  理应-1~1, 但是实际都是 0.1~0.2， 可能和训练方式有关
        filled_image_feat = (merged_image_feat * similarity_matrix.unsqueeze(-1)).sum(1)  ## 根据同image feat similarity 进行加权, 得到new classifier.
        filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * img_feat @ filled_image_feat.t()
        return logits.softmax(dim=1)

    # ########################## 今天做好这个就算完工。测试是否
    # def get_image_pred_local(self, model):
    #     ###################3 为每一类学一个 累呗独有的 image feature 太耗显存了，要为所有类学一个！
    #     #### 存在memory 里的是根据和该类的相似度算出来的，这里最简单的做法是直接求average 作为local feature.
    #     ## fixed init text features.
    #     merged_image_feat = torch.cat((model.local_feature_memory, model.text_feat.unsqueeze(1)), dim=1)  ## 200*51*1024
    #     ####### get the attented feature for each category!!
    #     # image_features = model.image_features_vanilla ## 1*(L+1)*D
    #     selected_image_features_local = model.image_features_local[:1] ## 1*49*512
    #     text_features = model.text_feat  ## 200*512
    #     ##### calculate the similarity between target text feat and selected image features, get the attented feat.
    #     cos_sim = (selected_image_features_local @ text_features.T)[0].T  ## 200*49,  categories * local regions
    #     # !!!!!!!!!!!!!!!!!!!!!!!!!!! 这里的sharp 程度是一个可以调整的超参！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     ####### 但是我没有调，后面再说。
    #     # weight_prob = (cos_sim * 100).softmax(-1)  ## 200*49, 这49个local region 和每一类的相似度； 这里没有直接用cos distance, 而是softmax sharp 地选出一个
    #     # weight_prob = cos_sim
    #     weight_prob = torch.exp(-self.beta * (-cos_sim + 1))
    #     attented_feat = (weight_prob @ selected_image_features_local[0]) ## 200*512, 每个类独有的image feature.
    #     attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 200*512, 每个类独有的image feature.
    #     # merged_image_feat: 200*51*512, category * memory length * D
    #     similarity_matrix = (merged_image_feat @ attented_feat.T)   ## 200*51*200  ## cos similarity.
    #     ### 200*51*512*1 * 200*51*1*200  --> 200*512*200; 第一个200 是200个类的memory indice; 第二个200 是200个类，每类独有image feature.
    #     filled_image_feat = (merged_image_feat.unsqueeze(-1) * similarity_matrix.unsqueeze(2)).sum(1)  ## 根据同image feat similarity 进行加权, 得到new classifier.
    #
    #     filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=1, keepdim=True)  ## 200*512*200
    #     filled_image_feat = filled_image_feat.permute(2,1,0) ## 倒过来，第一个200是200个独有的image feature indice, 后一个200 是200 个memory indice； 这样第二个200 其实就是分类器对应的200类了
    #     logit_scale = model.logit_scale.exp()
    #     ## 200*512  200*512*200 --> 200*200,
    #     logits = logit_scale * (attented_feat.unsqueeze(-1) * filled_image_feat).sum(1)
    #     prob = logits.softmax(dim=1) ### 200*200; 加在类别维度的softmax.
    #     ## 找出熵最小的那一类的indice & 且prediction 类别和category 类别一致， 先找出pred 和 indice 一致的。
    #     value, indice = prob.max(1)  ##
    #     given_indice = torch.arange(200).to(indice.device)
    #     selected_prob = prob[indice == given_indice]
    #     if selected_prob.shape[0] == 0:
    #         return torch.zeros(1,prob.shape[1]).to(prob.device)
    #     else:
    #         selected_indice = entropy(selected_prob).min(0)[1]
    #         # selected_indice_temp = prob.max(1)[0].max(0)[1]  ## 应该取entropy 最小的prediction? 改一下这里。
    #         # print(selected_indice == selected_indice_temp)  ## most same, also some different.
    #         return selected_prob[selected_indice.item()].unsqueeze(0)

    def forward(self, outputs, confused_output, strong_output, target, args, model, selected_idx, update_m=True):
        ## relation_matrix: nclass*nclass, max100 logit.
        all_loss = 0
        temperature = 4.0
        # self.average_predcition =  self.average_predcition.to(outputs.device)

        ########## instance wise entropy minimization.
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0, keepdim=True) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]; log(mean_prob)
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        confidence_entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
        all_loss = all_loss + confidence_entropy

        return all_loss



def main():
    args = parser.parse_args()
    # args.log = args.log + '_' + args.arch
    set_random_seed(args.seed)
    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    model = get_fixed_clip(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init, memory_size=args.memory_size, text_prompt=args.text_prompt)
    model_state = None

    for name, param in model.named_parameters():
        param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                             augmix=len(set_id) > 1)
        # augmix=len(set_id)>1) ### aug mix not used for ImageNet test set.
        batchsize = 1

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,  ## the input has been shuffled.
                    num_workers=args.workers, pin_memory=True)

        results[set_id] = direct_inference(val_loader, model, args)
        # results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

        log = open(os.path.join(args.log + '.txt'), 'a')
        state = {k: v for k, v in args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.write('                        Target_T1 acc: %3f, local acc: %3f, global acc: : %3f' % (results[set_id][0], results[set_id][1], results[set_id][2]))
        log.close()
        length = len(results[set_id])

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-1 local acc, \t\t Top-1 global acc \t\t Searched acc \t\t beta \t\t gama.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print('mean', end="	")
    print("\n")
    for i in range(length):
        cul_acc = 0
        cul_count = 0
        for id in results.keys():
            print("{:.2f}".format(results[id][i]), end="	")
            cul_acc += float(results[id][i])
            cul_count += 1
        print("{:.2f}".format(cul_acc), end="	")
        print("\n")


def entropy(outputs):
    # prob: 1*200, logit.
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0, keepdim=True) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]; log(mean_prob)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    confidence_entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    return confidence_entropy

def direct_inference(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_local = AverageMeter('Acc@local', ':6.2f', Summary.AVERAGE)
    top1_global = AverageMeter('AccImg@1', ':6.2f', Summary.AVERAGE)
    top1_local_vanilla = AverageMeter('AccLV@1', ':6.2f', Summary.AVERAGE)
    top1_combine = AverageMeter('AccCom@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top1_local, top1_global, top1_local_vanilla, top1_combine],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    with torch.no_grad():
        text_feat, text_feat_full = model.get_text_features()
    ## text_feat: 200*1024
    ## text_feat_full:  200 * 7 * 1024

    pred_vanilla = []
    pred_global = []
    pred_local = []
    labels = []
    cliptta = CLIPTTA(beta=args.beta)

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        else: ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            image_features_global, image_features_local =  model.get_image_features(images)

        # image_features_global: torch.Size([128, 1024])
        # image_features_local: torch.Size([128, 49, 1024])
        logit_scale = model.logit_scale.exp()
        img_text_logit = logit_scale * image_features_global @ text_feat.t() ## 128*200
        img_text = img_text_logit.softmax(-1)
        img_text_pred = img_text[:1]
        weak_output, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text_logit,
                                                                                                 args.selection_p)
        # ipdb.set_trace()
        cliptta.update_memory_bank(model, weak_output, target)
        img_global_pred = cliptta.get_image_pred(model)
        img_local_pred = cliptta.get_image_pred_local(model)

        local_vanilla_pred = img_text_pred + img_local_pred
        comb_prob = img_text_pred + img_local_pred + img_global_pred

        pred_vanilla.append(img_text_pred)
        pred_global.append(img_local_pred)
        pred_local.append(img_global_pred)
        labels.append(target)

        ############# ############# ############# ############# ############# ############# weighted averaged text features.
        # img2textfull_sim = (image_features_global.unsqueeze(1).unsqueeze(1) * text_feat_full.unsqueeze(0)).sum(-1)
        # ## if multiply zero, then it return to average pooling. verified!
        # ## 128*200*7
        # img2textfull_weight = img2textfull_sim
        # weighted_text_feat = (img2textfull_weight.unsqueeze(-1) * text_feat_full.unsqueeze(0)).sum(2)  ## 128*200*1024
        # weighted_text_feat = weighted_text_feat / weighted_text_feat.norm(dim=-1, keepdim=True)
        # img_text_att = logit_scale * (image_features_global.unsqueeze(1) * weighted_text_feat).sum(-1) ## 128*200
        # img_text_att = img_text_att.softmax(-1)
        ############# ############# ############# ############# ############# #############  not work

        # # measure accuracy and record loss
        acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
        acc1_global, _ = accuracy(img_global_pred, target, topk=(1, 5))
        acc1_local, _ = accuracy(img_local_pred, target, topk=(1, 5))
        acc1_local_vanilla, _ = accuracy(local_vanilla_pred, target, topk=(1, 5))
        acc1_comb, _ = accuracy(comb_prob, target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top1_global.update(acc1_global[0], image.size(0))
        top1_local.update(acc1_local[0], image.size(0))
        top1_local_vanilla.update(acc1_local_vanilla[0], image.size(0))
        top1_combine.update(acc1_comb[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()
    pred_vanilla = torch.cat(pred_vanilla, dim=0)
    pred_global = torch.cat(pred_global, dim=0)
    pred_local = torch.cat(pred_local, dim=0)
    labels = torch.cat(labels, dim=0)
    ########## put the hyper parameter search here.
    ## final prediction = image_text_pred + alpha * global + beta * local
    weight_search = True
    if weight_search:
        beta2_list = [i * (10 - 0.001) / 200 + 0.001 for i in range(200)] ## 0.001 - 10
        beta3_list = [i * (10 - 0.001) / 200 + 0.001 for i in range(200)] ## 0.001 - 10
        print('-' * 20)
        print('Starting searching...')
        print('     beta1 = 1.0')
        print('     beta2 searching range: [0.001, ' + str(10) + ']')
        print('     beta3 searching range: [0.001, ' + str(10) + ']')
        print('-' * 20)

        best_acc = 0.
        best_beta2 = 0.
        best_beta3 = 0.

        for beta2 in beta2_list:
            for beta3 in beta3_list:
                logits = pred_vanilla + pred_local * beta2 + pred_global * beta3
                acc, _ = accuracy(logits, labels, topk=(1, 5))
                acc = acc.item()
                if acc > best_acc:
                    print('New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; Acc: {:.2f}'.format(1, beta2,beta3, acc))
                    best_acc = acc
                    best_beta2 = beta2
                    best_beta3 = beta3
        print()
        print(f"Searched Acc: {best_acc:.2f} with local weight {best_beta2:.2f} and global weight {best_beta3:.2f}")

    return [top1.avg, top1_local.avg, top1_global.avg, best_acc, best_beta2, best_beta3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--note', type=str, default='default', help='some places to write note')
    parser.add_argument('--loss_type', type=str, default='default', help='some places to write note')
    parser.add_argument('--log', type=str, default='loga', help='some places to write note')
    parser.add_argument('--weight_maxent',  default=1e-3, type=float, help='loss weight')
    parser.add_argument('--beta',  default=5.5, type=float, help='loss weight')
    parser.add_argument('--memory_size', type=int, default=50)
    parser.add_argument('--text_prompt', type=str, default='tip', help='simple | tip | full')


    main()