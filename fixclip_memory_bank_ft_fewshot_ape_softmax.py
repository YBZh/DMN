######### 把attention 的非线性都改成softmax, 然后重新调softmax 的权重。

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
from data.datautils import AugMixAugmenter, build_dataset, AugMemAugmenter, StrongAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import ipdb

from typing import Callable


def print_logger(
        old_print: Callable,
        file_name: str,
) -> Callable:
    """Returns a function which calls `old_print` twice, specifying a `file=` on the second call.

    Arguments:
        old_print: The `print` function to call twice.
        file_name: The name to give the log file.
    """

    def log_print(*args, **kwargs):
        old_print(*args, **kwargs)
        with open(file_name, "a") as log_file:
            old_print(*args, file=log_file, **kwargs)

    return log_print

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        if len(labels.shape) == 1:
            num_classes = logits.shape[-1]
            alpha_div_k = self.alpha / num_classes
            target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
                (1. - self.alpha) + alpha_div_k
            loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            return loss.mean()
        else:
            loss = -(labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            return loss.mean()


def important_channel_indice(args, model, only_use_txt=True):
    if only_use_txt or args.shot  == 0:
        feats = model.text_feat.unsqueeze(1)  ## C * 1 * D
    else:
        feats = model.fixed_global_feat_vanilla ## C * L * D, including text feat & few shot image feat.
    cate_num, samp_num, feat_dim = feats.shape

    sim_sum = torch.zeros((feat_dim)).to(feats.device)
    count = 0
    # ipdb.set_trace()
    for i in range(cate_num):
        for j in range(cate_num):
            for m in range(samp_num):
                for n in range(samp_num):
                    if i != j:
                        sim_sum += feats[i, m, :] * feats[j, n, :]
                        count += 1
    sim = sim_sum / count
    # ipdb.set_trace()
    criterion = (-1) * args.lambda_ape * sim + (1-args.lambda_ape) * torch.var(model.text_feat, dim=0)
    _, indices = torch.topk(criterion, k=args.num_important_channel)
    # model.text_feat_indiced = model.text_feat[:, indices]
    # model.fixed_global_feat_indiced = model.fixed_global_feat[:, :, indices]
    # model.fixed_local_feat_indiced = model.fixed_local_feat[:, :, indices]
    # model.image_feature_memory_indiced = model.image_feature_memory[:, :, indices]
    # model.local_feature_memory_indiced = model.local_feature_memory[:, :, indices]
    # model.fixed_global_feat_vanilla_indiced = model.fixed_global_feat_vanilla[:, :, indices]
    # model.fixed_local_feat_vanilla_indiced = model.fixed_local_feat_vanilla[:, :, indices]
    return indices


def select_confident_samples(prob, top):
    # ipdb.set_trace()
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] ## pick the min entropy
    idx_confused = torch.argsort(batch_entropy, descending=False)[int(batch_entropy.size()[0] * top):] ## pick the max entropy
    return prob[idx], idx, prob[idx_confused], idx_confused

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
    def __init__(self, args=None, beta=5.5, feat_dim=1024, class_num=1000, mapping='bias'):
        super(CLIPTTA, self).__init__()
        self.args =  args
        self.indice = args.indice  ## indice of important channels.
        self.beta = beta
        self.rank = 4
        self.init_pred = 0
        # if args.num_important_channel != 0:
        #     feat_dim = args.num_important_channel
        if args.shared_param:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.local_affine = self.global_affine
            self.local_bias = self.global_bias
            self.local_ffn_affine = self.global_ffn_affine
            self.local_ffn_bias = self.global_ffn_bias
            self.text_affine = self.global_ffn_affine
            self.text_bias = self.global_ffn_bias

        else:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.local_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.local_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.local_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.local_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.

            self.text_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.text_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.learnable_mapping = mapping ### bias | affine | all





    def update_memory_bank(self, model, target):
        # outputs: batch * class.
        # prob = outputs.softmax(dim=1)
        # mean_prob = prob.mean(0, keepdim=True)
        mean_prob = self.init_pred[0]
        value, indice = mean_prob.max(0)
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
                # ipdb.set_trace()
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
                #################### 还有这里，local image feat 的加权方式也要商量，直接*100 softmax 应该也不是最优的。
                weight_prob = (cos_sim * self.args.softmax_local).softmax(-1)  ## 1*197  ## max about 6% - 60%.
                attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features_local).sum(1)[0]  ## 512

                attented_feat = attented_feat  ## merge local & global features.
                attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 512
                current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()
                if model.image_feature_count[pseudo_label] == model.memory_size:
                    ###### find the sample with the max entropy, and replace it with the new one; if the new one is low entropy.
                    # current_instance_entropy = avg_entropy(outputs)
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
                    model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = current_instance_entropy
                    model.image_feature_count[pseudo_label] += 1
        else:
            raise NotImplementedError

    def get_image_pred(self, model, return_full=False, return_logit=False):
        img_feat = model.image_features_global[:1]  # 1*1024
        count_image_feat = model.image_feature_count.clone()
        num_class = model.image_feature_memory.shape[0]
        image_classifier = 'similarity_weighted'  ## category_center | weighted_center | similarity_weighted

        ############ use the mix of new text feature and image feature as image feature.
        ## the online updated text feature.
        # merged_image_feat = torch.cat((model.image_feature_memory, model.text_features.unsqueeze(1)), dim=1) ## 200*11*1024
        ## fixed init text features.
        memorized_image_feat = torch.cat((model.image_feature_memory, model.fixed_global_feat_vanilla), dim=1)  ## 200*11*1024
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
            filled_image_feat = memorized_image_feat.sum(1) / (count_image_feat + 1)  ### no zero. 200*1024
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        elif image_classifier == 'similarity_weighted':  ## this is an instance adaptative method.
            ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized probability.
            # merged_image_prediction = torch.cat((model.image_prediction_mem, torch.eye(num_class).unsqueeze(1).to(merged_image_feat.device)), dim=1)
            ##  200*11*200；
            # ipdb.set_trace()
            ###################### 有一些memory 是空的，现在却往里面塞了一个self.global_bias， 这不合理，还要把它继续置空。
            with torch.no_grad():
                if self.learnable_mapping == 'bias':
                    memorized_image_feat_K = memorized_image_feat  + self.global_bias.unsqueeze(1)  ## class*shot*1024
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
                elif self.learnable_mapping == 'affine':
                    memorized_image_feat_K = memorized_image_feat + memorized_image_feat @ self.global_affine  ## class*shot*1024
                    img_feat_mappling = img_feat + img_feat @ self.global_affine  ## N*1024
                elif self.learnable_mapping == 'all':
                    memorized_image_feat_K = memorized_image_feat + memorized_image_feat @ self.global_affine + self.global_bias.unsqueeze(1)  ## class*shot*1024
                    img_feat_mappling = img_feat + img_feat @ self.global_affine + self.global_bias.mean(0, keepdim=True)  ## N*1024
                else:
                    raise NotImplementedError
                memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
                memorized_image_feat_K[memorized_image_feat.sum(-1) == 0] = 0
                img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)

            similarity_matrix = (img_feat_mappling * memorized_image_feat_K).sum(-1) ## 200*11  理应-1~1, 但是实际都是 0.1~0.2， 可能和训练方式有关
            # similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
            similarity_matrix = (similarity_matrix * self.args.softmax_read).softmax(-1)
            ####### 这里要修改吗？ 这里实际上还是类似对所有的样本做加权组合？ 只是权重是adaptative 的； 但是目前的权重基本都是相同的差异性没有拉开
            ########### image-image 的相似度是比较高的，0.5-0.6 左右，所以image 贡献了比较多的权重。
            ## 但是最开始image 可能是不准确的； 所以最好开始给 text 较大的权重:发现结果正好相反，开始要给text 比较小的权重，否则结果很差
            ######## 对similarity matrix 的shape 进行重新refine
            ## 根据同image feat similarity 进行加权, 得到new classifier.
            adaptive_image_feat = (memorized_image_feat_K * similarity_matrix.unsqueeze(-1)).sum(1)
            ## torch.Size([1, class, dim])
            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            if self.learnable_mapping == 'bias':
                adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024
            elif self.learnable_mapping == 'affine':
                adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.global_ffn_affine
            elif self.learnable_mapping == 'all':
                adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.global_ffn_affine + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024
            else:
                raise NotImplementedError
            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            # ipdb.set_trace()
            # logits = logit_scale * img_feat @ adaptive_image_feat.transpose(1,2)
            logits = logit_scale * adaptive_image_feat @ img_feat.unsqueeze(-1)  ## used feat is not update.
            logits = logits[:,:,0]
            del memorized_image_feat_K, img_feat_mappling, similarity_matrix, adaptive_image_feat
            return logits.softmax(dim=1)
        else:
            raise NotImplementedError

    ######## 基于pseudo label, 算top10 的相似度得到10 个特征，然后算10 个prediction.
    def get_image_pred_local(self, model, return_full=False, return_logit=False):
        init_pred = self.init_pred[0]   ## class_num, 从里面取出top 5, 来算attention
        value, indice = init_pred.sort(descending=True) ## from large to small.
        cancidate = indice[:1]
        # cancidate = indice[:, :10]
        # if return_full:
        #     img_feat = model.image_features_local
        # else:
        #     cancidate = cancidate[:1]
        #     img_feat = model.image_features_local[:1]  # 1*L*1024
        img_feat = model.image_features_local[:1]  # 1*L*1024
        text_features_cancidate = model.text_feat[cancidate]  ## can*1024
        # ipdb.set_trace()
        cos_sim = img_feat @ text_features_cancidate.T  ## 1*L*can
        weight_prob = (cos_sim * self.args.softmax_local).softmax(1)  ## L softmax.,  1*L*can
        attented_feat = (weight_prob.transpose(1, 2) @ img_feat)  ## 1*can*1024
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 和memorized local image feature 算prediction.
        memorized_image_feat = torch.cat((model.local_feature_memory, model.fixed_local_feat_vanilla), dim=1)  ## 200*11*1024
        if self.learnable_mapping == 'bias':
            memorized_image_feat_KV = memorized_image_feat + self.local_bias.unsqueeze(1)  ## class*shot*1024
            attented_feat = attented_feat + self.local_bias.mean(0, keepdim=True)  ## N*1024
        elif self.learnable_mapping == 'affine':
            memorized_image_feat_KV = memorized_image_feat + memorized_image_feat @ self.local_affine
            attented_feat = attented_feat + attented_feat @ self.local_affine
        elif self.learnable_mapping == 'all':
            memorized_image_feat_KV = memorized_image_feat + memorized_image_feat @ self.local_affine + self.local_bias.unsqueeze(1)  ## class*shot*1024
            attented_feat = attented_feat + attented_feat @ self.local_affine + self.local_bias.mean(0, keepdim=True)  ## N*1024
        else:
            raise NotImplementedError
        memorized_image_feat_KV[memorized_image_feat.sum(-1) == 0] = 0

        similarity_matrix = attented_feat.unsqueeze(1) @ memorized_image_feat_KV.transpose(1,2)  ## 1*200*10*11 (view*class*can*memory)
        # similarity_matrix: N*n_cls*can*shot
        ## fewshot_label:   class*shot*class   期望输出：N*can*class
        # similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))  ## make it more sharp.
        similarity_matrix = (similarity_matrix * self.args.softmax_read).softmax(-1)
        adaptive_image_feat = similarity_matrix @ memorized_image_feat_KV  ## batch*200*10*1024
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True) ## 200*10*1024
        if self.learnable_mapping == 'bias':
            adaptive_image_feat = adaptive_image_feat + self.local_ffn_bias.unsqueeze(1).unsqueeze(0)  ## class*shot*1024
        elif self.learnable_mapping == 'affine':
            adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.local_ffn_affine
        elif self.learnable_mapping == 'all':
            adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.local_ffn_affine + self.local_ffn_bias.unsqueeze(1).unsqueeze(0)  ## class*shot*1024
        else:
            raise NotImplementedError
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)  ## batch*200*10*1024

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (attented_feat * adaptive_image_feat).sum(-1)  ## 1*200*10

        logits = logits.transpose(1,2)

        pred_prob = logits.softmax(2) ## 10*200
        entropy = -(pred_prob * torch.log(pred_prob + 1e-8)).sum(-1)   ## 10个里挑一个。
        _, indice_min = entropy.min(1)  ### 这里要判断，这个特征是否是
        return  pred_prob[0, indice_min]  ## 1*102


    def get_image_pred_fewshot_local(self, model, return_full=False, return_logit=False):
        # image_features_local: N*L*Channel
        init_pred = self.init_pred   ## class_num, 从里面取出top 5, 来算attention
        value, indice = init_pred.sort(descending=True) ## from large to small.
        cancidate = indice[:, :1]
        if return_full:
            img_feat = model.image_features_local[:, :, self.args.indice] ## extract the importance feat.
        else:
            cancidate = cancidate[:1]
            img_feat = model.image_features_local[:1, :, self.args.indice]  # 1*L*1024

        text_features_cancidate = model.text_feat[cancidate][:,:,self.args.indice]  ## batch* can*1024
        cos_sim = img_feat @ text_features_cancidate.transpose(1, 2)  ## batch*L*can
        #################### ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        weight_prob = (cos_sim * self.args.softmax_local).softmax(1)  ## L softmax.,  batch*L*can, 分类用的softmax, 可能需要调参。

        attented_feat = (weight_prob.transpose(1, 2) @ img_feat)  ## batch*can*1024
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 和memorized local image feature 算prediction.
        memorized_image_feat = model.fixed_local_feat[:,:, self.args.indice]  ## 200*11*1024, only the fixed local feat
        if self.learnable_mapping == 'bias':
            memorized_image_feat = memorized_image_feat + self.local_bias[:,self.args.indice].unsqueeze(1)  ## class*shot*1024
            attented_feat = attented_feat + self.local_bias[:, self.args.indice].mean(0, keepdim=True)  ## N*1024
        elif self.learnable_mapping == 'affine':
            memorized_image_feat = memorized_image_feat + memorized_image_feat @ self.local_affine[self.args.indice,self.args.indice]
            attented_feat = attented_feat + attented_feat @ self.local_affine[self.args.indice,self.args.indice]
        elif self.learnable_mapping == 'all':
            memorized_image_feat = memorized_image_feat + memorized_image_feat @ self.local_affine[self.args.indice,self.args.indice] + self.local_bias[:,self.args.indice].unsqueeze(1)  ## class*shot*1024
            attented_feat = attented_feat + attented_feat @ self.local_affine[self.args.indice,self.args.indice] + self.local_bias[:,self.args.indice].mean(0, keepdim=True)  ## N*1024
        else:
            raise NotImplementedError

        similarity_matrix = attented_feat.unsqueeze(1) @ memorized_image_feat.transpose(1,2)  ## batch*class num*can*shot (view*class*can*memory)
        # similarity_matrix: N*n_cls*can*shot
        ## fewshot_label:   class*shot*class   期望输出：N*can*class
        # similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))  ## make it more sharp.
        similarity_matrix = (similarity_matrix * self.args.softmax_read).softmax(-1)
        adaptive_image_feat = similarity_matrix @ memorized_image_feat  ## batch*200*10*1024
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True) ## batch*200*10*1024
        if self.learnable_mapping == 'bias':
            adaptive_image_feat = adaptive_image_feat + self.local_ffn_bias[:,self.args.indice].unsqueeze(1).unsqueeze(0)  ## class*shot*1024
        elif self.learnable_mapping == 'affine':
            adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.local_ffn_affine[self.args.indice,self.args.indice]
        elif self.learnable_mapping == 'all':
            adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.local_ffn_affine[self.args.indice,self.args.indice] + self.local_ffn_bias[:,self.args.indice].unsqueeze(1).unsqueeze(0)  ## class*shot*1024
        else:
            raise NotImplementedError
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)  ## batch*200*10*1024

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (attented_feat.unsqueeze(1) * adaptive_image_feat).sum(-1)  ## batch*class num*candidate
        logits = logits.transpose(1,2)
        pred_prob = logits.softmax(2) ## batch*10*200
        entropy = -(pred_prob * torch.log(pred_prob + 1e-8)).sum(-1)   ## 10个里挑一个。
        _, indice_min = entropy.min(1)  ### 这里要判断，这个特征是否是, find the min entropy among cancidate.
        batch_indice = torch.arange(pred_prob.shape[0]).to(indice_min.device)
        if return_logit:
            return logits[batch_indice, indice_min, :]
        else:
            return pred_prob[batch_indice, indice_min,:]


    def get_image_pred_fewshot_global(self, model, return_full=False, return_logit=False):
        if return_full:
            img_feat = model.image_features_global[:, self.args.indice]  # 1*1024
        else:
            img_feat = model.image_features_global[:1, self.args.indice]  # 1*1024
        num_class = model.image_feature_memory.shape[0]

        ############ use the mix of new text feature and image feature as image feature.
        ## the online updated text feature.
        # memorized_image_feat = torch.cat((model.image_feature_memory, model.text_features.unsqueeze(1)), dim=1) ## 200*11*1024
        ## fixed init text features.
        memorized_image_feat = model.fixed_global_feat[:,:, self.args.indice]  ## 200*11*1024, few shot samples and text features.
        if self.learnable_mapping == 'bias':
            memorized_image_feat = memorized_image_feat + self.global_bias[:,self.args.indice].unsqueeze(1)  ## class*shot*1024
            img_feat_mappling = img_feat + self.global_bias[:,self.args.indice].mean(0, keepdim=True)  ## N*1024
        elif self.learnable_mapping == 'affine':
            memorized_image_feat = memorized_image_feat + memorized_image_feat @ self.global_affine[self.args.indice, self.args.indice]
            img_feat_mappling = img_feat + img_feat @ self.global_affine[self.args.indice, self.args.indice]
        elif self.learnable_mapping == 'all':
            memorized_image_feat = memorized_image_feat + memorized_image_feat @ self.global_affine[self.args.indice, self.args.indice] + self.global_bias[:,self.args.indice].unsqueeze(1)  ## class*shot*1024
            img_feat_mappling = img_feat + img_feat @ self.global_affine[self.args.indice, self.args.indice] + self.global_bias[:,self.args.indice].mean(0, keepdim=True)  ## N*1024
        else:
            raise NotImplementedError

        memorized_image_feat = memorized_image_feat / memorized_image_feat.norm(dim=-1, keepdim=True)
        img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)
        ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized probability.
        # merged_image_prediction = torch.cat((model.image_prediction_mem, torch.eye(num_class).unsqueeze(1).to(memorized_image_feat.device)), dim=1)
        ##  200*11*200；
        # similarity_matrix = (img_feat * memorized_image_feat_K).sum(-1) ## 200*11  理应-1~1, 但是实际都是 0.1~0.2， 可能和训练方式有关
        similarity_matrix = memorized_image_feat @ img_feat_mappling.T ## class*shot*Batch
        # similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
        similarity_matrix = (similarity_matrix * self.args.softmax_read).softmax(-1)
        ####### 这里要修改吗？ 这里实际上还是类似对所有的样本做加权组合？ 只是权重是adaptative 的； 但是目前的权重基本都是相同的差异性没有拉开
        ########### image-image 的相似度是比较高的，0.5-0.6 左右，所以image 贡献了比较多的权重。
        ## 但是最开始image 可能是不准确的； 所以最好开始给 text 较大的权重:发现结果正好相反，开始要给text 比较小的权重，否则结果很差
        ######## 对similarity matrix 的shape 进行重新refine
        # memorized_image_feat_V = memorized_image_feat + memorized_image_feat @ self.global_affine + self.global_bias.unsqueeze(1)
        # memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
        ## 根据同image feat similarity 进行加权, 得到new classifier.
        adaptive_image_feat = memorized_image_feat.transpose(1,2) @ similarity_matrix ## class * D * batch, 102*1024*204
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        adaptive_image_feat = adaptive_image_feat.transpose(0,2).transpose(1,2) ## 204*102*1024

        if self.learnable_mapping == 'bias':
            adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias[:,self.args.indice].unsqueeze(0)  ## class*shot*1024
        elif self.learnable_mapping == 'affine':
            adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.global_ffn_affine[self.args.indice, self.args.indice]
        elif self.learnable_mapping == 'all':
            adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.global_ffn_affine[self.args.indice, self.args.indice] + self.global_ffn_bias[:,self.args.indice].unsqueeze(0)  ## class*shot*1024
        else:
            raise NotImplementedError
        # adaptive_image_feat = adaptive_image_feat + adaptive_image_feat @ self.lora_FFN

        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
        logits = logit_scale * adaptive_image_feat @ img_feat.unsqueeze(-1) ## used feat is not update.
        # ipdb.set_trace()
        # logits = logit_scale *  img_feat @ adaptive_image_feat.t()
        if return_logit:
            return logits[:,:,0]
        else:
            return logits[:,:,0].softmax(dim=1)

    def get_text_prediction(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()
        # ipdb.set_trace()
        ## F=WF + B (W是不同类共享的，B是每个类独有的); 后面这个类别独享的 bias 也太关键了。。。
        # 这里要用full dimension, 只有memory 用selected dim.
        if self.learnable_mapping == 'bias':
            text_feat = model.text_feat + self.text_bias
        elif self.learnable_mapping == 'affine':
            text_feat = model.text_feat + model.text_feat @ self.text_affine
        elif self.learnable_mapping == 'all':
            text_feat = model.text_feat + model.text_feat @ self.text_affine + self.text_bias
        else:
            raise NotImplementedError
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)  ## already filtered with indice.
        ###### 用indice 将text_feat 反装回 full dimension.
        # full_text_feat = text_feat.new_zeros(text_feat.shape[0], 1024)
        # full_text_feat[:, self.args.indice] = text_feat
        img_text_logit = logit_scale * model.image_features_global @ text_feat.t() ## 128*200
        # img_text_logit = logit_scale * model.image_features_global[:, self.args.indice] @ text_feat.t() ## 128*200
        if return_full:
            pass
        else:
            img_text_logit = img_text_logit[:1]
        if return_logit:
            return img_text_logit
        else:
            return img_text_logit.softmax(-1)


def main():
    args = parser.parse_args()
    args.log = args.log + '_' + str(args.gpu)
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
    print('softmax_read', args.softmax_read)
    for set_id in datasets:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                             augmix=len(set_id) > 1) ### aug mix not used for ImageNet test set.
        # data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1, augmix=False)

        test_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution), transforms.ToTensor(), normalize])
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

        model.reset_classnames(classnames, set_id)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,  ## the input has been shuffled.
                    num_workers=args.workers, pin_memory=True)
        args.set_id = set_id
        model.eval()
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
            ######## calculate the similarities
            text_sim = model.text_feat @ model.text_feat.T  ##N*N
            soft_label = (text_sim * args.softmax_label).softmax(1) ## N*N
            model.soft_label = soft_label
            print('max prob of soft label', soft_label.max(1)[0])
        if args.n_shot:
            if args.n_augview == 0:
                train_dataset_mem = build_dataset(set_id, test_transform, args.data, mode='train', n_shot=args.n_shot)
                print("number of training samples: {}".format(len(train_dataset_mem)))
                train_loader_mem = torch.utils.data.DataLoader(
                            train_dataset_mem,
                            batch_size=1, shuffle=False,  ## the input has been shuffled.
                            num_workers=args.workers, pin_memory=True)
                init_image_memory(train_loader_mem, model, args)
                del train_dataset_mem, train_loader_mem
            else:
                ######### generate num_aug_view augmented views for each samples; APE adopt ten...
                assert args.n_augview % args.n_shot == 0
                num_aug_view = int(args.n_augview / args.n_shot)
                data_transform_aug = AugMemAugmenter(base_transform, preprocess, n_views=num_aug_view - 1,
                                                 augmix=len(set_id) > 1)  ### aug mix not used for ImageNet test set.
                train_dataset_mem = build_dataset(set_id, data_transform_aug, args.data, mode='train', n_shot=args.n_shot)
                print("number of training samples: {}, number of augview: {}".format(len(train_dataset_mem), args.n_augview))
                train_loader_mem = torch.utils.data.DataLoader(
                            train_dataset_mem,
                            batch_size=1, shuffle=False,  ## the input has been shuffled.
                            num_workers=args.workers, pin_memory=True)
                init_image_memory(train_loader_mem, model, args)
                del train_dataset_mem, train_loader_mem
        ########## extract the importance channels via APE.
        if args.num_important_channel != 0:
            important_indice = important_channel_indice(args, model) ##
            args.indice = important_indice
        else:
            important_indice = torch.arange(model.text_feat.shape[1]).to(model.text_feat.device) ## use all channels.
            args.indice = important_indice

        results[set_id] = direct_inference(val_loader, model, args)
        # results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

        # log = open(os.path.join(args.log + '.txt'), 'a')
        # state = {k: v for k, v in args._get_kwargs()}
        # log.write(json.dumps(state) + '\n')
        # log.write('                        Target_T1 acc: %3f, local acc: %3f, global acc: : %3f' % (results[set_id][0], results[set_id][1], results[set_id][2]))
        # log.close()
        length = len(results[set_id])
    args.indice = 0
    log = open(os.path.join(args.log + '.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()
    print_log = print_logger(print, os.path.join(args.log + '.txt'))
    print_log("======== Result Summary ========")
    print_log("params: nstep	lr	bs")
    print_log("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print_log("\t\t [set_id] \t\t Top-1 acc. \t\t Top-1 local acc, \t\t Top-1 global acc \t\t Searched acc \t\t beta \t\t gama.")
    for id in results.keys():
        print_log("{}".format(id), end="	")
    print_log('mean', end="	")
    print_log("\n")
    for i in range(length):
        cul_acc = 0
        cul_count = 0
        for id in results.keys():
            print_log("{:.2f}".format(results[id][i]), end="	")
            cul_acc += float(results[id][i])
            cul_count += 1
        print_log("{:.2f}".format(cul_acc), end="	")
        print_log("\n")



def entropy(outputs):
    # prob: 1*200, logit.
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0, keepdim=True) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]; log(mean_prob)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    confidence_entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    return confidence_entropy

def init_image_memory(train_loader, model, args):
    model.eval()
    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    memorized_image_global_feat = [] ## N*[shot*aug]*C
    memorized_image_local_feat = []  ## N*[shot*aug]*C
    memorized_image_global_feat_vanilla = [] ## N*shot*C
    memorized_image_local_feat_vanilla = []  ## N*shot*C
    memorized_labels = []

    for i in range(model.n_cls):
        memorized_image_global_feat.append([])
        memorized_image_local_feat.append([])
        memorized_image_global_feat_vanilla.append([])
        memorized_image_local_feat_vanilla.append([])
        memorized_labels.append([])

    for i, (images, target) in enumerate(train_loader):
        assert args.gpu is not None
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
        else: ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            image_features_global, image_features_local =  model.get_image_features(images) ## 4*1024; 4*49*1024.
        text_features = model.text_feat[target]  ## 512
        ## only use the original ?? we should use all; however, only use the vanilla one in the dynamic memory.
        selected_image_features_local = model.image_features_local
        cos_sim = (selected_image_features_local * text_features).sum(-1)  ## between 0.2-0.3, very close.
        weight_prob = (cos_sim * 100).softmax(-1)   ## 1*197  ## max about 6% - 60%. 这里可能需要再调一下。
        ########
        attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features_local).sum(1)  ## 1*512
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 1*512
        memorized_image_global_feat[target].append(image_features_global) ## aug*C
        memorized_image_local_feat[target].append(attented_feat)   # aug * C
        memorized_image_global_feat_vanilla[target].append(image_features_global[:1]) ## aug*C
        memorized_image_local_feat_vanilla[target].append(attented_feat[:1])   # aug * C
        one_hot_target = torch.zeros(1, model.n_cls).to(target.device)
        one_hot_target[0, target] = 1
        memorized_labels[target].append(one_hot_target)   ## 1 * C, turn it to one hot labels.

    for i in range(model.n_cls):
        memorized_image_global_feat[i] = torch.cat(memorized_image_global_feat[i], dim=0).unsqueeze(0) ## 1*augshot*C
        memorized_image_local_feat[i] = torch.cat(memorized_image_local_feat[i], dim=0).unsqueeze(0)
        memorized_image_global_feat_vanilla[i] = torch.cat(memorized_image_global_feat_vanilla[i], dim=0).unsqueeze(0) ## 1*shot*C
        memorized_image_local_feat_vanilla[i] = torch.cat(memorized_image_local_feat_vanilla[i], dim=0).unsqueeze(0)
        memorized_labels[i] = torch.cat(memorized_labels[i], dim=0).unsqueeze(0)

    memorized_image_global_feat = torch.cat(memorized_image_global_feat, dim=0) ## n*shot*c
    memorized_image_local_feat = torch.cat(memorized_image_local_feat, dim=0)
    memorized_image_global_feat_vanilla = torch.cat(memorized_image_global_feat_vanilla, dim=0) ## n*shot*c
    memorized_image_local_feat_vanilla = torch.cat(memorized_image_local_feat_vanilla, dim=0)
    memorized_labels = torch.cat(memorized_labels, dim=0)

    ######## memorized few shot features and labels.
    model.fewshot_image_global_feat = memorized_image_global_feat ## class*augshot*c
    model.fewshot_image_local_feat = memorized_image_local_feat
    model.fewshot_image_global_feat_vanilla = memorized_image_global_feat_vanilla ## class*shot*c
    model.fewshot_image_local_feat_vanilla = memorized_image_local_feat_vanilla
    model.fewshot_label = memorized_labels  ## class*shot*c, one hot labels

    ############# add features of labeled data to the dynamic memory. This is important when there are more labeled data.
    model.fixed_global_feat_vanilla = torch.cat((model.fixed_global_feat, memorized_image_global_feat_vanilla), dim=1)  ## N*1*C
    model.fixed_local_feat_vanilla = torch.cat((model.fixed_local_feat, memorized_image_local_feat_vanilla), dim=1)  ## N*1*C

    ###################### for static memory, with text feature and augmented image feat
    model.fixed_global_feat = torch.cat((model.fixed_global_feat, memorized_image_global_feat), dim=1)  ## N*1*C
    model.fixed_local_feat = torch.cat((model.fixed_local_feat, memorized_image_local_feat), dim=1)  ## N*1*C
    # ipdb.set_trace()
    ########### image aug 只用在statistics memory, while vanilla image used in dynamic memory.

    print('appending the few shot image feature to fixed image memories.')

def direct_inference(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_local = AverageMeter('Acc@local', ':6.2f', Summary.AVERAGE)
    top1_global = AverageMeter('AccImg@1', ':6.2f', Summary.AVERAGE)
    top1_local_fewshot = AverageMeter('AccLF@1', ':6.2f', Summary.AVERAGE)
    top1_global_fewshot = AverageMeter('AccGF@1', ':6.2f', Summary.AVERAGE)

    top1_text_vote = AverageMeter('AccVote1@1', ':6.2f', Summary.AVERAGE)
    top1_local_fewshot_vote = AverageMeter('AccVoteL@1', ':6.2f', Summary.AVERAGE)
    top1_global_fewshot_vote = AverageMeter('AccVoteG@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top1_local, top1_global, top1_local_fewshot, top1_global_fewshot, top1_text_vote, top1_local_fewshot_vote, top1_global_fewshot_vote],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    ## text_feat: 200*1024
    ## text_feat_full:  200 * 7 * 1024
    class_num, feat_dim = model.text_feat.shape[0], model.text_feat.shape[1]
    pred_vanilla = []
    pred_global = []
    pred_local = []
    pred_fewshot_global = []
    pred_fewshot_local = []
    labels = []
    cliptta = CLIPTTA(args=args, beta=args.beta, feat_dim=feat_dim, class_num=class_num, mapping=args.mapping).cuda()
    ################################ fine tune clip adapter with few labeled training data.
    if args.n_shot and args.ft:
        epoch = args.epoch
        training_size = model.text_feat.shape[0] * args.n_shot
        #### construct the data loader,
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform_aug = StrongAugmenter(base_transform, preprocess, augmix=len(args.set_id) > 1)  ### aug mix not used for ImageNet test set.
        train_dataset_mem = build_dataset(args.set_id, data_transform_aug, args.data, mode='train', n_shot=args.n_shot)
        print("number of training samples: {}, number of augview: {}".format(len(train_dataset_mem), args.n_augview))
        train_loader_ft = torch.utils.data.DataLoader(
            train_dataset_mem,
            batch_size=128 if training_size > 128 else training_size, shuffle=True,  ## the input has been shuffled.
            num_workers=2, pin_memory=True)
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(cliptta.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)  #
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(cliptta.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)  #
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch * len(train_loader_ft), eta_min=1e-7)
        Loss = SmoothCrossEntropy()
        for train_idx in range(epoch): ## for each epoch.
            cliptta.train()
            correct_samples, all_samples = 0, 0
            correct_samples_global, correct_samples_local =  0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, epoch))
            for i, (images, target) in enumerate(train_loader_ft):
                # print(cliptta.lora_b_FFN[0]) ## checked, learned parameters are udpated.
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features_global, image_features_local = model.get_image_features(images) ##B*D, B*L*D
                text_logit = cliptta.get_text_prediction(model, return_full=True, return_logit=True)
                cliptta.init_pred = text_logit  ## use it for local few shot.
                # print(model.soft_label[target])
                soft_label = model.soft_label[target]

                loss = Loss(text_logit, soft_label)
                # ipdb.set_trace()
                fewshot_global_logit= cliptta.get_image_pred_fewshot_global(model, return_full=True, return_logit=True)  ## N*class, probability
                fewshot_local_logit= cliptta.get_image_pred_fewshot_local(model, return_full=True, return_logit=True)  ### to do, get the prediction with local features.
                loss += Loss(fewshot_global_logit, soft_label)
                loss += Loss(fewshot_local_logit, soft_label)

                acc = cls_acc(text_logit, target)
                correct_samples += acc / 100 * len(text_logit)
                all_samples += len(text_logit)
                acc_global = cls_acc(fewshot_global_logit, target)
                correct_samples_global += acc_global / 100 * len(fewshot_global_logit)
                acc_local = cls_acc(fewshot_local_logit, target)
                correct_samples_local += acc_local / 100 * len(fewshot_local_logit)

                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, text:{:}, global: {:}, local: {:} All:{:}), Loss: {:.4f}'.format(current_lr, correct_samples ,
                                                                           correct_samples_global, correct_samples_local, all_samples,
                                                                           sum(loss_list) / len(loss_list)))
    cliptta.eval()
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

        with torch.no_grad():
            img_text = cliptta.get_text_prediction(model)
        img_text_pred = img_text[:1]  ## current prediction.
        confidence_prediction, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text, args.selection_p)
        # ipdb.set_trace()
        ## vote of multiple predictions, this is typically worse than img_text_pred, but introduce information of other views.
        cliptta.init_pred = confidence_prediction.mean(0, keepdim=True)
        acc1, _ = accuracy(cliptta.init_pred, target, topk=(1, 5))
        top1_text_vote.update(acc1[0], image.size(0))

        if args.n_shot:
            with torch.no_grad():
                with torch.no_grad():
                    fewshot_global_pred_fullview = cliptta.get_image_pred_fewshot_global(model) ## N*class, probability
                fewshot_global_pred = fewshot_global_pred_fullview[:1] ## 1*class
                confidence_prediction_fewshot_global, _, _, _ = select_confident_samples(fewshot_global_pred_fullview, 1.0)
                # ########### 这里需要一个 pseudo prediction, 来提供local prediction 的几个备选。
                # cliptta.init_pred = (confidence_prediction.mean(0, keepdim=True) + confidence_prediction_fewshot_global.mean(0, keepdim=True)) / 2  ## provide init guess of the prediction.
                # cliptta.init_pred = confidence_prediction_fewshot_global.mean(0)  ## provide init guess of the prediction.

                acc1, _ = accuracy(confidence_prediction_fewshot_global.mean(0, keepdim=True), target, topk=(1, 5))
                top1_global_fewshot_vote.update(acc1[0], image.size(0))

                with torch.no_grad():
                    fewshot_local_pred_full = cliptta.get_image_pred_fewshot_local(model)
                fewshot_local_pred = fewshot_local_pred_full[:1]
                confidence_prediction_fewshot_local, _, _, _ = select_confident_samples(fewshot_local_pred_full, 1.0)
                cliptta.init_pred = (confidence_prediction.mean(0, keepdim=True) + confidence_prediction_fewshot_global.mean(0, keepdim=True) * args.static_mem_weight
                + confidence_prediction_fewshot_local.mean(0, keepdim=True) * args.static_mem_weight) / (1 + 2* args.static_mem_weight)
                # cliptta.init_pred = confidence_prediction_fewshot_local.mean(0)  ## 如果shot 比较高，这里得准确率也会高很多。

                acc1, _ = accuracy(confidence_prediction_fewshot_local.mean(0, keepdim=True), target, topk=(1, 5))
                top1_local_fewshot_vote.update(acc1[0], image.size(0))

        cliptta.update_memory_bank(model, target)  ## 不更新mem, 应该也是类似 tip 的结果看看能否到达; 这个结果比 TIP 的结果更好
        ##### 直接不update mem, 就可以得到和 APE 差不多的performance; 这样如果update mem 岂不是结果可以非常好？

        with torch.no_grad():
            img_global_pred = cliptta.get_image_pred(model)  ## with updated local
            img_local_pred = cliptta.get_image_pred_local(model)

        # local_vanilla_pred = img_text_pred + img_local_pred
        # comb_prob = img_text_pred + img_local_pred + img_global_pred

        pred_vanilla.append(img_text_pred)
        pred_global.append(img_local_pred)
        pred_local.append(img_global_pred)
        if args.n_shot:
            pred_fewshot_global.append(fewshot_global_pred)
            pred_fewshot_local.append(fewshot_local_pred)
        labels.append(target)

        # # measure accuracy and record loss
        acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
        acc1_global, _ = accuracy(img_global_pred, target, topk=(1, 5))
        acc1_local, _ = accuracy(img_local_pred, target, topk=(1, 5))
        if args.n_shot:
            acc1_local_fs, _ = accuracy(fewshot_local_pred, target, topk=(1, 5))
            acc1_global_fs, _ = accuracy(fewshot_global_pred, target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top1_global.update(acc1_global[0], image.size(0))
        top1_local.update(acc1_local[0], image.size(0))
        if args.n_shot:
            top1_local_fewshot.update(acc1_local_fs[0], image.size(0))
            top1_global_fewshot.update(acc1_global_fs[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        torch.cuda.empty_cache()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()
    pred_vanilla = torch.cat(pred_vanilla, dim=0)
    pred_global = torch.cat(pred_global, dim=0)
    pred_local = torch.cat(pred_local, dim=0)
    if args.n_shot:
        pred_fewshot_global = torch.cat(pred_fewshot_global, dim=0)
        pred_fewshot_local = torch.cat(pred_fewshot_local, dim=0)
    else:
        pred_fewshot_global = pred_vanilla
        pred_fewshot_local = pred_vanilla
    labels = torch.cat(labels, dim=0)
    ########## put the hyper parameter search here.
    ## final prediction = image_text_pred + alpha * global + beta * local
    weight_search = True
    search_step = 10
    if weight_search:
        beta1_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
        beta2_list = beta1_list
        beta3_list = beta1_list
        beta4_list = beta1_list
        beta5_list = beta1_list
        # beta1_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)]  ## 0.001 - 10
        # beta2_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)] ## 0.001 - 10
        # beta3_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)] ## 0.001 - 10
        # beta4_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)] ## 0.001 - 10
        # beta5_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)] ## 0.001 - 10
        print('-' * 20)
        print('Starting searching...')
        print('     beta1 searching range: [0.001, ' + str(10) + ']')
        print('     beta2 searching range: [0.001, ' + str(10) + ']')
        print('     beta3 searching range: [0.001, ' + str(10) + ']')
        print('     beta4 searching range: [0.001, ' + str(10) + ']')
        print('     beta5 searching range: [0.001, ' + str(10) + ']')
        print('-' * 20)

        best_acc = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        best_beta4 = 0.
        best_beta5 = 0.
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                for beta3 in beta3_list:
                    for beta4 in beta4_list:
                        for beta5 in beta5_list:
                            logits = pred_vanilla * beta1 + pred_local * beta2 + pred_global * beta3 * pred_fewshot_local * beta4 + pred_fewshot_global * beta5
                            acc, _ = accuracy(logits, labels, topk=(1, 5))
                            acc = acc.item()
                            if acc > best_acc:
                                print('New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; beta4: {:.4f}; beta5: {:.4f}; Acc: {:.2f}'.format(beta1, beta2,beta3,beta4,beta5, acc))
                                best_acc = acc
                                best_beta1 = beta1
                                best_beta2 = beta2
                                best_beta3 = beta3
                                best_beta4 = beta4
                                best_beta5 = beta5

        print(f"Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.2f}, local weight {best_beta2:.2f} and global weight {best_beta3:.2f} FS local weight {best_beta4:.2f} and FS global weight {best_beta5:.2f}")

    return [top1.avg, top1_local.avg, top1_global.avg, top1_local_fewshot.avg, top1_global_fewshot.avg, best_acc, best_beta1, best_beta2, best_beta3, best_beta4, best_beta5]


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
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--n_augview', type=int, default=0)
    parser.add_argument('--ft', action='store_true', default=False, help="fine tuning the attention weight with few labeled data.")

    parser.add_argument('--note', type=str, default='default', help='some places to write note')
    parser.add_argument('--loss_type', type=str, default='default', help='some places to write note')
    parser.add_argument('--log', type=str, default='softmax', help='some places to write note')
    parser.add_argument('--weight_maxent',  default=1e-3, type=float, help='loss weight')
    parser.add_argument('--beta',  default=5.5, type=float, help='loss weight')

    parser.add_argument('--softmax_read',  default=100, type=float, help='weight in the softmax function')
    parser.add_argument('--softmax_local',  default=100, type=float, help='weight in the softmax function')
    parser.add_argument('--softmax_label',  default=100, type=float, help='the larger the sharper, the smaller the smoother')

    parser.add_argument('--mapping', type=str, default='bias', help='bias | affine | all')

    parser.add_argument('--optimizer', type=str, default='adamw', help='adamw | sgd')
    parser.add_argument('--lr',  default=1e-4, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps, default 1e-8')
    parser.add_argument('--wd',  default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--static_mem_weight', default=1.0, type=float, help='weight decay')

    parser.add_argument('--shared_param', action='store_true', default=False, help="use shared learnable param for local/global/text classifier.")
    parser.add_argument('--num_important_channel', type=int, default=0) ## if 0, use all channels; otherwise, selecting the ape_channel_num
    parser.add_argument('--lambda_ape', default=0.7, type=float, help='0.7 for training free, 0.2 for training required.')

    parser.add_argument('--memory_size', type=int, default=50)
    parser.add_argument('--text_prompt', type=str, default='tip', help='simple | tip | full')


    main()