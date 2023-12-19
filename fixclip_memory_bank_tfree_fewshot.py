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


# def select_confident_samples(logits, top):
#     batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
#     idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] ## pick the min entropy
#     idx_confused = torch.argsort(batch_entropy, descending=False)[int(batch_entropy.size()[0] * top):] ## pick the max entropy
#     return logits[idx], idx, logits[idx_confused], idx_confused

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
    def __init__(self, beta=5.5):
        super(CLIPTTA, self).__init__()
        self.beta = beta
        self.init_pred = 0

    def update_memory_bank(self, model, target):
        # outputs: batch * class.
        # prob = outputs.softmax(dim=1)
        # mean_prob = prob.mean(0, keepdim=True)
        mean_prob = self.init_pred
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
                weight_prob = (cos_sim * 100).softmax(-1)  ## 1*197  ## max about 6% - 60%.

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

    def get_image_pred(self, model):
        img_feat = model.image_features_global[:1]  # 1*1024
        count_image_feat = model.image_feature_count.clone()
        num_class = model.image_feature_memory.shape[0]
        image_classifier = 'similarity_weighted'  ## category_center | weighted_center | similarity_weighted

        ############ use the mix of new text feature and image feature as image feature.
        ## the online updated text feature.
        # merged_image_feat = torch.cat((model.image_feature_memory, model.text_features.unsqueeze(1)), dim=1) ## 200*11*1024
        ## fixed init text features.
        merged_image_feat = torch.cat((model.image_feature_memory, model.fixed_global_feat), dim=1)  ## 200*11*1024
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
            similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
            ####### 这里要修改吗？ 这里实际上还是类似对所有的样本做加权组合？ 只是权重是adaptative 的； 但是目前的权重基本都是相同的差异性没有拉开
            ########### image-image 的相似度是比较高的，0.5-0.6 左右，所以image 贡献了比较多的权重。
            ## 但是最开始image 可能是不准确的； 所以最好开始给 text 较大的权重:发现结果正好相反，开始要给text 比较小的权重，否则结果很差
            ######## 对similarity matrix 的shape 进行重新refine
            filled_image_feat = (merged_image_feat * similarity_matrix.unsqueeze(-1)).sum(1) ## 根据同image feat similarity 进行加权, 得到new classifier.
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        else:
            raise NotImplementedError

    ######## 基于pseudo label, 算top10 的相似度得到10 个特征，然后算10 个prediction.
    def get_image_pred_local(self, model):
        init_pred = self.init_pred   ## class_num, 从里面取出top 5, 来算attention
        value, indice = init_pred.sort(descending=True) ## from large to small.
        cancidate = indice[:10]

        img_feat = model.image_features_local[:1]  # 1*L*1024
        text_features_cancidate = model.text_feat[cancidate]  ## can*1024
        cos_sim = img_feat @ text_features_cancidate.T  ## 1*L*can
        weight_prob = (cos_sim * 100).softmax(1)  ## L softmax.,  1*L*can
        attented_feat = (weight_prob.transpose(1, 2) @ img_feat)  ## 1*can*1024
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 和memorized local image feature 算prediction.

        merged_image_feat = torch.cat((model.local_feature_memory, model.fixed_local_feat), dim=1)  ## 200*11*1024
        similarity_matrix = attented_feat.unsqueeze(1) @ merged_image_feat.transpose(1,2)  ## 1*200*10*11 (view*class*can*memory)
        # similarity_matrix: N*n_cls*can*shot
        ## fewshot_label:   class*shot*class   期望输出：N*can*class
        similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))  ## make it more sharp.
        filled_image_feat = similarity_matrix.squeeze(0) @ merged_image_feat  ## 200*10*1024
        filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True) ## 200*10*1024
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (attented_feat * filled_image_feat).sum(-1)  ## 200*10
        pred_prob = logits.transpose(0,1).softmax(1) ## 10*200

        entropy = -(pred_prob * torch.log(pred_prob + 1e-8)).sum(-1)   ## 10个里挑一个。
        # ipdb.set_trace()

        _, indice_min = entropy.min(0)  ### 这里要判断，这个特征是否是
        return  pred_prob[indice_min].unsqueeze(0)

        # img_feat = model.image_features_local[:1].mean(1)  # 1*1024
        # img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # merged_image_feat = torch.cat((model.local_feature_memory, model.fixed_local_feat), dim=1)  ## 200*11*1024
        # similarity_matrix = (img_feat * merged_image_feat).sum(-1)  ## 200*11  理应-1~1, 但是实际都是 0.1~0.2， 可能和训练方式有关
        # # 如何根据cos sin 从memory 中取 feature 是个需要调超参的地方， 这里直接用的cos similarity, 大概率不是最优的。
        # ######## 对similarity matrix 的shape 进行重新refine
        # similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
        # filled_image_feat = (merged_image_feat * similarity_matrix.unsqueeze(-1)).sum(1)  ## 根据同image feat similarity 进行加权, 得到new classifier.
        # filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
        # logit_scale = model.logit_scale.exp()
        # logits = logit_scale * img_feat @ filled_image_feat.t()
        # return logits.softmax(dim=1)


    def get_image_pred_fewshot_local(self, model):
        # image_features_local: N*L*Channel
        # ipdb.set_trace()
        init_pred = self.init_pred   ## class_num, 从里面取出top 5, 来算attention
        value, indice = init_pred.sort(descending=True) ## from large to small.
        cancidate = indice[:10]
        img_feat = model.image_features_local  # N*L*1024
        text_features_cancidate = model.text_feat[cancidate] ## can*1024
        cos_sim = img_feat @ text_features_cancidate.T  ## N*L*can
        weight_prob = (cos_sim * 100).softmax(1) ## L softmax.,  N*L*can
        attented_feat = (weight_prob.transpose(1,2) @ img_feat) ## N*can*1024
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 和memorized local image feature 算prediction.
        merged_image_feat = model.fewshot_image_local_feat ## class * shot * 1024

        #############  N*1*can*1024 @ clss*1024*shot --> N*class*can*shot
        similarity_matrix = attented_feat.unsqueeze(1) @ merged_image_feat.transpose(1,2)  ## 这里希望的输出是64*102*10*2,
        view_num, class_num, can_num, shot = similarity_matrix.shape
        # similarity_matrix: N*n_cls*can*shot
        ## fewshot_label:   class*shot*class   期望输出：N*can*class
        similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))  ## make it more sharp.
        similarity_matrix = similarity_matrix.transpose(1,2).reshape(view_num, can_num, -1)
        fewshot_labels = model.fewshot_label.view(-1, class_num)

        fewshot_global_pred = similarity_matrix @ fewshot_labels  ## 64*10*class_num， 这里每一个是很多个的和。
        fewshot_global_pred /= fewshot_global_pred.norm(dim=-1, p=1, keepdim=True)  ## prob.
        entropy = -(fewshot_global_pred * torch.log(fewshot_global_pred + 1e-8)).sum(-1)  ## 64*10, 从10个里选出一个
        # ipdb.set_trace()
        view_indice = torch.arange(view_num).to(entropy.device)
        value, indice = entropy.min(1)

        # del similarity_matrix
        # del attented_feat
        return  fewshot_global_pred[view_indice, indice]



        ################# use the feat mean as the local feat.
        # img_feat = model.image_features_local[:1].mean(1)  # 1*1024
        # img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # merged_image_feat = model.fewshot_image_local_feat
        # similarity_matrix = (img_feat * merged_image_feat).sum(-1)  ## 200*11  理应-1~1, 但是实际都是 0.1~0.2， 可能和训练方式有关
        # # 如何根据cos sin 从memory 中取 feature 是个需要调超参的地方， 这里直接用的cos similarity, 大概率不是最优的。
        # ######## 对similarity matrix 的shape 进行重新refine
        # similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1)) ## torch.Size([102, 2])
        # ##### 根据similarity 对 model.fewshot_label 进行加权
        # fewshot_local_pred = (similarity_matrix.unsqueeze(-1) * model.fewshot_label).view(-1, model.n_cls).mean(0, keepdim=True) ## 1*class
        # return fewshot_local_pred  ### 这里是 one_hot * sim, 再求mean 得到的，和小于 1

    def get_image_pred_fewshot_global(self, model):

        img_feat = model.image_features_global   # N*1024
        view_num = img_feat.shape[0]
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        merged_image_feat = model.fewshot_image_global_feat    ## class * shot * channel
        similarity_matrix = (img_feat.unsqueeze(1).unsqueeze(1) * merged_image_feat.unsqueeze(0)).sum(-1)  ## N * class * shot
        # 如何根据cos sin 从memory 中取 feature 是个需要调超参的地方， 这里直接用的cos similarity, 大概率不是最优的。
        ######## 对similarity matrix 的shape 进行重新refine --> (0,1)
        similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1)) ## torch.Size([N, class, shot])
        ##### 根据similarity 对 model.fewshot_label 进行加权
        # similarity_matrix: N, class, shot
        ## fewshot_label:   class*shot*class  --> N*class*shot*class
        fewshot_global_pred = (similarity_matrix.unsqueeze(-1) * model.fewshot_label.unsqueeze(0)).view(view_num, -1, model.n_cls).mean(1) ## N*class
        ### 所有的prediction 值全在均值附近波动
        return fewshot_global_pred / fewshot_global_pred.norm(dim=-1, p=1, keepdim=True) ##  N*class

    # ########################## 今天做好这个就算完工。测试是否
    # def get_image_pred_local(self, model):
    #     ################### 为每一类学一个 累呗独有的 image feature 太耗显存了，要为所有类学一个！
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
        if args.n_shot:
            train_dataset_mem = build_dataset(set_id, test_transform, args.data, mode='train', n_shot=args.n_shot)
            print("number of training samples: {}".format(len(train_dataset_mem)))
            train_loader_mem = torch.utils.data.DataLoader(
                        train_dataset_mem,
                        batch_size=1, shuffle=False,  ## the input has been shuffled.
                        num_workers=args.workers, pin_memory=True)
            init_image_memory(train_loader_mem, model, args)
            del train_dataset_mem, train_loader_mem

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

def init_image_memory(train_loader, model, args):
    model.eval()
    with torch.no_grad():
        text_feat, text_feat_full = model.get_text_features()
    memorized_image_global_feat = [] ## N*shot*C
    memorized_image_local_feat = []  ## N*shot*C
    memorized_labels = []
    for i in range(model.n_cls):
        memorized_image_global_feat.append([])
        memorized_image_local_feat.append([])
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
            image_features_global, image_features_local =  model.get_image_features(images)
        text_features = model.text_feat[target]  ## 512
        selected_image_features_local = model.image_features_local[:1]
        cos_sim = (selected_image_features_local * text_features).sum(-1)  ## between 0.2-0.3, very close.
        weight_prob = (cos_sim * 100).softmax(-1)   ## 1*197  ## max about 6% - 60%. 这里可能需要再调一下。
        ########
        attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features_local).sum(1)  ## 1*512
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 1*512
        memorized_image_global_feat[target].append(image_features_global) ## 1*C
        memorized_image_local_feat[target].append(attented_feat)   # 1 * C
        one_hot_target = torch.zeros(1, model.n_cls).to(target.device)
        one_hot_target[0, target] = 1
        memorized_labels[target].append(one_hot_target)   ## 1 * C, turn it to one hot labels.

    for i in range(model.n_cls):
        memorized_image_global_feat[i] = torch.cat(memorized_image_global_feat[i], dim=0).unsqueeze(0) ## 1*shot*C
        memorized_image_local_feat[i] = torch.cat(memorized_image_local_feat[i], dim=0).unsqueeze(0)
        memorized_labels[i] = torch.cat(memorized_labels[i], dim=0).unsqueeze(0)

    memorized_image_global_feat = torch.cat(memorized_image_global_feat, dim=0) ## n*shot*c
    memorized_image_local_feat = torch.cat(memorized_image_local_feat, dim=0)
    memorized_labels = torch.cat(memorized_labels, dim=0)

    ######## memorized few shot features and labels.
    model.fewshot_image_global_feat = memorized_image_global_feat ## class*shot*c
    model.fewshot_image_local_feat = memorized_image_local_feat
    model.fewshot_label = memorized_labels  ## class*shot*c, one hot labels

    ############# add features of labeled data to the dynamic memory. This is important when there are more labeled data.
    model.fixed_global_feat = torch.cat((model.fixed_global_feat, memorized_image_global_feat), dim=1)  ## N*1*C
    model.fixed_local_feat = torch.cat((model.fixed_local_feat, memorized_image_local_feat), dim=1)  ## N*1*C

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

    pred_vanilla = []
    pred_global = []
    pred_local = []
    pred_fewshot_global = []
    pred_fewshot_local = []
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
        img_text_logit = logit_scale * image_features_global @ model.text_feat.t() ## 128*200
        img_text = img_text_logit.softmax(-1)
        img_text_pred = img_text[:1]  ## current prediction.
        confidence_prediction, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text, args.selection_p)
        # ipdb.set_trace()
        ## vote of multiple predictions, this is typically worse than img_text_pred, but introduce information of other views.
        cliptta.init_pred = confidence_prediction.mean(0)

        acc1, _ = accuracy(cliptta.init_pred.unsqueeze(0), target, topk=(1, 5))
        top1_text_vote.update(acc1[0], image.size(0))

        if args.n_shot:
            with torch.no_grad():
                fewshot_global_pred_fullview = cliptta.get_image_pred_fewshot_global(model) ## N*class, probability
                fewshot_global_pred = fewshot_global_pred_fullview[:1] ## 1*class
                confidence_prediction_fewshot_global, _, _, _ = select_confident_samples(fewshot_global_pred_fullview, args.selection_p)
                ########### 这里需要一个 pseudo prediction, 来提供local prediction 的几个备选。
                # cliptta.init_pred = (confidence_prediction.mean(0) + confidence_prediction_fewshot_global.mean(0)) / 2  ## provide init guess of the prediction.
                # cliptta.init_pred = confidence_prediction_fewshot_global.mean(0)  ## provide init guess of the prediction.

                acc1, _ = accuracy(cliptta.init_pred.unsqueeze(0), target, topk=(1, 5))
                top1_global_fewshot_vote.update(acc1[0], image.size(0))


                fewshot_local_pred_full = cliptta.get_image_pred_fewshot_local(model)
                fewshot_local_pred = fewshot_local_pred_full[:1]
                confidence_prediction_fewshot_local, _, _, _ = select_confident_samples(fewshot_local_pred_full, args.selection_p)
                # cliptta.init_pred = (confidence_prediction.mean(0) + confidence_prediction_fewshot_global.mean(0) * 10 + confidence_prediction_fewshot_local.mean(0)) / 3
                # cliptta.init_pred = confidence_prediction_fewshot_local.mean(0)  ## 如果shot 比较高，这里得准确率也会高很多。

                acc1, _ = accuracy(cliptta.init_pred.unsqueeze(0), target, topk=(1, 5))
                top1_local_fewshot_vote.update(acc1[0], image.size(0))

        cliptta.update_memory_bank(model, target)  ## 不更新mem, 应该也是类似 tip 的结果看看能否到达; 这个结果比 TIP 的结果更好
        ##### 直接不update mem, 就可以得到和 APE 差不多的performance; 这样如果update mem 岂不是结果可以非常好？

        img_global_pred = cliptta.get_image_pred(model)  ## with updated local
        img_local_pred = cliptta.get_image_pred_local(model)

        # local_vanilla_pred = img_text_pred + img_local_pred
        # comb_prob = img_text_pred + img_local_pred + img_global_pred

        pred_vanilla.append(img_text_pred)
        ##################################### 把memory 注释掉，就是tip 的方法，应该能达到tip 的结果？ 可惜达不到，看看为什么;
        # img_local_pred = img_text_pred
        # img_global_pred = img_text_pred
        pred_global.append(img_local_pred)
        pred_local.append(img_global_pred)
        if args.n_shot:
            pred_fewshot_global.append(fewshot_global_pred)
            pred_fewshot_local.append(fewshot_local_pred)
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
        ############# ############# ############# ############# ############# #############  not work, 好像平均就是最好的

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

    return [top1.avg, top1_local.avg, top1_global.avg, top1_local_fewshot.avg, top1_global_fewshot.avg, best_acc, best_beta2, best_beta3]


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
    parser.add_argument('--n_shot', type=int, default=None)


    parser.add_argument('--note', type=str, default='default', help='some places to write note')
    parser.add_argument('--loss_type', type=str, default='default', help='some places to write note')
    parser.add_argument('--log', type=str, default='loga', help='some places to write note')
    parser.add_argument('--weight_maxent',  default=1e-3, type=float, help='loss weight')
    parser.add_argument('--beta',  default=5.5, type=float, help='loss weight')

    parser.add_argument('--memory_size', type=int, default=50)
    parser.add_argument('--text_prompt', type=str, default='tip', help='simple | tip | full')


    main()