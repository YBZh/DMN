
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes, imagenet_templates, tip_imagenet_templates, simple_imagenet_template, ID_to_prompts, ID_to_gptprompts_path
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
import json
import ipdb

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):   ### add image prompt here.
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls # False
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]  ## 512
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:  ## a photo of a
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            ipdb.set_trace()
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" ")) ## 4
            prompt = tokenize(ctx_init).to(self.device) ## 1*77
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)  ## torch.Size([1, 77, 512])
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]  ## torch.Size([4, 512])
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()  ## torch.Size([4, 512])
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        self.ctx_std = nn.Parameter(torch.ones(ctx_vectors.shape) * 1e-4)  # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]  ## ['a photo of a agaric.', ]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device) ## torch.Size([1000, 77])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) ## torch.Size([1000, 77, 512])

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Size([1000, 77])
        self.name_lens = name_lens
        self.class_token_position = ctx_position  ## end
        self.n_cls = n_cls # 1000
        self.n_ctx = n_ctx ## 4
        self.classnames = classnames


    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx = self.ctx.copy_(ctx_vectors)  # to be optimized torch.Size([4, 512])
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)  ## 200
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames] ## 200
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)  ## torch.Size([200, 77])

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)  ## torch.Size([200, 77, 512])

        self.token_prefix = embedding[:, :1, :] ## 200*1*512 前缀
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS ## torch.Size([200, 72, 512]) 后缀

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts  ## torch.Size([200, 77])
        self.classnames = classnames


    def forward(self, init=None, with_std=True):
        # the init will be used when computing CLIP directional loss
        # ipdb.set_trace()
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipFixed(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, memory_size=10, text_prompt='tip'):
        super(ClipFixed, self).__init__()
        clip, _, transform = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        print('clip transform', transform)
        self.clip = clip
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.first_flag = True
        self.memory_size = memory_size
        self.return_local_feat = False
        self.text_prompt_type = text_prompt

        self.logit_scale = clip.logit_scale.data
        self.text_feat = None
        self.few_shot_mem = False
        # self.n_cls = len(classnames)  ## 200
        # self.image_encoder = clip.visual
        # # ipdb.set_trace()
        # self.text_encoder = TextEncoder(clip)
        # # prompt tuning
        # self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        # self.criterion = criterion

        
    # @property
    # def dtype(self):
    #     return self.image_encoder.conv1.weight.dtype

    # # restore the initial state of the prompt_learner (tunable prompt)
    # def reset(self):
    #     self.prompt_learner.reset()
    #
    def reset_classnames(self, classnames, test_sets):
        self.n_cls = len(classnames)  ## 200
        self.classnames = [name.replace("_", " ") for name in classnames]
        print('class number:', self.n_cls)
        if self.text_prompt_type == 'simple':
            self.text_prompt = simple_imagenet_template  ## ['a photo of a {}.']
        elif self.text_prompt_type =='tip':
            if len(test_sets)>1:
                self.text_prompt = ID_to_prompts[test_sets.lower()]
            else:
                self.text_prompt = tip_imagenet_templates  ## seven text prompts
        elif self.text_prompt_type =='tip_cupl':
            if len(test_sets)>1:
                self.text_prompt = ID_to_prompts[test_sets.lower()]
                self.cupl_file = ID_to_gptprompts_path[test_sets.lower()]
            else:
                self.text_prompt = tip_imagenet_templates  ## seven text prompts
                self.cupl_file = "CuPL_prompts_imagenet.json"
            f = open('./data/gpt3_prompts/' + self.cupl_file)
            self.cupl_prompts = json.load(f)
        elif self.text_prompt_type == 'full':
            self.text_prompt = imagenet_templates
        else:
            raise NotImplementedError
        print('test sets, prompt', test_sets,  self.text_prompt)
        # ipdb.set_trace()
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [self.prompt_prefix + " " + name + "." for name in classnames] ## 200
        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)  ## torch.Size([200, 77])
        #
        # clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
        #
        # with torch.no_grad():
        #     embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)  ## torch.Size([200, 77, 512])
        #
        # self.token_prefix = embedding[:, :1, :] ## 200*1*512 前缀
        # self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS ## torch.Size([200, 72, 512]) 后缀
        #
        # self.name_lens = name_lens
        # self.tokenized_prompts = tokenized_prompts  ## torch.Size([200, 77])
        # self.classnames = classnames
        self.first_flag = True

    def get_text_features(self):
        ## get the text feature only once, multiple class & multiple prompt
        text_feat = []
        text_label = []
        text_full = []
        count = 0
        for name in self.classnames:
            text_prompts = [template.format(name) for template in self.text_prompt]  # format with class
            if self.text_prompt_type =='tip_cupl':
                text_prompts += self.cupl_prompts[name]
            texts = tokenize(text_prompts).cuda()  # tokenize
            class_embeddings = self.clip.encode_text(texts)  # embed with text encoder
            class_embeddings_full = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_full.append(class_embeddings_full) ## different length for each category.
            class_embedding_mean = class_embeddings_full.mean(dim=0)
            class_embedding_mean /= class_embedding_mean.norm()
            text_feat.append(class_embedding_mean) ### 1024
            one_hot_target = torch.zeros(self.n_cls).to(class_embedding_mean.device)
            one_hot_target[count] = 1
            text_label.append(one_hot_target)  ## 1 * d, turn it to one hot labels.
            count = count + 1
        self.text_feat = torch.stack(text_feat, dim=0).cuda() ## N*1024
        self.text_label = torch.stack(text_label, dim=0).cuda()  ## N*N
        for i in range(len(text_full)):
            text_full[i] = text_full[i].cuda()

        self.text_feat_full = text_full ## not used.
        # ipdb.set_trace()
        ######## 直接从这里找出 important text feat following APE. TO DO
        self.fixed_global_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_global_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C

        self.fixed_global_label = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label = self.text_label.clone().unsqueeze(1)
        self.fixed_global_label_vanilla = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label_vanilla = self.text_label.clone().unsqueeze(1)

        if self.first_flag:  ## initlize
            self.image_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device)       ## 如果满了，把entropy 最高的扔出去
            self.image_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.image_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.image_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            self.local_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device)
            self.local_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.local_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.local_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            self.first_flag = False

        return self.text_feat, self.text_feat_full

        # text_features = []
        # prompts = self.prompt_learner(with_std=True)  ## torch.Size([1000, 77, 512])
        # tokenized_prompts = self.prompt_learner.tokenized_prompts
        # t_features = self.text_encoder(prompts, tokenized_prompts)  ## torch.Size([1000, 1024])
        # text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        # self.num_class = t_features.size(0)
        # text_features = torch.stack(text_features, dim=0)
        # # return text_features
        #
        # return torch.mean(text_features, dim=0)

    def get_image_features(self, image):
        # image_features_vanilla = self.image_encoder(image.type(self.dtype))
        ## for Res50 128*1024 or 128*50*1024 [global feat; 7*7 local feature]
        ## for VIT,  128*512 or 128*197*512 [global feat; 14*14 local features]
        image_features = self.clip.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_local = image_features[:,1:,:]  ## B*L*C
        image_features_global = image_features[:, 0, :] ## B*C

        self.image_features_local = image_features_local
        self.image_features_global = image_features_global

        return self.image_features_global, self.image_features_local

        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        # return logits

    def forward(self, input):
        pass
        # if isinstance(input, Tuple):
        #     view_0, view_1, view_2 = input
        #     return self.contrast_prompt_tuning(view_0, view_1, view_2)
        # elif len(input.size()) == 2:
        #     return self.directional_prompt_tuning(input)
        # else:
        #     return self.inference(input)


def get_fixed_clip(clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False, memory_size=10, text_prompt='tip'):

    model = ClipFixed(device, classnames, None, arch=clip_arch, n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, memory_size=memory_size, text_prompt=text_prompt)

    return model

