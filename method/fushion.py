import math
import numpy as np
import torch
from typing import Union, List, Optional, Tuple, Sequence
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

class LocalGlobalPosterior(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        prompt_dim=512,
        num_prompts=3,
        topk_ratio=0.1,
    ):
        super().__init__()

        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.topk_ratio = topk_ratio

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.posterior = nn.Linear(
            embed_dim,
            2 * num_prompts * prompt_dim
        )

    def forward(
        self,
        global_image_feature,
        patch_features,
        normal_text_features,
        abnormal_text_features,
        sample=True,
    ):
        B, N, D = patch_features.shape

        # [D] or [B, D] -> [B, D]
        if normal_text_features.ndim == 1:
            normal_text_features = (normal_text_features.unsqueeze(0).expand(B, -1))

        if abnormal_text_features.ndim == 1:
            abnormal_text_features = (abnormal_text_features.unsqueeze(0).expand(B, -1))

        # 1. Normalize for cosine similarity
        patches = F.normalize(patch_features, dim=-1)
        normal = F.normalize(normal_text_features, dim=-1)
        abnormal = F.normalize(abnormal_text_features, dim=-1)

        # 2. Text-guided anomaly evidence: [B, N]
        sim_normal = torch.einsum("bnd,bd->bn", patches, normal)
        sim_abnormal = torch.einsum("bnd,bd->bn", patches, abnormal)

        anomaly_evidence = sim_abnormal - sim_normal

        # 3. Select suspicious patches
        k = max(1, min(N, int(N * self.topk_ratio)))

        indices = anomaly_evidence.topk(
            k, dim=1
        ).indices

        selected_patches = torch.gather(
            patch_features,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )

        local_feature = selected_patches.mean(dim=1)

        # 4. Feature dispersion
        std_feature = patch_features.std(
            dim=1, unbiased=False
        )

        # 5. Local-global fusion
        fused_feature = torch.cat(
            [
                global_image_feature,
                local_feature,
                std_feature,
            ],
            dim=-1,
        )

        h = self.fusion(fused_feature)

        # 6. Gaussian posterior
        params = self.posterior(h)
        params = params.reshape(
            B, self.num_prompts, 2 * self.prompt_dim
        )

        mu, logvar = params.chunk(2, dim=-1)

        logvar = logvar.clamp(-10, 4)

        # 7. Reparameterization
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        return z, mu, logvar

class Global_Feature(nn.Module):
    def __init__(self, dim_i=1024, dim_hid=512, dim_out=768, k=4):

        super(Global_Feature, self).__init__()
        
        self.fuse_modules = nn.Linear(dim_i * k, dim_hid)
        self.compress =  nn.Linear(dim_hid, 1)
        self.post_process = nn.Linear(dim_hid, dim_out)


    def forward(self, inps):
        x = torch.cat(inps, dim = 2)
        x = self.fuse_modules(x)
        x_temp = self.compress(x)
        attention_weights = nn.Softmax(dim=1)(x_temp) 
        x = torch.sum(attention_weights * x, dim=1)
        x = self.post_process(x)
        x = F.normalize(x, dim=-1)
        return x

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas, stack=False, is_array=True):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_replicas)])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                temp = self.head[i](tokens[i][:, 1:, :]) # for ViT, we exclude the class token and only extract patch tokens here.
            else:
                temp = self.head[i](tokens)

            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens

class TextEncoder(nn.Module):
    def __init__(self, fixed):
        super(TextEncoder, self).__init__()
        self.tokenizer = _Tokenizer()
        self.ensemble_text_features = {}
        self.prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                              '{} without defect',
                              '{} without damage']
        self.prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = ['a bad photo of a {}.',
                                 'a low resolution photo of the {}.',
                                 'a bad photo of the {}.',
                                 'a cropped photo of the {}.',
                                 ]
        self.fixed = fixed

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
        torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    ## TODO: text layeer is not compitable with multiple batches...
    def forward(self, model, texts, device):
        text_feature_list = []

        for indx, text in enumerate(texts):

            if self.fixed:
                if self.ensemble_text_features.get(text) is None:
                    text_features = self.encode_text(model, text, device)
                    self.ensemble_text_features[text] = text_features
                else:
                    text_features = self.ensemble_text_features[text]
            else:
                text_features = self.encode_text(model, text, device)
                self.ensemble_text_features[text] = text_features

            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=1)

        return text_features

    def encode_text(self, model, text, device):
        text_features = []
        for i in range(len(self.prompt_state)):
            text = text.replace('-', ' ')
            prompted_state = [state.format(text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = self.tokenize(prompted_sentence, context_length=77).to(device)

            class_embeddings = model.encode_text(prompted_sentence)

            #class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            #class_embedding /= class_embedding.norm()
            class_embedding = class_embedding / class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1)

        return text_features
