import math
import numpy as np
import torch
from typing import Union, List, Optional, Tuple, Sequence
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer


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

class PromptLayer(nn.Module):
    def __init__(self, channel, length, depth, is_text, enabled=True):
        super(PromptLayer, self).__init__()
        self.enabled = enabled
        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text

        self.rep_tokens_text_layers = [6, 7, 8, 9, 10, 11, 12]
        self.rep_tokens_visual_layers = [18, 19, 20, 21, 22, 23, 24]
        

        if self.enabled:
            # self.text_prompts = nn.ParameterList(
            #         [nn.Parameter(torch.empty(self.length, self.channel))
            #          for _ in range(self.depth)])
            # self.visual_prompts = nn.ParameterList(
            #         [nn.Parameter(torch.empty(self.length, self.channel))
            #             for _ in range(self.depth)])

            self.text_prompts = nn.ParameterList(
                                [nn.Parameter(torch.empty(self.length, self.channel))
                                 for _ in range(len(self.rep_tokens_text_layers))])
            self.visual_prompts = nn.ParameterList(
                    [nn.Parameter(torch.empty(self.length, self.channel))
                        for _ in range(len(self.rep_tokens_visual_layers))])

            for single_para in self.text_prompts:
                nn.init.normal_(single_para, std=0.02)
            for single_para in self.visual_prompts:
                nn.init.normal_(single_para, std=0.02)

    def forward_text(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            # only prompt the first J layers
            # if indx < self.depth:
            #     textual_context = self.text_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)

            # if indx == 0:  # for the first layer
            #     x = x
            # else:
            #     if indx < self.depth:  # replace with learnalbe tokens
            #         prefix = x[:1, :, :]
            #         suffix = x[1 + length:, :, :]
            #         textual_context = textual_context.permute(1, 0, 2).half()
            #         x = torch.cat([prefix, textual_context, suffix], dim=0)
            #     else:  # keep the same
            #         x = x

            if indx in self.rep_tokens_text_layers:
                if indx == self.rep_tokens_text_layers[0]:
                    width = x.shape[0]
                    prefix = x[:1, :, :]
                    suffix = x[1:, :, :]
                else:
                    width = x.shape[0] - length
                    prefix = x[:1, :, :]
                    suffix = x[1 + length:, :, :]

                index = self.rep_tokens_text_layers.index(indx)
                textual_context = self.text_prompts[index].unsqueeze(0).expand(x.shape[1], -1, -1)
                textual_context = textual_context.permute(1, 0, 2).half()
                x = torch.cat([prefix, textual_context, suffix], dim=0)

            if indx >= self.rep_tokens_text_layers[0]:
                width = x.shape[0]
                attn_mask = x.new_full((width, width), float('-inf')).triu_(1)

        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)

        return x, attn_tmp

    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            # only prompt the first J layers
            # if indx < self.depth:
            #     visual_context = self.visual_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)

            # if indx == 0:  # for the first layer
            #     visual_context = visual_context.permute(1, 0, 2).half()
            #     x = torch.cat([x, visual_context], dim=0)
            # else:
            #     if indx < self.depth:  # replace with learnalbe tokens
            #         prefix = x[0:x.shape[0] - length, :, :]
            #         visual_context = visual_context.permute(1, 0, 2).half()
            #         x = torch.cat([prefix, visual_context], dim=0)
            #     else:  # keep the same
            #         x = x

            if indx in self.rep_tokens_visual_layers:
                if indx == self.rep_tokens_visual_layers[0]:
                    prefix = x[:1, :, :]
                    suffix = x[1:, :, :]
                else:
                    prefix = x[:1, :, :]
                    suffix = x[1 + length:, :, :]

                index = self.rep_tokens_visual_layers.index(indx)
                visual_context = self.visual_prompts[index].unsqueeze(0).expand(x.shape[1], -1, -1)
                visual_context = visual_context.permute(1, 0, 2).half()
                x = torch.cat([prefix, visual_context, suffix], dim=0)

        else:  
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)

        if self.enabled and indx >= self.rep_tokens_visual_layers[0]:
            # Prompts follow CLS; preserve CLS and every spatial patch token.
            tokens = torch.cat([x[:1, :, :], x[1 + self.length:, :, :]], dim=0)
        else:
            tokens = x

        return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.is_text:
            return self.forward_text(resblock, indx, x, k_x, v_x, attn_mask)
        else:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask)

class PromptGenerator(nn.Module):
    def __init__(self, cls_names, device):
        super(PromptGenerator, self).__init__()
        

    def forward(self, class_idx):
        pass
