import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .clip_model import CLIP
from method.fushion import ProjectLayer, TextEncoder, Global_Feature, PromptLayer


class Baseline(nn.Module):
    def __init__(self, freeze_clip: CLIP, text_channel: int, visual_channel: int,
                 prompting_length: int, prompting_depth: int, prompting_branch: str, prompting_type: str,
                 use_hsf: bool, k_clusters: int,
                 output_layers: list, device: str, image_size: int):
        super(Baseline, self).__init__()
        self.freeze_clip = freeze_clip

        self.visual = self.freeze_clip.visual
        self.transformer = self.freeze_clip.transformer
        self.token_embedding = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final = self.freeze_clip.ln_final
        self.text_projection = self.freeze_clip.text_projection
        self.attn_mask = self.freeze_clip.attn_mask

        self.output_layers = output_layers

        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters

        if 'L' in self.prompting_branch:
            self.enable_text_prompt = True
        else:
            self.enable_text_prompt = False

        if 'V' in self.prompting_branch:
            self.enable_visual_prompt = True
        else:
            self.enable_visual_prompt = False

        self.text_encoder = TextEncoder(fixed=(not self.enable_text_prompt))
        self.text_prompter = PromptLayer(text_channel, prompting_length, prompting_depth, is_text=True,
                                         enabled=self.enable_text_prompt)
        self.visual_prompter = PromptLayer(visual_channel, prompting_length, prompting_depth, is_text=False,
                                           enabled=self.enable_visual_prompt)

        self.dropout = nn.Dropout(p=0.1)
        self.patch_token_layer = ProjectLayer(
            visual_channel,
            text_channel,
            len(output_layers), stack=False, is_array=True
        )

        self.cls_token_layer = ProjectLayer(
            text_channel,
            text_channel,
            1, stack=False, is_array=False
        )
        self.image_mapping = nn.Linear(text_channel, text_channel)
        self.fuse = Global_Feature(
            dim_i=text_channel,
            dim_out=text_channel,
            k=len(output_layers),
        )
        self.image_size = image_size
        self.device = device

    def encode_image(self, image):

        x = image
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1],
                          self.visual.grid_size[0],
                          self.visual.patch_size[0],
                          self.visual.grid_size[1],
                          self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        patch_embedding = x

        x = x.permute(1, 0, 2)  # NLD -> LND

        patch_tokens = []

        for indx, resblock in enumerate(self.visual.transformer.resblocks):
            # x, attn_tmp = resblock(q_x=x, k_x=None, v_x=None, attn_mask=None)
            # tokens = x 

            x, tokens, attn_tmp = self.visual_prompter(resblock, indx, x, k_x=None, v_x=None, attn_mask=None)

            if (indx + 1) in self.output_layers:
                patch_tokens.append(tokens)

        x = x.permute(1, 0, 2)  # LND -> NLD
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]  # LND -> NLD

        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        return pooled, patch_tokens, patch_embedding

    def proj_visual_tokens(self, image_features, patch_tokens):

        # for patch tokens
        proj_patch_tokens = self.patch_token_layer(patch_tokens)
        
        for layer in range(len(proj_patch_tokens)):
            #proj_patch_tokens[layer] /= proj_patch_tokens[layer].norm(dim=-1, keepdim=True)
            proj_patch_tokens[layer] = proj_patch_tokens[layer] / proj_patch_tokens[layer].norm(dim=-1, keepdim=True)

        # for cls tokens
        # proj_cls_tokens = self.cls_token_layer(image_features)[0]
        proj_cls_tokens = self.dropout(self.cls_token_layer(image_features)[0])

        proj_cls_tokens = proj_cls_tokens / proj_cls_tokens.norm(dim=-1, keepdim=True)

        return proj_cls_tokens, proj_patch_tokens

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for indx, resblock in enumerate(self.transformer.resblocks):
            # x, attn_tmp = resblock(q_x=x, k_x=None, v_x=None, attn_mask=self.attn_mask)

            x, attn_tmp = self.text_prompter(resblock, indx, x, k_x=None, v_x=None, attn_mask=self.attn_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation):
        anomaly_maps = []

        for layer in range(len(patch_token)):
            anomaly_map = (100.0 * patch_token[layer] @ text_feature)
            anomaly_maps.append(anomaly_map)

        cur_image_feature = self.image_mapping(image_feature + self.fuse(patch_token))
        cur_image_feature = cur_image_feature / cur_image_feature.norm(dim=-1, keepdim=True)

        anomaly_score = (100.0 * cur_image_feature.unsqueeze(1) @ text_feature)
        anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)

        # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)

        if aggregation: # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = anomaly_score[:, 1]
            return anomaly_map, anomaly_score
        else: # otherwise, we do the softmax normalization for individual hierarchies
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_score

    def extract_feat(self, image, cls_name):
        image_features, patch_tokens, _ = self.encode_image(image)
        text_features = self.text_encoder(self, cls_name, self.device)

        proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return proj_cls_tokens, proj_patch_tokens, text_features

    @torch.cuda.amp.autocast()
    def forward(self, image, cls_name, aggregation=True):
        # extract features for images and texts
        image_features, patch_tokens, text_features = self.extract_feat(image, cls_name)
        anomaly_map, anomaly_score = self.visual_text_similarity(image_features, patch_tokens, text_features, aggregation)

        if aggregation:
            anomaly_map = anomaly_map # tensor
            anomaly_score = anomaly_score
            anomaly_map = anomaly_map.squeeze(1)

            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_map # list
            anomaly_score = anomaly_score

            return anomaly_maps, anomaly_score

