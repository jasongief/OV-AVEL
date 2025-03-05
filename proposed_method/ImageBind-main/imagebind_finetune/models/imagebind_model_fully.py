#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial
from types import SimpleNamespace
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F 

from imagebind_finetune.models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from imagebind_finetune.models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from imagebind_finetune.models.transformer import MultiheadAttention, SimpleTransformer

from layers.temporal_av_attn_layer import TemporalAttentionModule, TemporalLinearModule

import pdb

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)



class ImageBindModel(nn.Module):
    def __init__(
        self,
        video_frames=2,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        # spatial attention
        spatial_av_attn_layer_ids=([], []),
        sattn_flag='none',
        # temporal attention
        tattn_flag=False,
        sa_layer_num=1,
        xa_layer_num=1,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=True,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=True,
        win_size=4,
        norm_flag=None,
        # text residual tuning
        text_tune_flag=False,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

        # spatial attention
        self.spatial_av_attn_layer_ids = spatial_av_attn_layer_ids
        self.sattn_flag = sattn_flag
        if self.sattn_flag != 'none':
            print("==> building spatial av layer (spatial attention is True)")
            self.spatial_av_layers = self._create_spatial_av_layers(
                spatial_av_attn_layer_ids,
                audio_embed_dim,
                vision_embed_dim,
            )
        # else:
        #     # self.spatial_av_layers = None
        #     print("==> not performing spatial attention")


        # temporal attention
        self.tattn_flag = tattn_flag
        if self.tattn_flag:
            print("==> building temporal attention layers")
            self.temporal_av_layer = TemporalAttentionModule(
            # print("==> building temporal linear layers")
            # self.temporal_av_layer = TemporalLinearModule(
                sa_layer_num,
                xa_layer_num,
                feat_dim,
                hid_dim,
                d_ff,
                head_num,
                dropout,
                use_adj_in_attn,
                gamma,
                bias,
                use_mask_in_attn,
                win_size,
                norm_flag
            )
        
        # task-residual text learner
        self.text_tune_flag = text_tune_flag
        if self.text_tune_flag:
            print("==> building text projection tuning layers")
            # self.task_res_text_learner = nn.Linear(text_embed_dim, feat_dim, bias=True)
            self.task_res_text_learner = nn.Linear(text_embed_dim, feat_dim, bias=False)
            # self.task_res_alpha = nn.Parameter(torch.FloatTensor([0.])) # or torch.FloatTensor(torch.-inf)
            self.task_res_alpha = nn.Parameter(torch.FloatTensor([float('-inf')])) # or torch.FloatTensor(torch.-inf)

    def _create_spatial_av_layers(
        self,
        spatial_av_attn_layer_ids,
        audio_embed_dim=768,
        vision_embed_dim=1024,
    ):
        audio_lids, vision_lids = spatial_av_attn_layer_ids
        module_a2v = nn.Linear(audio_embed_dim, vision_embed_dim, bias=False)
        module_v2a = nn.Linear(vision_embed_dim, audio_embed_dim, bias=False)
        layers_a2v = nn.ModuleList([copy.deepcopy(module_a2v) for i in range(len(audio_lids))])
        layers_v2a = nn.ModuleList([copy.deepcopy(module_v2a) for i in range(len(vision_lids))])
        
        modality_spatial_attn_layers = {
            ModalityType.AUDIO: layers_a2v,
            ModalityType.VISION: layers_v2a
        }
        return nn.ModuleDict(modality_spatial_attn_layers)



    def _create_modality_preprocessors(
        self,
        video_frames=2,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_embed_dim=512,
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path,
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0), #! 0 denoting selecting first [cls] token
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)


    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        return nn.ModuleDict(modality_postprocessors)


    def spatial_attention(self, audio_tokens, vision_tokens, layer_a2v, layer_v2a):
        # audio_tokens: [229, bs*10, 768]
        # vision_tokens: [257, bs*10, 1024]
        def process_sattn(a_cls_token, v_patch_tokens, layer_a2v):
            a_cls_token = layer_a2v(a_cls_token) # [1, bs*10, 768->1024]
            norm_a_cls_token = F.normalize(a_cls_token, dim=-1) # [1, bs*10, 1024]
            norm_v_patch_tokens = F.normalize(v_patch_tokens, dim=-1) # [256, bs*10, 1024]
            av_simm = torch.sum(torch.mul(norm_a_cls_token, norm_v_patch_tokens), dim=-1) # [256, bs*10]
            # soft_av_simm = F.softmax(av_simm, dim=0).unsqueeze(-1) # [256, bs*10, 1] #! may also try using 'sigmoid'
            # updated_v_patch_tokens = v_patch_tokens + torch.mul(v_patch_tokens, soft_av_simm)
            updated_v_patch_tokens = v_patch_tokens + torch.mul(v_patch_tokens, av_simm.unsqueeze(-1))
            # pdb.set_trace()
            return updated_v_patch_tokens

        a_cls_token = audio_tokens[0, :, :].unsqueeze(0) # [1, bs*10, 768]
        a_patch_tokens = audio_tokens[1:, :, :] # [228, bs*10, 768]
        v_cls_token = vision_tokens[0, :, :].unsqueeze(0) # [1, bs*10, 1024]
        v_patch_tokens = vision_tokens[1:, :, :] # [256, bs*10, 1024]

        updated_a_patch_tokens = process_sattn(v_cls_token, a_patch_tokens, layer_v2a) # [228, bs*10, 768]
        updated_v_patch_tokens = process_sattn(a_cls_token, v_patch_tokens, layer_a2v) # [256, bs*10, 1024]

        updated_a_tokens = torch.cat([a_cls_token, updated_a_patch_tokens], dim=0) # [229, bs*10, 768]
        updated_v_tokens = torch.cat([v_cls_token, updated_v_patch_tokens], dim=0) # [257, bs*10, 1024]
        return updated_a_tokens, updated_v_tokens


    def forward(self, inputs):
        # pdb.set_trace()
        outputs = {}
        # for modality_key, modality_value in inputs.items():
        inputs_temp = {}
        reduce_flag = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  #! Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )
                reduce_flag[modality_key] = True
            else:
                reduce_flag[modality_key] = False
            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                inputs_temp[modality_key] = modality_value
                # trunk_inputs = modality_value["trunk"]
                # head_inputs = modality_value["head"]
                # inputs_temp[modality_key]['trunk'] = trunk_inputs
                # inputs_temp[modality_key]['head'] = head_inputs
            # pdb.set_trace()
        ###################! processing text embedding ####################
        text_modality_key = 'text'
        text_trunk_inputs = inputs_temp[text_modality_key]['trunk']
        text_head_inputs = inputs_temp[text_modality_key]['head']
        text_transformer_blocks = self.modality_trunks[text_modality_key].blocks
        text_tokens = text_trunk_inputs['tokens'] #! Transformer encoders runs here
        #! add several learnable tokens here
        # TODO
        if self.modality_trunks[text_modality_key].pre_transformer_layer:
            text_tokens = self.modality_trunks[text_modality_key].pre_transformer_layer(text_tokens)
        # pdb.set_trace()
        for blk_id, blk in enumerate(text_transformer_blocks):
            #! processing each Transformer block
            text_tokens = blk(text_tokens, attn_mask=text_trunk_inputs['attn_mask'])
        # pdb.set_trace()
        if self.modality_trunks[text_modality_key].post_transformer_layer:
            text_tokens = self.modality_trunks[text_modality_key].post_transformer_layer(text_tokens)
        text_modality_value = text_tokens

        text_modality_value = self.modality_heads[text_modality_key](
            text_modality_value, **text_head_inputs
        )
        text_modality_value = self.modality_postprocessors[text_modality_key](
            text_modality_value
        )
        if reduce_flag[text_modality_key]:
            text_modality_value = text_modality_value.reshape(B, S, -1)
            # text_modality_value = text_modality_value.mean(dim=1)
        # outputs[text_modality_key] = text_modality_value # [46/67 + 1, 1024]
        #! learnable text projection
        if self.text_tune_flag:
            # print("==> performing text projetion tuning")
            # pdb.set_trace()
            text_modality_value = text_modality_value + torch.sigmoid(self.task_res_alpha) * self.task_res_text_learner(text_modality_value)
            # pdb.set_trace()
        # text_modality_value = text_modality_value + 0.5 * self.task_res_text_learner(text_modality_value) # fixed version


        ###################! pre-processing audio and visual embedding ####################
        # audio tokens before sending to audio Transformer blocks
        audio_modality_key = 'audio'
        audio_trunk_inputs = inputs_temp[audio_modality_key]['trunk']
        audio_head_inputs = inputs_temp[audio_modality_key]['head']
        audio_transformer_blocks = self.modality_trunks[audio_modality_key].blocks
        audio_tokens = audio_trunk_inputs['tokens'] #! Transformer encoders runs here
        if self.modality_trunks[audio_modality_key].pre_transformer_layer:
            audio_tokens = self.modality_trunks[audio_modality_key].pre_transformer_layer(audio_tokens) # [229, bs*10, 768]
        # vision tokens before sending to vision Transformer blocks
        vision_modality_key = 'vision'
        vision_trunk_inputs = inputs_temp[vision_modality_key]['trunk']
        vision_head_inputs = inputs_temp[vision_modality_key]['head']
        vison_transformer_blocks = self.modality_trunks[vision_modality_key].blocks
        vision_tokens = vision_trunk_inputs['tokens'] #! Transformer encoders runs here
        if self.modality_trunks[vision_modality_key].pre_transformer_layer:
            vision_tokens = self.modality_trunks[vision_modality_key].pre_transformer_layer(vision_tokens) # [257, bs*10, 1024]

        # pdb.set_trace()        
        sattn_type = self.sattn_flag
        if sattn_type == 'none':
            for audio_blk_id, audio_blk in enumerate(audio_transformer_blocks):
                audio_tokens = audio_blk(audio_tokens, attn_mask=None)
            for vision_blk_id, vision_blk in enumerate(vison_transformer_blocks):
                vision_tokens = vision_blk(vision_tokens, attn_mask=None)
        else: #! learnable mutual audio-vision spatial attention
            # print("==> performing spatial attention, type: ", sattn_type)
            a2v_modulelist, v2a_modulelist = self.spatial_av_layers[audio_modality_key], self.spatial_av_layers[vision_modality_key]
            a2v_layer_ids, v2a_layer_ids = self.spatial_av_attn_layer_ids # list
            audio_blocks_num, vision_blocks_num = len(audio_transformer_blocks), len(vison_transformer_blocks) # 12, 32
            if sattn_type in ['bothFirst', 'bothLast', 'evenFirst', 'evenLast', 'fixedBlkids']:
                a_blk_id, v_blk_id = 0, 0
                for i in range(len(a2v_layer_ids)):
                    while(a_blk_id <= a2v_layer_ids[i]):
                        audio_tokens = audio_transformer_blocks[a_blk_id](audio_tokens, attn_mask=None)
                        a_blk_id += 1
                    while(v_blk_id <= v2a_layer_ids[i]):
                        vision_tokens = vison_transformer_blocks[v_blk_id](vision_tokens, attn_mask=None)
                        v_blk_id += 1
                    audio_tokens, vision_tokens = self.spatial_attention(audio_tokens, vision_tokens, a2v_modulelist[i], v2a_modulelist[i]) #!
                for ai in range(a_blk_id, audio_blocks_num):
                    audio_tokens = audio_transformer_blocks[ai](audio_tokens, attn_mask=None)
                for vi in range(v_blk_id, vision_blocks_num):
                    vision_tokens = vison_transformer_blocks[vi](vision_tokens, attn_mask=None)
            else:
                raise NotImplementedError

        # audio token post processing in origianl Imagebind model
        if self.modality_trunks[audio_modality_key].post_transformer_layer:
            audio_tokens = self.modality_trunks[audio_modality_key].post_transformer_layer(audio_tokens)
        audio_modality_value = audio_tokens
        audio_modality_value = self.modality_heads[audio_modality_key](
            audio_modality_value, **audio_head_inputs
        )
        audio_modality_value = self.modality_postprocessors[audio_modality_key](
            audio_modality_value
        )
        if reduce_flag[audio_modality_key]:
            audio_modality_value = audio_modality_value.reshape(B, S, -1)
            # audio_modality_value = audio_modality_value.mean(dim=1)
        # outputs[audio_modality_key] = audio_modality_value # [bs, 10, 1024], only select one [cls] token of each audio segment


        # vision token post processing
        if self.modality_trunks[vision_modality_key].post_transformer_layer:
            vision_tokens = self.modality_trunks[vision_modality_key].post_transformer_layer(vision_tokens)
        vision_modality_value = vision_tokens
        vision_modality_value = self.modality_heads[vision_modality_key](
            vision_modality_value, **vision_head_inputs
        )
        vision_modality_value = self.modality_postprocessors[vision_modality_key](
            vision_modality_value
        )
        if reduce_flag[vision_modality_key]:
            vision_modality_value = vision_modality_value.reshape(B, S, -1)
            # vision_modality_value = vision_modality_value.mean(dim=1)
        # outputs[vision_modality_key] = vision_modality_value # [bs, 10, 1024], only [cls] token of each frame
        if  torch.sum(vision_modality_value.isnan()) > 0 or torch.sum(audio_modality_value.isnan()) > 0:
            pdb.set_trace()

        #! learnable temporal audio-vision attention
        if self.tattn_flag:
            # print("==> performing temporal attention")
            audio_modality_value, vision_modality_value = self.temporal_av_layer(audio_modality_value, vision_modality_value) # [bs, 10, 1024]
        if  torch.sum(vision_modality_value.isnan()) > 0 or torch.sum(audio_modality_value.isnan()) > 0:
            pdb.set_trace()

        outputs[text_modality_key] = text_modality_value
        outputs[audio_modality_key] = audio_modality_value
        outputs[vision_modality_key] = vision_modality_value
        outputs['pred'] = None

        return outputs



def imagebind_huge(
    pretrained=False,
    # spatial attention
    spatial_av_attn_layer_ids=([0], [0]),
    sattn_flag='none',
    # temporal attention
    tattn_flag=False,
    sa_layer_num=1,
    xa_layer_num=1,
    feat_dim=1024,
    hid_dim=256,
    d_ff=512,
    head_num=1,
    dropout=0.1,
    use_adj_in_attn=False,
    gamma=0.6,
    bias=0.2,
    use_mask_in_attn=False,
    win_size=4,
    norm_flag=None,
    # text tuning
    text_tune_flag=False,
):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
        # spatial attention
        spatial_av_attn_layer_ids=spatial_av_attn_layer_ids,
        sattn_flag=sattn_flag,
        # temporal attention
        tattn_flag=tattn_flag,
        sa_layer_num=sa_layer_num,
        xa_layer_num=xa_layer_num,
        feat_dim=feat_dim,
        hid_dim=hid_dim,
        d_ff=d_ff,
        head_num=head_num,
        dropout=dropout,
        use_adj_in_attn=use_adj_in_attn,
        gamma=gamma,
        bias=bias,
        use_mask_in_attn=use_mask_in_attn,
        win_size=win_size,
        norm_flag=norm_flag,
        # text resudual tuning
        text_tune_flag=text_tune_flag,
    )

    def initialize_imagebind_weights(model):
        imagebind_model_dict = model.state_dict()
        pretrained_path = '/root/autodl-tmp/OV_AVEL/proposed_method/ImageBind-main/.checkpoints/imagebind_huge.pth'
        pretrained_state_dicts = torch.load(pretrained_path)
        # pretrained_state_dicts = torch.load(".checkpoints/imagebind_huge.pth")
        state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in imagebind_model_dict.keys()}
        imagebind_model_dict.update(state_dict)
        model.load_state_dict(imagebind_model_dict)
        # pdb.set_trace()
        print("==> Load pretrained Imagemodel parameters")
        return model

    if pretrained:
        pretrained_path = '/root/autodl-tmp/OV_AVEL/proposed_method/ImageBind-main/.checkpoints/imagebind_huge.pth'
        if not os.path.exists(pretrained_path):
        # if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        # model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))
        model = initialize_imagebind_weights(model)

    return model
