# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from EoMT (https://github.com/tue-mps/eomt).
# Modified by Ronald de Jong, 2025.
# ---------------------------------------------------------------


import logging
import os
from typing import Optional
from urllib.parse import urlparse

import torch
import torch.nn as nn

import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel


# DINOv3 backbones, built with the HF transformers implementation from a local
# config (no gated HF hub download). Matches the SurgeNetDINO pretraining
# configs: no storage/register tokens, unmasked K bias, RoPE base 100 without
# coordinate augmentation, LayerNorm eps 1e-6.
DINOV3_ARCHS = {
    "dinov3_vits16": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6),
    "dinov3_vitb16": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12),
    "dinov3_vitl16": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16),
}


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
        pretrained_weights: Optional[str] = None,
    ):
        """``pretrained_weights``: URL (downloaded once to the torch hub cache)
        or local path of a DINO backbone checkpoint, e.g. the SurgeNetDINO
        weights. It is skipped when ``ckpt_path`` is set, since the full model
        checkpoint then provides the backbone weights."""
        super().__init__()

        load_weights = pretrained_weights is not None and ckpt_path is None

        if backbone_name in DINOV3_ARCHS:
            self.backbone = DINOv3ViTModel(
                self.dinov3_config(backbone_name, img_size, patch_size)
            )
            if load_weights:
                self.load_dinov3_weights(pretrained_weights)
            self.backbone = self.transformers_to_timm(self.backbone, img_size)
        elif "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    backbone_name,
                ),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None and pretrained_weights is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )
            if load_weights:
                self.load_timm_weights(pretrained_weights)

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    @staticmethod
    def dinov3_config(backbone_name: str, img_size: tuple[int, int], patch_size: int):
        arch = DINOV3_ARCHS[backbone_name]
        return DINOv3ViTConfig(
            **arch,
            intermediate_size=4 * arch["hidden_size"],
            patch_size=patch_size,
            image_size=img_size[0],
            num_register_tokens=0,
            key_bias=True,
            layer_norm_eps=1e-6,
            rope_theta=100.0,
            pos_embed_shift=None,
            pos_embed_jitter=None,
            pos_embed_rescale=None,
        )

    @staticmethod
    def read_weights(pretrained_weights: str) -> dict[str, torch.Tensor]:
        if urlparse(pretrained_weights).scheme in ("http", "https"):
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_weights,
                map_location="cpu",
                file_name=os.path.basename(urlparse(pretrained_weights).path),
            )
        else:
            state_dict = torch.load(pretrained_weights, map_location="cpu")

        for key in ("teacher", "model", "state_dict"):
            if isinstance(state_dict.get(key), dict):
                state_dict = state_dict[key]

        return state_dict

    @staticmethod
    def check_loaded(pretrained_weights, missing, unexpected, ignore_prefixes=()):
        """Every backbone parameter must come from the checkpoint; only heads and
        other pretraining-only entries may be left unused."""
        if missing:
            raise RuntimeError(
                f"Backbone weights missing from {pretrained_weights}: {missing}"
            )
        unexpected = [k for k in unexpected if not k.startswith(ignore_prefixes)]
        if unexpected:
            raise RuntimeError(
                f"Unexpected weights in {pretrained_weights}: {unexpected}"
            )
        logging.info("Loaded all backbone weights from %s", pretrained_weights)

    def load_timm_weights(self, pretrained_weights: str):
        # timm's own converter handles DINOv1 and DINOv2 checkpoints (register
        # tokens, mask token, pos_embed resizing to img_size).
        state_dict = checkpoint_filter_fn(
            self.read_weights(pretrained_weights), self.backbone
        )
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        self.check_loaded(
            pretrained_weights,
            missing,
            unexpected,
            ignore_prefixes=("head.", "dino_head.", "ibot_head."),
        )

    def load_dinov3_weights(self, pretrained_weights: str):
        state_dict = self.read_weights(pretrained_weights)

        # RoPE periods are not learned; the model recomputes them in fp32, the
        # checkpoint stores them rounded to bf16 (pos_embed_rope_dtype: bf16).
        periods = state_dict.pop("rope_embed.periods", None)
        if periods is not None:
            expected = 1 / self.backbone.rope_embeddings.inv_freq
            if not torch.allclose(periods.float(), expected, rtol=1e-2):
                raise RuntimeError(
                    f"RoPE periods in {pretrained_weights} do not match the model."
                )

        converted = {}
        for k, v in state_dict.items():
            if k == "mask_token":
                k, v = "embeddings.mask_token", v.reshape(1, 1, -1)
            elif k == "cls_token":
                k = "embeddings.cls_token"
            elif k.startswith("patch_embed.proj."):
                k = k.replace("patch_embed.proj.", "embeddings.patch_embeddings.")
            elif k.startswith("blocks."):
                k = "layer." + k[len("blocks.") :]
                if ".attn.qkv." in k:
                    for name, part in zip(("q", "k", "v"), v.chunk(3, dim=0)):
                        converted[k.replace("attn.qkv.", f"attention.{name}_proj.")] = part
                    continue
                k = (
                    k.replace(".attn.proj.", ".attention.o_proj.")
                    .replace(".gamma_1", ".layer_scale1.lambda1")
                    .replace(".gamma_2", ".layer_scale2.lambda1")
                    .replace(".mlp.fc1.", ".mlp.up_proj.")
                    .replace(".mlp.fc2.", ".mlp.down_proj.")
                )
            converted[k] = v

        missing, unexpected = self.backbone.load_state_dict(converted, strict=False)
        # The register-token parameter is empty (0 tokens), so nothing to load.
        missing = [k for k in missing if k != "embeddings.register_tokens"]
        self.check_loaded(
            pretrained_weights,
            missing,
            unexpected,
            ignore_prefixes=("dino_head.", "ibot_head."),
        )

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone
