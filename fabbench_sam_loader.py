#!/usr/bin/env python3
"""

Requirements:
1. Install the official Segment Anything code so these imports work:
   `from segment_anything import sam_model_registry`
   `from segment_anything.modeling import Sam`
2. Install PyTorch Lightning.
3. Provide the official SAM checkpoint path via `CKPT_ROOT` or override
   `args.checkpoint` in the caller.

"""

from __future__ import annotations

import argparse
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import Sam
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import the official Segment Anything package. "
        "Please download/install the official SAM code before using geometry metrics."
    ) from exc


class _LoRA_Linear(nn.Module):
    def __init__(self, linear_layer: nn.Module, linear_a: nn.Module, linear_b: nn.Module):
        super().__init__()
        self.linear_layer = linear_layer
        self.linear_a = linear_a
        self.linear_b = linear_b

    def forward(self, x):
        return self.linear_layer(x) + self.linear_b(self.linear_a(x))


class _LoRA_qkv(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super().__init__()
        assert r > 0

        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        if lora_layer:
            self.lora_layer_encoder = lora_layer
        else:
            self.lora_layer_encoder = list(range(len(sam_model.image_encoder.blocks)))

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False

        for layer_idx, blk in enumerate(sam_model.image_encoder.blocks):
            if layer_idx not in self.lora_layer_encoder:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])

            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.lora_layer_decoder = list(range(len(sam_model.mask_decoder.transformer.layers)))
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        for layer_idx, blk in enumerate(sam_model.mask_decoder.transformer.layers):
            if layer_idx not in self.lora_layer_decoder:
                continue

            q_proj_layer = blk.self_attn.q_proj
            in_dim = q_proj_layer.in_features
            out_dim = q_proj_layer.out_features
            w_a_linear_q = nn.Linear(in_dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, out_dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            blk.self_attn.q_proj = _LoRA_Linear(q_proj_layer, w_a_linear_q, w_b_linear_q)

            v_proj_layer = blk.self_attn.v_proj
            in_dim = v_proj_layer.in_features
            out_dim = v_proj_layer.out_features
            w_a_linear_v = nn.Linear(in_dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, out_dim, bias=False)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.self_attn.v_proj = _LoRA_Linear(v_proj_layer, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class LoRA_SAM_Lightning(pl.LightningModule):
    """
    Minimal Lightning wrapper kept only for checkpoint compatibility.
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.args = args

        sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        self.lora_sam_model = LoRA_Sam(sam_model, r=args.rank)

        for param in self.lora_sam_model.sam.prompt_encoder.parameters():
            param.requires_grad = False

        embed_dim = sam_model.prompt_encoder.no_mask_embed.embedding_dim
        self.no_mask_embed_top = nn.Embedding(1, embed_dim)
        self.no_mask_embed_top.weight.data.copy_(sam_model.prompt_encoder.no_mask_embed.weight.data)
        self.no_mask_embed_down = nn.Embedding(1, embed_dim)
        self.no_mask_embed_down.weight.data.copy_(sam_model.prompt_encoder.no_mask_embed.weight.data)

    def forward(self, image, mode):
        image_embedding = self.lora_sam_model.sam.image_encoder.forward(image)
        sparse_embeddings = torch.zeros(
            image.shape[0],
            0,
            self.lora_sam_model.sam.prompt_encoder.embed_dim,
            device=image.device,
        )
        dense_embeddings = torch.zeros(
            image.shape[0],
            256,
            self.lora_sam_model.sam.prompt_encoder.image_embedding_size[0],
            self.lora_sam_model.sam.prompt_encoder.image_embedding_size[1],
            device=image.device,
        )
        idx_mask_top = (mode == 1).nonzero(as_tuple=True)[0]
        idx_mask_down = (mode == 2).nonzero(as_tuple=True)[0]
        dense_embeddings[idx_mask_top] = self.no_mask_embed_top.weight.reshape(1, -1, 1, 1).expand(
            1,
            -1,
            self.lora_sam_model.sam.prompt_encoder.image_embedding_size[0],
            self.lora_sam_model.sam.prompt_encoder.image_embedding_size[1],
        ).clone()
        dense_embeddings[idx_mask_down] = self.no_mask_embed_down.weight.reshape(1, -1, 1, 1).expand(
            1,
            -1,
            self.lora_sam_model.sam.prompt_encoder.image_embedding_size[0],
            self.lora_sam_model.sam.prompt_encoder.image_embedding_size[1],
        ).clone()

        low_res_masks, _ = self.lora_sam_model.sam.mask_decoder.forward(
            image_embeddings=image_embedding,
            image_pe=self.lora_sam_model.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(low_res_masks, size=(512, 512), mode="bilinear", align_corners=False)

    def configure_optimizers(self):
        return torch.optim.AdamW([], lr=1e-4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args, _ = parser.parse_known_args()
    return args
