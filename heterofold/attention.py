from __future__ import annotations

import torch

from .modeling import decoder, text_config


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def apply_rope(x: torch.Tensor, cosine: torch.Tensor,
               sine: torch.Tensor) -> torch.Tensor:
    return x * cosine[:, None] + rotate_half(x) * sine[:, None]


def rotary_cos_sin(model, positions: torch.Tensor, layer_index: int,
                   dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    config = text_config(model)
    rotary = decoder(model).rotary_emb
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    dummy = torch.zeros(
        1, 1, int(config.hidden_size), device=positions.device, dtype=dtype)
    layer_types = getattr(config, "layer_types", None)
    if (layer_types is not None
            and isinstance(getattr(rotary, "rope_type", None), dict)):
        cosine, sine = rotary(
            dummy, positions, layer_type=layer_types[layer_index])
    else:
        cosine, sine = rotary(dummy, positions)
    return cosine[0].to(dtype), sine[0].to(dtype)


def output_projection(attention):
    for name in ("o_proj", "out_proj", "dense"):
        module = getattr(attention, name, None)
        if module is not None:
            return module
    raise AttributeError("attention module exposes no output projection")


def causal_mask(attention, query_positions: torch.Tensor,
                length: int) -> torch.Tensor:
    token_positions = torch.arange(
        length, device=query_positions.device, dtype=torch.long)
    mask = token_positions[None, :] <= query_positions[:, None]
    window = getattr(attention, "sliding_window", None)
    if window is not None and int(window) > 0:
        mask &= token_positions[None, :] > (
            query_positions[:, None] - int(window))
    return mask


def softcap_logits(attention, logits: torch.Tensor) -> torch.Tensor:
    cap = getattr(attention, "attn_logit_softcapping", None)
    if cap is None:
        cap = getattr(getattr(attention, "config", None),
                      "attn_logit_softcapping", None)
    if cap is None:
        return logits
    return torch.tanh(logits / float(cap)) * float(cap)


def apply_native_query_scale(attention, query: torch.Tensor,
                             positions: torch.Tensor) -> torch.Tensor:
    config = getattr(attention, "config", None)
    if getattr(config, "model_type", None) != "ministral3":
        return query
    parameters = config.rope_parameters
    beta = parameters.get("llama_4_scaling_beta")
    if beta is None:
        return query
    original = parameters["original_max_position_embeddings"]
    scale = 1 + beta * torch.log1p(torch.floor(
        positions.to(device=query.device, dtype=torch.float32) / original))
    return query * scale.to(query.dtype)[:, None, None]


def attention_probabilities(attention, query: torch.Tensor, key: torch.Tensor,
                            mask: torch.Tensor, groups: int) -> torch.Tensor:
    repeated_key = key.repeat_interleave(groups, dim=1)
    logits = torch.einsum("phd,thd->hpt", query, repeated_key)
    logits.mul_(float(attention.scaling))
    logits = softcap_logits(attention, logits)
    logits.masked_fill_(~mask[None], float("-inf"))
    return torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
