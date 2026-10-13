from __future__ import annotations

import torch

from .alignment import receiver_to_sender_indices
from .attention import rotate_half
from .capture import capture_kv
from .maps import sender_layers
from .modeling import decoder, input_device, layers_of, text_config


@torch.no_grad()
def capture_sender(tokenizer, model, mapping: dict, text: str, device=None):
    return capture_kv(tokenizer, model, text, sender_layers(mapping),
                      device=device, dtype=torch.bfloat16)


def source_indices(sender_offsets, receiver_offsets, device) -> torch.Tensor:
    return torch.tensor(
        receiver_to_sender_indices(sender_offsets, receiver_offsets),
        device=device, dtype=torch.long)


def map_layer(mapping: dict, keys: dict, values: dict, source: torch.Tensor,
              layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    stencil = mapping["depth_stencil"]
    weight_k, weight_v = mapping["A_k"][layer], mapping["A_v"][layer]
    source_k = torch.cat([
        keys[int(tap)].to(device=weight_k.device, dtype=weight_k.dtype)[source]
        for tap in stencil["taps_k"][layer]], dim=-1)
    key = (source_k / mapping["rms_k"][layer]) @ weight_k + mapping["b_k"][layer]
    del source_k
    source_v = torch.cat([
        values[int(tap)].to(device=weight_v.device, dtype=weight_v.dtype)[source]
        for tap in stencil["taps_v"][layer]], dim=-1)
    value = (source_v / mapping["rms_v"][layer]) @ weight_v + mapping["b_v"][layer]
    del source_v
    return key.to(torch.bfloat16), value.to(torch.bfloat16)


def finalize_layer(model, layer: int, key: torch.Tensor, value: torch.Tensor,
                   cosine: torch.Tensor, sine: torch.Tensor):
    config = text_config(model)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads))
    attention = layers_of(model)[layer].self_attn
    tokens = key.shape[0]
    key = key.float().view(tokens, kv_heads, head_dim)
    value = value.float().view(tokens, kv_heads, head_dim)
    key_norm = getattr(attention, "k_norm", None)
    if key_norm is not None:
        parameter = next(key_norm.parameters(), None)
        norm_device = parameter.device if parameter is not None else key.device
        key = key_norm(key.to(device=norm_device, dtype=torch.bfloat16)).float()
    key = key.transpose(0, 1)
    value = value.transpose(0, 1)
    rotated = key * cosine.unsqueeze(0) + rotate_half(key) * sine.unsqueeze(0)
    return rotated.to(torch.bfloat16), value.to(torch.bfloat16)


def _rotary_tables(model, positions: torch.Tensor):
    config = text_config(model)
    rotary = decoder(model).rotary_emb
    dummy = torch.zeros(1, 1, config.hidden_size, device=positions.device,
                        dtype=torch.bfloat16)
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(getattr(rotary, "rope_type", None), dict):
        layer_types = None
    tables = {}

    def table(layer: int):
        layer_type = None if layer_types is None else layer_types[layer]
        if layer_type not in tables:
            if layer_type is None:
                cosine, sine = rotary(dummy, positions.unsqueeze(0))
            else:
                cosine, sine = rotary(dummy, positions.unsqueeze(0),
                                      layer_type=layer_type)
            tables[layer_type] = (cosine[0], sine[0])
        return tables[layer_type]

    return table


def _cache_layer(cache, layer: int):
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


@torch.no_grad()
def build_receiver_cache(tokenizer, model, mapping: dict, sender_offsets,
                         sender_keys: dict, sender_values: dict, text: str,
                         prefix):
    device = input_device(model)
    if isinstance(prefix, str):
        prefix_ids = tokenizer(prefix, add_special_tokens=False,
                               return_tensors="pt").input_ids
    else:
        prefix_ids = torch.as_tensor(prefix, dtype=torch.long).reshape(1, -1)
    prefix_ids = prefix_ids.to(device)
    prefix_length = prefix_ids.shape[1]
    cache = model(prefix_ids, use_cache=True).past_key_values

    encoded = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)
    map_device = mapping["A_k"][0].device
    source = source_indices(
        sender_offsets, encoded["offset_mapping"], map_device)
    tokens = int(source.numel())
    positions = torch.arange(prefix_length, prefix_length + tokens,
                             device=device)
    rotary_table = _rotary_tables(model, positions)
    for layer in range(int(mapping["Lr"])):
        key, value = map_layer(mapping, sender_keys, sender_values, source,
                               layer)
        cosine, sine = rotary_table(layer)
        cache_device = _cache_layer(cache, layer)[0].device
        rotated, value = finalize_layer(
            model, layer, key, value, cosine, sine)
        cache.update(rotated.unsqueeze(0).to(cache_device),
                     value.unsqueeze(0).to(cache_device), layer)
        del key, value, rotated
    return cache, prefix_length + tokens


@torch.no_grad()
def transfer(sender_tokenizer, sender_model, receiver_tokenizer,
             receiver_model, mapping: dict, text: str, prefix):
    map_device = mapping["A_k"][0].device
    offsets, keys, values = capture_sender(
        sender_tokenizer, sender_model, mapping, text, device=map_device)
    cache, length = build_receiver_cache(
        receiver_tokenizer, receiver_model, mapping, offsets, keys, values,
        text, prefix)
    return cache, length, len(offsets)
