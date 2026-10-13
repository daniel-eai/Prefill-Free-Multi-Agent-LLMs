from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from .modeling import input_device, layers_of, text_config


def _forward(model, input_ids: torch.Tensor) -> None:
    try:
        model(input_ids, use_cache=False, logits_to_keep=1)
    except TypeError:
        model(input_ids, use_cache=False)


@torch.no_grad()
def capture_kv(tokenizer, model, text: str, layers: Iterable[int], *,
               device=None, dtype=torch.bfloat16):
    layers = sorted(set(int(layer) for layer in layers))
    encoded = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)
    input_ids = torch.tensor(
        list(encoded["input_ids"]), device=input_device(model)).unsqueeze(0)
    blocks = layers_of(model)
    keys: dict[int, torch.Tensor] = {}
    values: dict[int, torch.Tensor] = {}
    hooks = []

    def saver(store, index):
        def save(module, inputs, output):
            del module, inputs
            tensor = output.detach()[0]
            store[index] = tensor.to(
                device=device if device is not None else tensor.device,
                dtype=dtype).contiguous()
        return save

    for index in layers:
        attention = blocks[index].self_attn
        hooks.append(attention.k_proj.register_forward_hook(saver(keys, index)))
        hooks.append(attention.v_proj.register_forward_hook(saver(values, index)))
    try:
        _forward(model, input_ids)
    finally:
        for hook in hooks:
            hook.remove()
    missing = [index for index in layers
               if index not in keys or index not in values]
    if missing:
        raise RuntimeError(f"K/V hooks missed layers: {missing}")
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    return offsets, keys, values


def _token_major(output, heads: int, head_dim: int) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if value.ndim == 3:
        value = value[0]
        if value.shape[-1] != heads * head_dim:
            raise RuntimeError(
                f"projection width {value.shape[-1]} != {heads * head_dim}")
        return value.reshape(-1, heads, head_dim)
    if value.ndim == 4:
        value = value[0]
        if value.shape[1] == heads and value.shape[-1] == head_dim:
            return value
    raise RuntimeError(f"unexpected projection layout {tuple(value.shape)}")


def register_qkv_hooks(
        attention, config, *,
        query: Callable[[torch.Tensor], None] | None = None,
        key: Callable[[torch.Tensor], None] | None = None,
        value: Callable[[torch.Tensor], None] | None = None):
    def normalized(norm_name: str, projection_name: str):
        norm = getattr(attention, norm_name, None)
        return norm if norm is not None else getattr(attention, projection_name)

    modules = {
        "query": normalized("q_norm", "q_proj"),
        "key": normalized("k_norm", "k_proj"),
        "value": attention.v_proj,
    }
    callbacks = {"query": query, "key": key, "value": value}
    heads = {"query": int(config.num_attention_heads),
             "key": int(config.num_key_value_heads),
             "value": int(config.num_key_value_heads)}
    head_dim = int(getattr(
        config, "head_dim",
        config.hidden_size // int(config.num_attention_heads)))
    hooks = []
    for name, callback in callbacks.items():
        if callback is None:
            continue

        def save(module, inputs, output, *, cb=callback, count=heads[name]):
            del module, inputs
            cb(_token_major(output, count, head_dim))

        hooks.append(modules[name].register_forward_hook(save))
    return hooks


@torch.no_grad()
def capture_native_qkv(model, input_ids: torch.Tensor,
                       query_positions: torch.Tensor,
                       dtype=torch.bfloat16):
    config = text_config(model)
    blocks = layers_of(model)
    captures: list[dict] = [{} for _ in blocks]
    hooks = []
    for index, block in enumerate(blocks):
        store = captures[index]

        def save_query(tensor, store=store):
            store["q"] = tensor.index_select(
                0, query_positions.to(tensor.device)).detach().to(dtype)

        def save_key(tensor, store=store):
            store["k"] = tensor.detach().to(dtype)

        def save_value(tensor, store=store):
            store["v"] = tensor.detach().to(dtype)

        hooks.extend(register_qkv_hooks(
            block.self_attn, config, query=save_query, key=save_key,
            value=save_value))
    try:
        _forward(model, input_ids)
    finally:
        for hook in hooks:
            hook.remove()
    missing = [index for index, store in enumerate(captures)
               if set(store) != {"q", "k", "v"}]
    if missing:
        raise RuntimeError(f"incomplete receiver captures: {missing}")
    return captures
