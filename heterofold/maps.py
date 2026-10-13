from __future__ import annotations

import os
from pathlib import Path

import torch

MAP_KIND = "kv_depth_stencil_recolor"
SOURCE_SPACE = "kv_prerope_joint"
TARGET_SPACE = "kv_pre_norm"


def atomic_save(value, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_map(mapping: dict) -> None:
    if mapping.get("A_k") is None or mapping.get("A_v") is None:
        raise ValueError("not a HeteroFold map: missing A_k / A_v")
    if mapping.get("source_space") != SOURCE_SPACE:
        raise ValueError(
            f"unsupported source_space={mapping.get('source_space')!r}")
    if mapping.get("target_space") != TARGET_SPACE:
        raise ValueError(
            f"unsupported target_space={mapping.get('target_space')!r}")
    if mapping.get("depth_stencil") is None:
        raise ValueError("map has no depth_stencil (sender-layer taps)")
    layers = int(mapping["Lr"])
    for role in "kv":
        if len(mapping[f"A_{role}"]) != layers:
            raise ValueError(f"A_{role} does not cover {layers} receiver layers")


def load_map(path, *, device="cpu", dtype=None, mmap: bool = False) -> dict:
    mapping = torch.load(path, map_location="cpu", weights_only=False,
                         mmap=mmap)
    validate_map(mapping)
    if device != "cpu" or dtype is not None:
        mapping = materialize(mapping, device=device, dtype=dtype)
    return mapping


def materialize(value, device="cuda:0", dtype=None):
    if torch.is_tensor(value):
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device)
    if isinstance(value, list):
        return [materialize(item, device, dtype) for item in value]
    if isinstance(value, dict):
        return {key: materialize(item, device, dtype)
                for key, item in value.items()}
    return value


def sender_layers(mapping: dict) -> list[int]:
    stencil = mapping["depth_stencil"]
    return sorted({int(layer) for role in ("taps_k", "taps_v")
                   for taps in stencil[role] for layer in taps})


def parameter_count(mapping: dict) -> int:
    return sum(int(tensor.numel()) for role in "kv"
               for name in (f"A_{role}", f"b_{role}")
               for tensor in mapping[name])


def describe(mapping: dict) -> dict:
    correction = mapping.get("functional_correction") or {}
    stencil = mapping["depth_stencil"]
    return {
        "map": mapping.get("map"),
        "sender": mapping.get("sender"),
        "receiver": mapping.get("receiver"),
        "sender_layers": int(mapping["Ls"]),
        "receiver_layers": int(mapping["Lr"]),
        "source_kv_width": int(mapping["source_kv_width"]),
        "kv_width": int(mapping["kv_width"]),
        "span": int(stencil["span"]),
        "stride": int(stencil["stride"]),
        "calibration_rank": correction.get("rank", 0),
        "best_validation_loss": correction.get("best_validation_loss"),
        "parameters": parameter_count(mapping),
    }
