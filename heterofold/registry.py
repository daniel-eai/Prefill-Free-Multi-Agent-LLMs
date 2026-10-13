from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    display: str


MODELS: dict[str, ModelSpec] = {
    "llama8": ModelSpec(
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659", "Llama-3.1-8B"),
    "qwen4": ModelSpec(
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c", "Qwen3-4B"),
    "ministral14": ModelSpec(
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "3cea74c1ebaf5ce5f5a2553de470e2ceab825142", "Ministral-3-14B"),
}

_ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "model-*.safetensors", "model.safetensors", "tokenizer*",
    "special_tokens_map.json", "vocab.json", "merges.txt", "tekken.json",
    "processor_config.json", "preprocessor_config.json", "chat_template*",
    "params.json",
]


def parse_pair(pair: str) -> tuple[str, str]:
    if pair.count("_to_") != 1:
        raise ValueError(
            f"pair must look like '<sender>_to_<receiver>', got {pair!r}")
    sender, receiver = pair.split("_to_")
    return sender, receiver


def display_name(name: str) -> str:
    return MODELS[name].display if name in MODELS else Path(name).name


def resolve_model(name: str, *, download: bool = True) -> str:
    override = os.environ.get(f"HETEROFOLD_{name.upper()}_PATH")
    if override:
        if not os.path.isdir(override):
            raise FileNotFoundError(
                f"HETEROFOLD_{name.upper()}_PATH={override} is not a directory")
        return override
    if os.path.isdir(name):
        return os.path.abspath(name)
    from huggingface_hub import snapshot_download

    if name in MODELS:
        repo, revision = MODELS[name].repo, MODELS[name].revision
    else:
        repo, revision = name, None
    kwargs = dict(repo_id=repo, revision=revision,
                  allow_patterns=_ALLOW_PATTERNS)
    try:
        return snapshot_download(local_files_only=True, **kwargs)
    except Exception:
        if not download:
            raise
    return snapshot_download(
        token=os.environ.get("HF_TOKEN") or None, max_workers=4, **kwargs)
