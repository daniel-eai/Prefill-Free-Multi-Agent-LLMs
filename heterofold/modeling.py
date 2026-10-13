from __future__ import annotations

import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BatchEncoding,
)


class MistralOffsetTokenizer:
    _CONTROL_STRINGS = ("[/SYSTEM_PROMPT]", "[SYSTEM_PROMPT]", "[/INST]",
                        "[INST]", "</s>", "<s>")

    def __init__(self, backend):
        self.backend = backend
        self.tekken = backend.tokenizer.instruct_tokenizer.tokenizer

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _control_id(self, marker: str) -> int:
        if marker == "<s>":
            return int(self.tekken.bos_id)
        if marker == "</s>":
            return int(self.tekken.eos_id)
        return int(self.tekken.get_special_token(marker))

    def _split_controls(self, text: str):
        position = 0
        while position < len(text):
            best = None
            for marker in self._CONTROL_STRINGS:
                found = text.find(marker, position)
                if found != -1 and (best is None or found < best[0]):
                    best = (found, marker)
            if best is None:
                yield text[position:], None
                return
            start, marker = best
            if start > position:
                yield text[position:start], None
            yield marker, self._control_id(marker)
            position = start + len(marker)

    @staticmethod
    def _byte_to_character(text: str):
        raw = text.encode("utf-8")
        result = [0] * (len(raw) + 1)
        byte_position = 0
        for character_position, character in enumerate(text):
            width = len(character.encode("utf-8"))
            for position in range(byte_position, byte_position + width):
                result[position] = character_position
            byte_position += width
            result[byte_position] = character_position + 1
        return raw, result

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, return_tensors=None, **kwargs):
        if not isinstance(text, str) or add_special_tokens or kwargs:
            if return_offsets_mapping:
                raise ValueError(
                    "Mistral offset wrapper supports raw single strings only")
            return self.backend(text, add_special_tokens=add_special_tokens,
                                return_tensors=return_tensors, **kwargs)
        input_ids, pieces = [], []
        for segment, control in self._split_controls(text):
            if control is not None:
                input_ids.append(control)
                pieces.append(segment.encode("utf-8"))
                continue
            ids = self.tekken.encode(segment, bos=False, eos=False)
            input_ids.extend(ids)
            pieces.extend(self.tekken.id_to_byte_piece(token) for token in ids)
        result = {"input_ids": input_ids,
                  "attention_mask": [1] * len(input_ids)}
        if return_offsets_mapping:
            raw, byte_to_character = self._byte_to_character(text)
            if b"".join(pieces) != raw:
                raise ValueError(
                    "mistral-common byte pieces do not reconstruct the input")
            byte_position = 0
            offsets = []
            for piece in pieces:
                start = byte_position
                byte_position += len(piece)
                offsets.append((byte_to_character[start],
                                byte_to_character[byte_position]))
            result["offset_mapping"] = offsets
        return BatchEncoding(result, tensor_type=return_tensors,
                             prepend_batch_axis=return_tensors is not None)


def load_tokenizer(path: str):
    tekken = os.path.join(str(path), "tekken.json")
    if os.path.isfile(tekken):
        from transformers import MistralCommonBackend

        return MistralOffsetTokenizer(MistralCommonBackend(tokenizer_path=tekken))
    return AutoTokenizer.from_pretrained(path)


def load_model(path: str, device: str = "cuda:0", dtype=torch.bfloat16):
    tokenizer = load_tokenizer(path)
    config = AutoConfig.from_pretrained(path)
    auto_model = (AutoModelForImageTextToText
                  if config.model_type == "mistral3" else AutoModelForCausalLM)
    model = auto_model.from_pretrained(path, dtype=dtype).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def decoder(model):
    base = model.model
    return base.language_model if hasattr(base, "language_model") else base


def layers_of(model):
    return decoder(model).layers


def text_config(model):
    config = model.config
    return getattr(config, "text_config", config)


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def kv_geometry(model) -> tuple[int, int, int]:
    config = text_config(model)
    head_dim = int(getattr(
        config, "head_dim",
        config.hidden_size // config.num_attention_heads))
    return (int(config.num_hidden_layers), int(config.num_key_value_heads),
            head_dim)


def set_attention_implementation(model, implementation: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(implementation)
        return
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config._attn_implementation = implementation


def chat_affixes(tokenizer, thinking: bool = False) -> tuple[str, str]:
    marker = "__HETEROFOLD_USER_CONTENT_7F3A91C2__"
    messages = [{"role": "user", "content": marker}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking)
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(marker) != 1:
        raise ValueError("chat template did not preserve the content marker")
    prefix, suffix = rendered.split(marker, 1)
    if not prefix or not suffix:
        raise ValueError("chat template produced an empty prefix or suffix")
    return prefix, suffix
