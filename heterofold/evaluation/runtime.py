from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..maps import describe, load_map
from ..modeling import chat_affixes, input_device, load_model
from ..registry import parse_pair, resolve_model
from ..transfer import build_receiver_cache, capture_sender


def default_devices() -> tuple[str, str]:
    return "cuda:0", ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")


class Runtime:
    def __init__(self, *, receiver: str, sender: str | None = None,
                 map_path: str | None = None, receiver_device: str | None = None,
                 sender_device: str | None = None):
        auto_receiver, auto_sender = default_devices()
        self.receiver_device = receiver_device or auto_receiver
        self.sender_device = sender_device or auto_sender
        self.receiver_name, self.sender_name = receiver, sender
        self.receiver_path = resolve_model(receiver)
        self.tokenizer, self.model = load_model(
            self.receiver_path, self.receiver_device)
        self.prefix, self.assistant = chat_affixes(self.tokenizer)
        self.mode = "native" if map_path is None else "transfer"
        self.mapping = self.sender_tokenizer = self.sender_model = None
        self.map_path = None
        if map_path is not None:
            if sender is None:
                raise ValueError("transfer evaluation needs a sender model")
            self.map_path = str(Path(map_path).resolve())
            self.mapping = load_map(map_path, device=self.receiver_device,
                                    dtype=torch.bfloat16)
            self.sender_path = resolve_model(sender)
            self.sender_tokenizer, self.sender_model = load_model(
                self.sender_path, self.sender_device)

    @classmethod
    def from_args(cls, args) -> "Runtime":
        sender = receiver = None
        if args.pair:
            sender, receiver = parse_pair(args.pair)
        receiver = args.receiver or receiver
        sender = args.sender or sender
        if receiver is None:
            raise SystemExit("give --pair or --receiver")
        if args.native:
            return cls(receiver=receiver, receiver_device=args.receiver_device)
        if not args.map:
            raise SystemExit("transfer evaluation needs --map (or use --native)")
        return cls(receiver=receiver, sender=sender, map_path=args.map,
                   receiver_device=args.receiver_device,
                   sender_device=args.sender_device)

    @torch.inference_mode()
    def transfer_cache(self, prompt: str):
        offsets, keys, values = capture_sender(
            self.sender_tokenizer, self.sender_model, self.mapping, prompt,
            device=self.receiver_device)
        cache, length = build_receiver_cache(
            self.tokenizer, self.model, self.mapping, offsets, keys, values,
            prompt, self.prefix)
        return cache, length, len(offsets)

    def native_ids(self, prompt: str, suffix: str) -> torch.Tensor:
        return self.tokenizer(
            self.prefix + prompt + suffix, add_special_tokens=False,
            return_tensors="pt").input_ids.to(input_device(self.model))

    def suffix_ids(self, suffix: str) -> torch.Tensor:
        return self.tokenizer(
            suffix, add_special_tokens=False,
            return_tensors="pt").input_ids.to(input_device(self.model))

    def describe(self) -> dict:
        info = {"mode": self.mode, "receiver": self.receiver_name,
                "receiver_path": self.receiver_path}
        if self.mode == "transfer":
            info.update(sender=self.sender_name, sender_path=self.sender_path,
                        map_path=self.map_path, map=describe(self.mapping))
        return info


def candidate_ids(tokenizer, candidate: str) -> list[int]:
    ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"empty candidate tokenization: {candidate!r}")
    return ids


def _candidate_log_likelihood(logits, target, normalize: bool) -> float:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    value = log_probs.gather(-1, target.unsqueeze(-1)).sum()
    if normalize:
        value = value / target.shape[1]
    return float(value.cpu())


@torch.inference_mode()
def native_candidate_scores(runtime: Runtime, prompt: str, candidates,
                            normalize: bool, suffix: str):
    context = runtime.native_ids(prompt, suffix)
    device = context.device
    scores = []
    for candidate in candidates:
        target = torch.tensor(candidate_ids(runtime.tokenizer, candidate),
                              device=device, dtype=torch.long).unsqueeze(0)
        output = runtime.model(torch.cat((context, target), dim=1),
                               use_cache=False)
        start = context.shape[1] - 1
        scores.append(_candidate_log_likelihood(
            output.logits[:, start:start + target.shape[1]], target, normalize))
    return scores, int(context.shape[1])


@torch.inference_mode()
def transfer_candidate_scores(runtime: Runtime, cache, past_length: int,
                              candidates, normalize: bool, suffix: str):
    suffix_ids = runtime.suffix_ids(suffix)
    device = suffix_ids.device
    scores = []
    for candidate in candidates:
        target = torch.tensor(candidate_ids(runtime.tokenizer, candidate),
                              device=device, dtype=torch.long).unsqueeze(0)
        input_ids = torch.cat((suffix_ids, target), dim=1)
        total = past_length + input_ids.shape[1]
        output = runtime.model(
            input_ids, past_key_values=cache,
            attention_mask=torch.ones(1, total, device=device, dtype=torch.long),
            position_ids=torch.arange(past_length, total,
                                      device=device).unsqueeze(0),
            use_cache=False)
        start = suffix_ids.shape[1] - 1
        scores.append(_candidate_log_likelihood(
            output.logits[:, start:start + target.shape[1]], target, normalize))
        cache.crop(past_length)
    return scores, int(suffix_ids.shape[1])


def candidate_scores(runtime: Runtime, prompt: str, candidates,
                     normalize: bool, suffix: str) -> dict:
    if runtime.mode == "native":
        scores, tokens = native_candidate_scores(
            runtime, prompt, candidates, normalize, suffix)
        return {"scores": scores, "native_prefill_tokens": tokens,
                "transfer_cache_tokens": 0, "receiver_suffix_tokens": 0}
    cache, past_length, _ = runtime.transfer_cache(prompt)
    scores, suffix_tokens = transfer_candidate_scores(
        runtime, cache, past_length, candidates, normalize, suffix)
    del cache
    return {"scores": scores, "native_prefill_tokens": 0,
            "transfer_cache_tokens": past_length,
            "receiver_suffix_tokens": suffix_tokens}


@torch.inference_mode()
def generate(runtime: Runtime, input_ids: torch.Tensor, max_new_tokens: int,
             *, past_key_values=None, past_length: int = 0,
             temperature: float = 0.0, top_p: float = 0.95, top_k: int = 20,
             seed: int = 0) -> tuple[str, int]:
    torch.manual_seed(seed)
    device = input_ids.device
    total = past_length + int(input_ids.shape[1])
    kwargs = {
        "attention_mask": torch.ones(1, total, device=device, dtype=torch.long),
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": runtime.tokenizer.pad_token_id,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
        kwargs["position_ids"] = torch.arange(
            past_length, total, device=device).unsqueeze(0)
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    output = runtime.model.generate(input_ids, **kwargs)
    generated = output[0, input_ids.shape[1]:]
    return (runtime.tokenizer.decode(generated, skip_special_tokens=False),
            int(generated.numel()))


@torch.inference_mode()
def greedy_decode(runtime: Runtime, input_ids: torch.Tensor,
                  max_new_tokens: int) -> tuple[str, int]:
    model, tokenizer = runtime.model, runtime.tokenizer
    device = input_ids.device
    total = int(input_ids.shape[1])
    output = model(
        input_ids,
        attention_mask=torch.ones(1, total, device=device, dtype=torch.long),
        position_ids=torch.arange(total, device=device).unsqueeze(0),
        use_cache=True)
    cache = output.past_key_values
    stop_ids = {value for value in (
        tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if value is not None and value >= 0}
    generated = []
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    for _ in range(max_new_tokens):
        token = int(next_token.item())
        if token in stop_ids:
            break
        generated.append(token)
        total += 1
        output = model(
            next_token, past_key_values=cache,
            attention_mask=torch.ones(1, total, device=device, dtype=torch.long),
            position_ids=torch.tensor([[total - 1]], device=device),
            use_cache=True)
        cache = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    return tokenizer.decode(generated, skip_special_tokens=False), len(generated)


def terminal_markers(runtime: Runtime) -> tuple[str, ...]:
    tokenizer, model = runtime.tokenizer, runtime.model
    token_ids = []
    for source in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(getattr(model, "generation_config", None),
                    "eos_token_id", None),
            getattr(getattr(model, "config", None), "eos_token_id", None)):
        values = source if isinstance(source, (list, tuple)) else [source]
        for token_id in values:
            if isinstance(token_id, int) and token_id not in token_ids:
                token_ids.append(token_id)
    markers = []
    eos_token = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_token, str) and eos_token:
        markers.append(eos_token)
    for token_id in token_ids:
        marker = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(marker, str) and marker and marker not in markers:
            markers.append(marker)
    if "<|im_end|>" not in markers:
        markers.append("<|im_end|>")
    return tuple(sorted(markers, key=len, reverse=True))


def strip_terminal_markers(text: str, markers: tuple[str, ...]) -> str:
    import re

    text = text.rstrip()
    while text:
        marker = next((candidate for candidate in markers
                       if candidate and text.endswith(candidate)), None)
        if marker is None:
            break
        text = text[:-len(marker)].rstrip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def shard_range(total: int, shard: str | None, start: int | None,
                end: int | None) -> tuple[int, int]:
    if shard:
        index, count = (int(value) for value in shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit("--shard must be i/n with 0 <= i < n")
        quotient, remainder = divmod(total, count)
        begin = index * quotient + min(index, remainder)
        return begin, begin + quotient + int(index < remainder)
    begin = 0 if start is None else start
    stop = total if end is None else min(end, total)
    if not 0 <= begin <= stop:
        raise SystemExit("invalid --start/--end range")
    return begin, stop


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    os.replace(temporary, path)


class Progress:
    def __init__(self, tag: str, total: int, every: int = 20):
        self.tag, self.total, self.every = tag, total, every
        self.started, self.count, self.score = time.time(), 0, 0.0

    def update(self, score: float) -> None:
        self.count += 1
        self.score += score
        if self.count % self.every == 0 or self.count == self.total:
            elapsed = time.time() - self.started
            print(f"[{self.tag}] {self.count}/{self.total} "
                  f"mean={100 * self.score / self.count:.2f} "
                  f"{elapsed / self.count:.2f}s/item", flush=True)
