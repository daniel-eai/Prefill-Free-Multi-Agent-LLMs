from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .alignment import receiver_to_sender_indices
from .attention import (
    apply_native_query_scale,
    apply_rope,
    attention_probabilities,
    causal_mask,
    output_projection,
    rotary_cos_sin,
)
from .capture import capture_kv, capture_native_qkv
from .maps import atomic_save, materialize, sender_layers
from .modeling import chat_affixes, layers_of, text_config

ATTENTION_VERSION = "explicit_axes_and_native_query_scale_20260909"


@dataclass
class CalibrationExample:
    prefix_length: int
    content_length: int
    probe_positions: torch.Tensor
    base_key: torch.Tensor
    base_value: torch.Tensor
    native_query: torch.Tensor
    native_attention: torch.Tensor
    native_output: torch.Tensor
    native_value: torch.Tensor


def question_token_positions(offsets, start: int, end: int) -> list[int]:
    result = [index for index, (token_start, token_end) in enumerate(offsets)
              if token_end > token_start and token_end > start
              and token_start < end]
    if not result:
        raise ValueError("literal question span selected no receiver tokens")
    return result


class ExampleBuilder:
    def __init__(self, sender, receiver, base_map: dict, device):
        self.sender_tokenizer, self.sender_model = sender
        self.receiver_tokenizer, self.receiver_model = receiver
        self.device = device
        self.mapping = materialize(base_map, device=device, dtype=torch.float32)
        self.source_layers = sender_layers(base_map)
        prefix, _ = chat_affixes(self.receiver_tokenizer, thinking=False)
        self.prefix_ids = torch.tensor(
            self.receiver_tokenizer(prefix, add_special_tokens=False).input_ids,
            dtype=torch.long)

    @torch.no_grad()
    def recolor_outputs(self, prompt: str, receiver_offsets):
        mapping, device = self.mapping, self.device
        sender_offsets, keys, values = capture_kv(
            self.sender_tokenizer, self.sender_model, prompt,
            self.source_layers, device=device, dtype=torch.bfloat16)
        source = torch.tensor(
            receiver_to_sender_indices(sender_offsets, receiver_offsets),
            device=device, dtype=torch.long)
        stencil = mapping["depth_stencil"]
        base_key, base_value = [], []
        for layer in range(int(mapping["Lr"])):
            source_key = torch.cat(
                [keys[int(tap)].float()[source]
                 for tap in stencil["taps_k"][layer]], dim=-1)
            source_value = torch.cat(
                [values[int(tap)].float()[source]
                 for tap in stencil["taps_v"][layer]], dim=-1)
            key = (source_key / float(mapping["rms_k"][layer])
                   @ mapping["A_k"][layer] + mapping["b_k"][layer])
            value = (source_value / float(mapping["rms_v"][layer])
                     @ mapping["A_v"][layer] + mapping["b_v"][layer])
            base_key.append(key.to(torch.bfloat16))
            base_value.append(value.to(torch.bfloat16))
            del source_key, source_value, key, value
        return torch.stack(base_key), torch.stack(base_value)

    @torch.no_grad()
    def native_targets(self, content_ids: torch.Tensor,
                       probe_positions: torch.Tensor):
        model, device = self.receiver_model, self.device
        config = text_config(model)
        query_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(
            config, "head_dim", config.hidden_size // query_heads))
        groups = query_heads // kv_heads
        blocks = layers_of(model)
        full_ids = torch.cat((self.prefix_ids, content_ids)).to(
            device).unsqueeze(0)
        total_length = int(full_ids.shape[1])
        captures = capture_native_qkv(model, full_ids, probe_positions)
        positions = torch.arange(total_length, device=device, dtype=torch.long)
        queries, attentions, outputs, native_values = [], [], [], []
        for layer, captured in enumerate(captures):
            attention = blocks[layer].self_attn
            cosine, sine = rotary_cos_sin(model, positions, layer, model.dtype)
            mask = causal_mask(attention, probe_positions, total_length)
            query = apply_rope(
                captured["q"], cosine.index_select(0, probe_positions),
                sine.index_select(0, probe_positions))
            query = apply_native_query_scale(attention, query, probe_positions)
            key = apply_rope(captured["k"], cosine, sine)
            probability = attention_probabilities(
                attention, query, key, mask, groups)
            repeated_value = captured["v"].repeat_interleave(groups, dim=1)
            context = torch.einsum("hpt,thd->phd", probability, repeated_value)
            context = context.reshape(
                int(probe_positions.numel()), query_heads * head_dim)
            post_wo = output_projection(attention)(context)
            queries.append(query.to(torch.bfloat16))
            attentions.append(probability.to(torch.bfloat16))
            outputs.append(post_wo.to(torch.bfloat16))
            native_values.append(captured["v"].to(torch.bfloat16))
            del query, key, probability, repeated_value, context, post_wo
        return (torch.stack(queries), torch.stack(attentions),
                torch.stack(outputs), torch.stack(native_values))

    @torch.no_grad()
    def build(self, row: dict) -> CalibrationExample:
        prompt = row["prompt"]
        encoding = self.receiver_tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        probes = question_token_positions(
            offsets, int(row["question_char_start"]),
            int(row["question_char_end"]))
        prefix_length = int(self.prefix_ids.numel())
        probe_positions = torch.tensor(
            probes, dtype=torch.long, device=self.device) + prefix_length
        base_key, base_value = self.recolor_outputs(prompt, offsets)
        content_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        query, attention, output, value = self.native_targets(
            content_ids, probe_positions)
        return CalibrationExample(
            prefix_length=prefix_length,
            content_length=int(content_ids.numel()),
            probe_positions=probe_positions,
            base_key=base_key, base_value=base_value,
            native_query=query, native_attention=attention,
            native_output=output, native_value=value)


class LayerCorrection(torch.nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.width, self.rank = width, rank
        for role in "kv":
            left = torch.nn.Parameter(torch.empty(width, rank))
            right = torch.nn.Parameter(torch.zeros(rank, width))
            torch.nn.init.normal_(left, std=0.02)
            setattr(self, f"left_{role}", left)
            setattr(self, f"right_{role}", right)

    def _correct(self, value: torch.Tensor, role: str) -> torch.Tensor:
        left = getattr(self, f"left_{role}")
        right = getattr(self, f"right_{role}")
        return value + (value @ left) @ right

    def key(self, value: torch.Tensor) -> torch.Tensor:
        return self._correct(value, "k")

    def value(self, value: torch.Tensor) -> torch.Tensor:
        return self._correct(value, "v")

    def factors(self, role: str) -> tuple[torch.Tensor, torch.Tensor]:
        return (getattr(self, f"left_{role}").detach().cpu().float(),
                getattr(self, f"right_{role}").detach().cpu().float())


def prefix_cache(model, prefix_ids: torch.Tensor, device):
    with torch.no_grad():
        result = model(prefix_ids.to(device).unsqueeze(0), use_cache=True,
                       logits_to_keep=1)
    cache = result.past_key_values
    keys, values = [], []
    for layer in range(len(layers_of(model))):
        if hasattr(cache, "layers"):
            key, value = cache.layers[layer].keys, cache.layers[layer].values
        else:
            key, value = cache.key_cache[layer], cache.value_cache[layer]
        keys.append(key.detach()[0].transpose(0, 1).to(model.dtype))
        values.append(value.detach()[0].transpose(0, 1).to(model.dtype))
    return keys, values


def example_objective(model, corrections, example: CalibrationExample,
                      prefix_keys, prefix_values, *, lambda_k_output: float,
                      backward: bool) -> dict[str, float]:
    blocks = layers_of(model)
    config = text_config(model)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim",
                           config.hidden_size // query_heads))
    groups = query_heads // kv_heads
    device = example.base_key.device
    content_length = example.content_length
    full_length = example.prefix_length + content_length
    probes = example.probe_positions
    positions = torch.arange(
        example.prefix_length, full_length, device=device, dtype=torch.long)
    total_kl = total_k_output = total_v_output = 0.0
    for layer, correction in enumerate(corrections):
        base_key = example.base_key[layer].float()
        base_value = example.base_value[layer].float()
        key = correction.key(base_key).reshape(
            content_length, kv_heads, head_dim).to(model.dtype)
        value = correction.value(base_value).reshape(
            content_length, kv_heads, head_dim).to(model.dtype)
        attention = blocks[layer].self_attn
        cosine, sine = rotary_cos_sin(model, positions, layer, model.dtype)
        mask = causal_mask(attention, probes, full_length)
        if getattr(attention, "k_norm", None) is not None:
            key = attention.k_norm(key)
        key = apply_rope(key, cosine, sine)
        all_key = torch.cat((prefix_keys[layer], key), dim=0)
        all_value = torch.cat((prefix_values[layer], value), dim=0)
        query = example.native_query[layer]
        probability = attention_probabilities(
            attention, query, all_key, mask, groups)
        log_probability = probability.float().clamp_min(1e-20).log()
        teacher = example.native_attention[layer].float().clamp_min(0)
        teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-20)
        routing_kl = (teacher * (teacher.clamp_min(1e-12).log()
                                 - log_probability)).sum(-1).mean()

        native_value = example.native_value[layer].float().repeat_interleave(
            groups, dim=1)
        key_context_error = torch.einsum(
            "hpt,thd->phd", probability.float() - teacher, native_value)
        projection = output_projection(attention)
        wo_by_head = projection.weight.float().reshape(
            config.hidden_size, query_heads, head_dim).permute(1, 0, 2)
        key_output = torch.einsum(
            "phd,hmd->phm", key_context_error, wo_by_head).square().mean()

        context = torch.einsum(
            "hpt,thd->phd", probability.detach(),
            all_value.repeat_interleave(groups, dim=1))
        context = context.reshape(query.shape[0], query_heads * head_dim)
        mapped_output = projection(context).float()
        native_output = example.native_output[layer].float()
        value_output = ((mapped_output - native_output).square().sum()
                        / native_output.square().sum().clamp_min(1e-8))

        layer_loss = routing_kl + lambda_k_output * key_output + value_output
        if backward:
            (layer_loss / len(corrections)).backward()
        total_kl += float(routing_kl.detach())
        total_k_output += float(key_output.detach())
        total_v_output += float(value_output.detach())
        del key, value, all_key, all_value, probability, teacher, layer_loss
    count = len(corrections)
    metrics = {"routing_kl": total_kl / count,
               "k_post_wo_mse": total_k_output / count,
               "post_wo_nmse": total_v_output / count}
    metrics["loss"] = (metrics["routing_kl"]
                       + lambda_k_output * metrics["k_post_wo_mse"]
                       + metrics["post_wo_nmse"])
    return metrics


METRIC_NAMES = ("loss", "routing_kl", "k_post_wo_mse", "post_wo_nmse")


def _mean(rows: list[dict]) -> dict:
    return {name: sum(row[name] for row in rows) / max(len(rows), 1)
            for name in METRIC_NAMES}


def fold(base_map: dict, corrections, metadata: dict) -> dict:
    result = copy.deepcopy(base_map)
    diagnostics = []
    with torch.no_grad():
        for layer, correction in enumerate(corrections):
            record = {"layer": layer}
            for role in "kv":
                left, right = correction.factors(role)
                weight = base_map[f"A_{role}"][layer].float()
                bias = base_map[f"b_{role}"][layer].float()
                result[f"A_{role}"][layer] = (
                    weight + (weight @ left) @ right).to(
                        base_map[f"A_{role}"][layer].dtype)
                result[f"b_{role}"][layer] = (
                    bias + (bias @ left) @ right).to(
                        base_map[f"b_{role}"][layer].dtype)
                record[f"{role}_relative_update"] = float(
                    (left @ right).norm() / (left.shape[0] ** 0.5))
            diagnostics.append(record)
    result["functional_variant"] = f"output_aware_rank{metadata['rank']}"
    result["functional_correction"] = {
        **metadata,
        "structure": "folded_kv_separated_low_rank_residual",
        "inference_extra_modules": 0,
        "diagnostics": diagnostics,
    }
    return result


def calibrate(sender, receiver, base_map: dict, rows: list[dict], *,
              device, out_path: Path, checkpoint_path: Path,
              rank: int = 16, epochs: int = 4, lr: float = 3e-4,
              weight_decay: float = 1e-4, seed: int = 20260825,
              grad_clip: float = 1.0, lambda_k_output: float = 1.0,
              grad_accum: int = 8, log=print) -> dict:
    _, model = receiver
    torch.manual_seed(seed)
    random.seed(seed)
    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    validation_indices = [i for i, row in enumerate(rows)
                          if row["split"] == "validation"]
    if not train_indices or not validation_indices:
        raise ValueError("calibration needs train and validation prompts")
    builder = ExampleBuilder(sender, receiver, base_map, device)
    prefix_keys, prefix_values = prefix_cache(model, builder.prefix_ids, device)
    width = int(base_map["kv_width"])
    corrections = torch.nn.ModuleList([
        LayerCorrection(width, rank).to(device).float()
        for _ in range(int(base_map["Lr"]))])
    parameters = list(corrections.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    signature = dict(rank=rank, epochs=epochs, lr=lr, seed=seed,
                     grad_clip=grad_clip, lambda_k_output=lambda_k_output,
                     grad_accum=grad_accum, train=len(train_indices),
                     validation=len(validation_indices))

    def evaluate() -> dict:
        rows_ = []
        with torch.no_grad():
            for index in validation_indices:
                rows_.append(example_objective(
                    model, corrections, builder.build(rows[index]), prefix_keys,
                    prefix_values, lambda_k_output=lambda_k_output,
                    backward=False))
        return _mean(rows_)

    def state() -> dict:
        return {key: value.detach().cpu().clone()
                for key, value in corrections.state_dict().items()}

    start_epoch, history, best_loss, best_state = 0, [], None, None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError(
                f"{checkpoint_path} was written with different settings; "
                "remove it or use another output directory")
        corrections.load_state_dict(checkpoint["corrections"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["completed_epochs"])
        history = list(checkpoint["history"])
        baseline = checkpoint["baseline_validation"]
        best_loss = checkpoint["best_validation_loss"]
        best_state = checkpoint["best_corrections"]
        log(f"[calibrate] resumed after epoch {start_epoch}")
    else:
        started = time.time()
        baseline = evaluate()
        log(f"[calibrate] recolor baseline (held-out) {json.dumps(baseline)} "
            f"[{time.time() - started:.0f}s]")

    for epoch in range(start_epoch, epochs):
        order = list(train_indices)
        random.Random(seed + epoch).shuffle(order)
        groups = [order[i:i + grad_accum]
                  for i in range(0, len(order), grad_accum)]
        corrections.train()
        running, started = [], time.time()
        for step, group in enumerate(groups, 1):
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                with torch.enable_grad():
                    metrics = example_objective(
                        model, corrections, builder.build(rows[index]),
                        prefix_keys,
                        prefix_values, lambda_k_output=lambda_k_output,
                        backward=True)
                running.append(metrics)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(len(group))
            gradient = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            if step % 10 == 0 or step == len(groups):
                log(f"[calibrate] epoch={epoch + 1}/{epochs} "
                    f"step={step}/{len(groups)} loss={metrics['loss']:.6f} "
                    f"routing={metrics['routing_kl']:.6f} "
                    f"k_output={metrics['k_post_wo_mse']:.6f} "
                    f"output={metrics['post_wo_nmse']:.6f} "
                    f"grad={float(gradient):.6f} "
                    f"[{time.time() - started:.0f}s]")
        corrections.eval()
        entry = {"epoch": epoch + 1, "train": _mean(running),
                 "validation": evaluate()}
        history.append(entry)
        log(f"[calibrate] {json.dumps(entry)}")
        if best_loss is None or entry["validation"]["loss"] < best_loss:
            best_loss = entry["validation"]["loss"]
            best_state = state()
        atomic_save({
            "signature": signature, "completed_epochs": epoch + 1,
            "baseline_validation": baseline, "best_validation_loss": best_loss,
            "corrections": state(), "best_corrections": best_state,
            "optimizer": optimizer.state_dict(), "history": history,
        }, checkpoint_path)

    corrections.load_state_dict(best_state)
    metadata = {
        "receiver_attention_version": ATTENTION_VERSION,
        "calibration_train_examples": len(train_indices),
        "calibration_validation_examples": len(validation_indices),
        "probe": "all receiver tokens overlapping the literal question string",
        "continuation_tokens_used": 0,
        "rank": rank, "epochs": epochs, "lr": lr, "seed": seed,
        "grad_accum": grad_accum, "lambda_k_output": lambda_k_output,
        "k_objective": ("native attention KL + lambda * key-induced "
                        "per-query-head post-W_O MSE (native values fixed)"),
        "v_objective": "post-W_O attention-output NMSE with detached routing",
        "baseline_validation": baseline,
        "best_validation_loss": best_loss,
        "history": history,
    }
    result = fold(base_map, corrections, metadata)
    atomic_save(result, out_path)
    return result
