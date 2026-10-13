from __future__ import annotations

import torch

from .alignment import exact_end_pairs, proportional_layers, stencil_taps
from .capture import capture_kv
from .maps import MAP_KIND, SOURCE_SPACE, TARGET_SPACE
from .modeling import kv_geometry

FIELDS = ("sX", "sY", "XtX", "XtY", "YtY")
MIN_ALIGNED_TOKENS = 8


class RoleStats:
    def __init__(self, source_width: int, target_width: int, layers: int,
                 device):
        self.sX = [torch.zeros(source_width, device=device) for _ in range(layers)]
        self.sY = [torch.zeros(target_width, device=device) for _ in range(layers)]
        self.XtX = [torch.zeros(source_width, source_width, device=device)
                    for _ in range(layers)]
        self.XtY = [torch.zeros(source_width, target_width, device=device)
                    for _ in range(layers)]
        self.YtY = [torch.zeros(target_width, target_width, device=device)
                    for _ in range(layers)]

    def add(self, layer: int, source: torch.Tensor, target: torch.Tensor) -> None:
        self.sX[layer].add_(source.sum(0))
        self.sY[layer].add_(target.sum(0))
        self.XtX[layer].add_(source.T @ source)
        self.XtY[layer].add_(source.T @ target)
        self.YtY[layer].add_(target.T @ target)

    def cpu(self) -> dict:
        return {field: [tensor.cpu() for tensor in getattr(self, field)]
                for field in FIELDS}


@torch.no_grad()
def collect_statistics(sender, receiver, texts: list[str], *, span: int = 1,
                       stride: int = 4, device="cuda:0", log=print,
                       tag: str = "") -> dict:
    sender_tokenizer, sender_model = sender
    receiver_tokenizer, receiver_model = receiver
    sender_layers, sender_heads, sender_dim = kv_geometry(sender_model)
    receiver_layers, receiver_heads, receiver_dim = kv_geometry(receiver_model)
    source_width = sender_heads * sender_dim
    target_width = receiver_heads * receiver_dim
    taps = stencil_taps(sender_layers, receiver_layers, span, stride)
    tap_layers = {layer for layer_taps in taps for layer in layer_taps}
    stencil_width = (2 * span + 1) * source_width
    roles = {role: RoleStats(stencil_width, target_width, receiver_layers,
                             device) for role in "kv"}
    aligned = 0
    for count, text in enumerate(texts, 1):
        sender_offsets, sender_k, sender_v = capture_kv(
            sender_tokenizer, sender_model, text, tap_layers, device=device)
        receiver_offsets, receiver_k, receiver_v = capture_kv(
            receiver_tokenizer, receiver_model, text, range(receiver_layers),
            device=device)
        pairs = exact_end_pairs(sender_offsets, receiver_offsets)
        if len(pairs) < MIN_ALIGNED_TOKENS:
            raise ValueError(
                f"calibration text {count - 1} has only {len(pairs)} aligned "
                "tokens")
        source_index = torch.tensor([pair[0] for pair in pairs], device=device)
        target_index = torch.tensor([pair[1] for pair in pairs], device=device)
        for layer in range(receiver_layers):
            for role, sender_feature, receiver_feature in (
                    ("k", sender_k, receiver_k), ("v", sender_v, receiver_v)):
                source = torch.cat(
                    [sender_feature[tap][source_index] for tap in taps[layer]],
                    dim=-1).float()
                target = receiver_feature[layer][target_index].float()
                roles[role].add(layer, source, target)
                del source, target
        aligned += len(pairs)
        if count % 50 == 0 or count == len(texts):
            log(f"[recolor-stats]{tag} {count}/{len(texts)} texts, "
                f"{aligned} aligned tokens")
        del sender_k, sender_v, receiver_k, receiver_v
    centers = proportional_layers(sender_layers, receiver_layers)
    return {
        "roles": {role: stats.cpu() for role, stats in roles.items()},
        "N": aligned, "texts": len(texts),
        "pi": centers, "taps": taps,
        "Ls": sender_layers, "Lr": receiver_layers,
        "source_kv_width": source_width, "kv_width": target_width,
        "stencil_source_width": stencil_width,
        "span": span, "stride": stride,
    }


def balance_corpora(per_corpus: dict[str, dict]) -> dict:
    labels = sorted(per_corpus)
    total = sum(int(per_corpus[label]["N"]) for label in labels)
    target_mass = float(total) / len(labels)
    scales = {label: target_mass / float(per_corpus[label]["N"])
              for label in labels}
    reference = per_corpus[labels[0]]
    layers = int(reference["Lr"])
    combined = {key: value for key, value in reference.items()
                if key not in ("roles", "N", "texts")}
    combined["roles"] = {}
    for role in "kv":
        combined["roles"][role] = {}
        for field in FIELDS:
            tensors = []
            for layer in range(layers):
                value = torch.zeros_like(reference["roles"][role][field][layer])
                for label in labels:
                    value.add_(per_corpus[label]["roles"][role][field][layer],
                               alpha=scales[label])
                tensors.append(value)
            combined["roles"][role][field] = tensors
    combined["N"] = total
    combined["texts"] = sum(int(per_corpus[label]["texts"]) for label in labels)
    combined["source_aligned_token_count"] = {
        label: int(per_corpus[label]["N"]) for label in labels}
    combined["moment_scales"] = scales
    combined["moment_weighting"] = (
        "equal_effective_aligned_token_mass_per_corpus"
        if len(labels) > 1 else "natural_aligned_token_mass")
    return combined


def _sym_sqrt_pair(matrix: torch.Tensor, eps: float = 1e-3):
    matrix = matrix + eps * torch.eye(
        matrix.shape[0], device=matrix.device) * torch.diag(matrix).mean()
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(1e-6)
    inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    sqrt = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
    return inverse_sqrt, sqrt


def fit_affine(sX, sY, XtX, XtY, YtY, count: float, *, device="cuda:0"):
    XtX, XtY, YtY = XtX.to(device), XtY.to(device), YtY.to(device)
    sX, sY = sX.to(device), sY.to(device)
    width = XtX.shape[0]
    rms = float((torch.trace(XtX) / (count * width)).clamp_min(1e-12).sqrt())
    mean_x, mean_y = sX / count, sY / count
    mean_z = mean_x / rms
    cross = (XtY - count * torch.outer(mean_x, mean_y)) / rms
    cov_z = (XtX - count * torch.outer(mean_x, mean_x)) / (rms * rms * count)
    cov_y = (YtY - count * torch.outer(mean_y, mean_y)) / count
    z_inverse_sqrt, _ = _sym_sqrt_pair(cov_z)
    y_inverse_sqrt, y_sqrt = _sym_sqrt_pair(cov_y)
    whitened = z_inverse_sqrt @ (cross / count) @ y_inverse_sqrt
    left, _, right = torch.linalg.svd(whitened, full_matrices=False)
    weight = z_inverse_sqrt @ (left @ right) @ y_sqrt
    bias = mean_y - mean_z @ weight
    return weight.cpu(), bias.cpu(), rms


def fit_map(stats: dict, *, device="cuda:0", log=print) -> dict:
    layers = int(stats["Lr"])
    count = float(stats["N"])
    fitted = {}
    for role in "kv":
        weights, biases, scales = [], [], []
        for layer in range(layers):
            role_stats = stats["roles"][role]
            weight, bias, rms = fit_affine(
                *(role_stats[field][layer] for field in FIELDS), count,
                device=device)
            weights.append(weight)
            biases.append(bias)
            scales.append(rms)
        fitted[role] = (weights, biases, scales)
        log(f"[recolor-fit] role={role} layers={layers}")
    return {
        "A_k": fitted["k"][0], "b_k": fitted["k"][1], "rms_k": fitted["k"][2],
        "A_v": fitted["v"][0], "b_v": fitted["v"][1], "rms_v": fitted["v"][2],
        "pi": stats["pi"], "pi_k": stats["pi"], "pi_v": stats["pi"],
        "Ls": stats["Ls"], "Lr": layers,
        "ds": 2 * stats["stencil_source_width"], "dr": 2 * stats["kv_width"],
        "source_kv_width": stats["source_kv_width"],
        "kv_width": stats["kv_width"],
        "source_space": SOURCE_SPACE, "target_space": TARGET_SPACE,
        "sender_context": "plain", "receiver_context": "plain",
        "native_sender": False,
        "map": MAP_KIND,
        "depth_stencil": {
            "span": stats["span"], "stride": stats["stride"], "per_tap_rms": 0,
            "taps_k": stats["taps"], "taps_v": stats["taps"],
        },
        "calibration": {
            "aligned_token_count": int(stats["N"]),
            "text_count": int(stats["texts"]),
            "moment_weighting": stats.get("moment_weighting"),
            "source_aligned_token_count": stats.get(
                "source_aligned_token_count"),
            "moment_scales": stats.get("moment_scales"),
        },
    }
