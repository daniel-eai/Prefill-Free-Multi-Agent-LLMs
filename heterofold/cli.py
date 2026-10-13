from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

print = functools.partial(print, flush=True)


def _add_device_arguments(parser) -> None:
    parser.add_argument(
        "--receiver-device", default=None,
        help="default: cuda:0")
    parser.add_argument(
        "--sender-device", default=None,
        help="default: cuda:1 if a second GPU is visible, else cuda:0")


def _data_dir_default() -> str:
    from .data.benchmarks import default_data_dir

    return str(default_data_dir())


def prepare_data(args) -> None:
    from .data import benchmarks, calibration_corpus

    root = Path(args.data_dir)
    if not args.skip_benchmarks:
        benchmarks.prepare_all(root, print)
    else:
        benchmarks.prepare_longbench(root, print)
    manifest = root / "calibration" / "openr1_hotpot.json"
    if manifest.exists() and not args.overwrite:
        print(f"[calibration-data] {manifest} exists")
        return
    calibration_corpus.build_manifest(
        manifest, qasper=root / "longbench/qasper.jsonl",
        hotpot=root / "longbench/hotpotqa.jsonl",
        sources_dir=root / "calibration/sources", log=print)


def download_models(args) -> None:
    from .registry import MODELS, resolve_model

    for name in args.models:
        path = resolve_model(name)
        label = MODELS[name].repo if name in MODELS else name
        print(f"[models] {name}: {label} -> {path}")


def calibrate(args) -> None:
    import torch

    from . import calibration, recolor
    from .data.calibration_corpus import load_manifest
    from .evaluation.runtime import default_devices
    from .maps import atomic_save
    from .modeling import load_model
    from .registry import parse_pair, resolve_model

    sender_name, receiver_name = parse_pair(args.pair)
    auto_receiver, auto_sender = default_devices()
    receiver_device = args.receiver_device or auto_receiver
    sender_device = args.sender_device or auto_sender
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "heterofold.pt"
    if final_path.exists():
        print(f"[calibrate] {final_path} exists; nothing to do")
        return
    manifest_path = Path(args.manifest or (
        Path(args.data_dir) / "calibration" / "openr1_hotpot.json"))
    manifest = load_manifest(manifest_path)
    rows = manifest["rows"]
    train_rows = [row for row in rows if row["split"] == "train"]
    print(f"[calibrate] {args.pair}: {len(train_rows)} train / "
          f"{len(rows) - len(train_rows)} held-out prompts; "
          f"sender on {sender_device}, receiver on {receiver_device}")

    sender = load_model(resolve_model(sender_name), sender_device)
    receiver = load_model(resolve_model(receiver_name), receiver_device)

    recolor_path = out / "recolor.pt"
    if recolor_path.exists():
        base_map = torch.load(recolor_path, map_location="cpu",
                              weights_only=False)
        print(f"[calibrate] reusing Stage-1 map {recolor_path}")
    else:
        started = time.time()
        per_corpus = {}
        for label in sorted({row["dataset"] for row in train_rows}):
            texts = [row["prompt"] for row in train_rows
                     if row["dataset"] == label]
            per_corpus[label] = recolor.collect_statistics(
                sender, receiver, texts, span=args.span, stride=args.stride,
                device=receiver_device, log=print, tag=f" {label}:")
        stats = recolor.balance_corpora(per_corpus)
        del per_corpus
        base_map = recolor.fit_map(stats, device=receiver_device, log=print)
        del stats
        base_map.update(sender=sender_name, receiver=receiver_name)
        base_map["calibration"].update(
            calibration_id=manifest.get("calibration_id"),
            prompt_sha256_sequence=manifest.get("prompt_sha256_sequence"),
            train_prompts=len(train_rows))
        atomic_save(base_map, recolor_path)
        torch.cuda.empty_cache()
        print(f"[calibrate] Stage 1 done in {time.time() - started:.0f}s "
              f"-> {recolor_path}")

    started = time.time()
    result = calibration.calibrate(
        sender, receiver, base_map, rows, device=receiver_device,
        out_path=final_path, checkpoint_path=out / "train_checkpoint.pt",
        rank=args.rank, epochs=args.epochs, lr=args.lr, seed=args.seed,
        grad_clip=args.grad_clip, lambda_k_output=args.lambda_k_output,
        grad_accum=args.grad_accum, log=print)
    record = result["functional_correction"]
    (out / "calibration_log.json").write_text(json.dumps({
        "pair": args.pair,
        "arguments": {key: value for key, value in vars(args).items()
                      if key not in ("function", "command")},
        "baseline_validation": record["baseline_validation"],
        "best_validation_loss": record["best_validation_loss"],
        "history": record["history"],
        "stage2_seconds": time.time() - started}, indent=2))
    print(f"[calibrate] Stage 2 done in {time.time() - started:.0f}s "
          f"-> {final_path}")


def evaluate(args) -> None:
    from .evaluation import run

    run.run(args)


def summarize(args) -> None:
    from .evaluation import summarize as module

    module.run(args)


def inspect_map(args) -> None:
    from .maps import describe, load_map

    mapping = load_map(args.map, mmap=True)
    info = describe(mapping)
    correction = mapping.get("functional_correction") or {}
    info["history"] = correction.get("history")
    info["baseline_validation"] = correction.get("baseline_validation")
    print(json.dumps(info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    from .evaluation.run import BENCHMARKS

    parser = argparse.ArgumentParser(
        prog="python -m heterofold",
        description="HeteroFold calibration and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser(
        "prepare-data", help="download benchmarks, build the calibration set")
    p.add_argument("--data-dir", default=_data_dir_default())
    p.add_argument("--skip-benchmarks", action="store_true",
                   help="only what the calibration set needs")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(function=prepare_data)

    p = commands.add_parser(
        "download-models", help="fetch pinned checkpoints from the Hub")
    p.add_argument("models", nargs="*",
                   default=["llama8", "qwen4", "ministral14"])
    p.set_defaults(function=download_models)

    p = commands.add_parser("calibrate", help="fit a HeteroFold map")
    p.add_argument("--pair", required=True,
                   help="<sender>_to_<receiver>, e.g. llama8_to_qwen4")
    p.add_argument("--out", required=True, help="output directory for the map")
    p.add_argument("--data-dir", default=_data_dir_default())
    p.add_argument("--manifest", default=None,
                   help="calibration manifest (default: built by prepare-data)")
    group = p.add_argument_group("layer alignment and Recolor (Stage 1)")
    group.add_argument("--span", type=int, default=1,
                       help="sender layers on each side of the matched layer "
                            "(1 -> three sender layers)")
    group.add_argument("--stride", type=int, default=4,
                       help="depth offset between neighbouring sender layers")
    group = p.add_argument_group("output-aware calibration (Stage 2)")
    group.add_argument("--rank", type=int, default=16, help="correction rank")
    group.add_argument("--epochs", type=int, default=4)
    group.add_argument("--lr", type=float, default=3e-4)
    group.add_argument("--seed", type=int, default=20260825)
    group.add_argument("--grad-clip", type=float, default=1.0)
    group.add_argument("--grad-accum", type=int, default=8,
                       help="prompts averaged per optimizer step")
    group.add_argument("--lambda-k-output", type=float, default=1.0)
    _add_device_arguments(p)
    p.set_defaults(function=calibrate)

    p = commands.add_parser("evaluate", help="evaluate benchmarks")
    p.add_argument("--pair", default=None,
                   help="<sender>_to_<receiver>, e.g. llama8_to_qwen4")
    p.add_argument("--sender", default=None)
    p.add_argument("--receiver", default=None)
    p.add_argument("--map", default=None, help="HeteroFold map (.pt)")
    p.add_argument("--native", action="store_true",
                   help="receiver prefills the prompt itself (no transfer)")
    p.add_argument("--benchmark", required=True, nargs="+",
                   choices=BENCHMARKS, metavar="BENCHMARK",
                   help="one or more of: " + ", ".join(BENCHMARKS))
    p.add_argument("--out", required=True,
                   help="run directory; results go to <out>/<benchmark>/")
    p.add_argument("--data-dir", default=_data_dir_default())
    p.add_argument("--shard", default=None, help="i/n: evaluate shard i of n")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N items of the range")
    p.add_argument("--overwrite", action="store_true")
    _add_device_arguments(p)
    p.set_defaults(function=evaluate)

    p = commands.add_parser("summarize", help="merge shards into a table")
    p.add_argument("--results", default="results")
    p.add_argument("--pair", default=None)
    p.add_argument("--paper", action="store_true",
                   help="also print the paper's Table 1 values")
    p.add_argument("--out", default=None, help="write <out>.md and <out>.json")
    p.set_defaults(function=summarize)

    p = commands.add_parser("inspect-map", help="print map metadata")
    p.add_argument("map")
    p.set_defaults(function=inspect_map)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)
