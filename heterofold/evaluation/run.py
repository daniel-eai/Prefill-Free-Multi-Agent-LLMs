from __future__ import annotations

import time
from pathlib import Path

from . import longbench, quality, short_context
from .runtime import Runtime, shard_range, write_json

EXPECTED_ITEMS = {
    "qasper": 200, "hotpotqa": 200, "locomo": 1540, "quality": 2086,
    "arc": 1172, "mmlu": 14042, "winogrande": 1267, "hellaswag": 10042,
    "gsm8k": 1319,
}
LONG_CONTEXT = ("qasper", "hotpotqa", "locomo", "quality")
SHORT_CONTEXT = ("arc", "mmlu", "winogrande", "hellaswag", "gsm8k")
BENCHMARKS = LONG_CONTEXT + SHORT_CONTEXT


def evaluate_benchmark(runtime: Runtime, benchmark: str, data_dir: Path,
                       start: int, end: int) -> dict:
    if benchmark in short_context.BENCHMARKS:
        return short_context.evaluate(runtime, benchmark, start, end)
    if benchmark == "quality":
        return quality.evaluate(runtime, data_dir, start, end)
    if benchmark in longbench.BENCHMARKS:
        return longbench.evaluate(runtime, benchmark, data_dir, start, end)
    raise ValueError(f"unknown benchmark {benchmark!r}")


def run(args) -> None:
    pending = []
    for benchmark in args.benchmark:
        start, end = shard_range(
            EXPECTED_ITEMS[benchmark], args.shard, args.start, args.end)
        if args.limit:
            end = min(end, start + args.limit)
        out_path = (Path(args.out) / benchmark
                    / f"items_{start:05d}_{end:05d}.json")
        if out_path.exists() and not args.overwrite:
            print(f"[evaluate] {out_path} exists; skipping (use --overwrite)")
        elif end > start:
            pending.append((benchmark, start, end, out_path))
    if not pending:
        return
    runtime = Runtime.from_args(args)
    for benchmark, start, end, out_path in pending:
        started = time.time()
        result = evaluate_benchmark(
            runtime, benchmark, Path(args.data_dir), start, end)
        rows = result["rows"]
        score = (100.0 * sum(row[result["score_key"]] for row in rows)
                 / max(len(rows), 1))
        write_json(out_path, {
            "benchmark": benchmark, "metric": result["metric"],
            "score_key": result["score_key"], "start": start, "end": end,
            "n": len(rows), "score_percent": score,
            "pair": args.pair, **runtime.describe(),
            "elapsed_seconds": time.time() - started, "rows": rows,
        })
        print(f"[evaluate] DONE {benchmark} {runtime.mode} items "
              f"{start}:{end} score={score:.2f} -> {out_path}", flush=True)
