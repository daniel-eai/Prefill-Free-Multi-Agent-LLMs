from __future__ import annotations

import json
from pathlib import Path

from .run import BENCHMARKS, EXPECTED_ITEMS

DISPLAY = {"qasper": "Qasper", "hotpotqa": "HotpotQA", "locomo": "LoCoMo",
           "quality": "QuALITY", "arc": "ARC-C", "mmlu": "MMLU",
           "winogrande": "WinoGrande", "hellaswag": "HellaSwag",
           "gsm8k": "GSM8K"}

PAPER_NATIVE = {
    "llama8": [44.64, 57.54, 52.33, 75.07, 82.76, 66.74, 65.82, 70.78, 85.67],
    "qwen4": [43.07, 56.05, 38.92, 70.61, 87.80, 67.95, 55.09, 41.54, 90.14],
    "ministral14": [45.59, 64.56, 52.01, 83.32, 92.66, 75.63, 61.56, 68.33, 94.69],
}
PAPER_HETEROFOLD = {
    "qwen4_to_llama8": [29.12, 38.48, 29.66, 62.85, 78.92, 63.28, 57.30, 65.60, 55.12],
    "ministral14_to_qwen4": [42.00, 52.74, 35.59, 75.55, 88.99, 72.14, 53.83, 41.46, 89.23],
    "llama8_to_qwen4": [35.99, 49.44, 34.83, 62.90, 77.05, 62.39, 52.88, 41.74, 70.74],
    "qwen4_to_ministral14": [20.57, 42.65, 17.75, 66.30, 85.49, 66.24, 55.88, 65.22, 77.26],
    "ministral14_to_llama8": [33.72, 50.29, 43.15, 79.00, 90.36, 71.76, 63.22, 69.16, 63.46],
    "llama8_to_ministral14": [21.58, 39.00, 30.43, 72.05, 81.14, 65.92, 57.14, 66.41, 72.33],
}


def paper_values(name: str, run: str) -> list[float] | None:
    if run == "native":
        return PAPER_NATIVE.get(name.split("_to_")[-1])
    if run == "heterofold":
        return PAPER_HETEROFOLD.get(name)
    return None


def merge_benchmark(directory: Path, benchmark: str) -> dict | None:
    rows, key = {}, None
    for path in sorted(directory.glob("items_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = payload["score_key"]
        for row in payload["rows"]:
            rows[int(row["index"])] = row
    if not rows:
        return None
    expected = EXPECTED_ITEMS[benchmark]
    complete = sorted(rows) == list(range(expected))
    score = 100.0 * sum(row[key] for row in rows.values()) / len(rows)
    return {"n": len(rows), "expected": expected, "complete": complete,
            "score": score}


def summarize_run(run_dir: Path) -> dict:
    return {benchmark: merged for benchmark in BENCHMARKS
            if (merged := merge_benchmark(run_dir / benchmark, benchmark))}


def _cell(entry: dict | None) -> str:
    if entry is None:
        return "--"
    text = f"{entry['score']:.2f}"
    return text if entry["complete"] else f"{text}* ({entry['n']}/{entry['expected']})"


def run(args) -> None:
    root = Path(args.results)
    pairs = [p for p in sorted(root.iterdir()) if p.is_dir()] if not args.pair \
        else [root / args.pair]
    summary = {}
    header = "| Direction | Run | " + " | ".join(DISPLAY[b] for b in BENCHMARKS) + " |"
    lines = [header, "|---|---|" + "---:|" * len(BENCHMARKS)]
    for pair_dir in pairs:
        if not pair_dir.is_dir():
            continue
        for run_dir in sorted(p for p in pair_dir.iterdir() if p.is_dir()):
            cells = summarize_run(run_dir)
            if not cells:
                continue
            summary.setdefault(pair_dir.name, {})[run_dir.name] = cells
            lines.append(
                f"| {pair_dir.name} | {run_dir.name} | "
                + " | ".join(_cell(cells.get(b)) for b in BENCHMARKS) + " |")
            paper = paper_values(pair_dir.name, run_dir.name)
            if paper and args.paper:
                lines.append(
                    f"| | *paper* | "
                    + " | ".join(f"*{value:.2f}*" for value in paper) + " |")
    table = "\n".join(lines)
    print(table)
    if any(not cell["complete"] for runs in summary.values()
           for cells in runs.values() for cell in cells.values()):
        print("\n* partial coverage (evaluated / expected items)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        out.with_suffix(".md").write_text(table + "\n")
        print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
