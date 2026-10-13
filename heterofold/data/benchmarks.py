from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import requests

LONGBENCH_REPO = "THUDM/LongBench"
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
QUALITY_URL = ("https://raw.githubusercontent.com/nyu-mll/quality/main/data/"
               "v1.0.1/QuALITY.v1.0.1.htmlstripped.dev")
LOCOMO_URL = ("https://raw.githubusercontent.com/snap-research/locomo/main/"
              "data/locomo10.json")

SHA256 = {
    "longbench/qasper.jsonl":
        "29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854",
    "longbench/hotpotqa.jsonl":
        "a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f",
    "quality/quality_dev.jsonl":
        "0cb88973a370d21a7f6ffdb6db1dfa5dc39f3eaca074a73d775f2dd0e1ab07b1",
    "locomo/locomo10.json":
        "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    "locomo/locomo10_flat.jsonl":
        "43cb7302445367ee6d5cd3fe402f8fed88e7c987e6a94c91b6cbec4418f264d7",
}
SHORT_CONTEXT = {
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
}
LOCOMO_ANSWERABLE = (1, 2, 3, 4)


def default_data_dir() -> Path:
    return Path(os.environ.get("HETEROFOLD_DATA", "data")).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m heterofold prepare-data`")
    actual = sha256(path)
    if actual != SHA256[relative]:
        raise ValueError(
            f"{path} has sha256 {actual}, expected {SHA256[relative]}")
    return path


def _ready(root: Path, relative: str) -> bool:
    path = root / relative
    return path.is_file() and sha256(path) == SHA256[relative]


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(8 << 20):
                if chunk:
                    stream.write(chunk)
    os.replace(temporary, output)


def prepare_longbench(root: Path, log=print) -> None:
    names = ("qasper", "hotpotqa")
    if all(_ready(root, f"longbench/{name}.jsonl") for name in names):
        log("[data] LongBench ready")
        return
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(LONGBENCH_REPO, "data.zip", repo_type="dataset",
                              revision=LONGBENCH_REVISION)
    (root / "longbench").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in names:
            (root / "longbench" / f"{name}.jsonl").write_bytes(
                bundle.read(f"data/{name}.jsonl"))
    for name in names:
        verify(root, f"longbench/{name}.jsonl")
    log("[data] LongBench qasper/hotpotqa written")


def prepare_quality(root: Path, log=print) -> None:
    relative = "quality/quality_dev.jsonl"
    if _ready(root, relative):
        log("[data] QuALITY ready")
        return
    raw = root / "quality" / "QuALITY.v1.0.1.htmlstripped.dev"
    _download(QUALITY_URL, raw)
    order, articles, questions = [], {}, {}
    with raw.open(encoding="utf-8") as stream:
        for record in map(json.loads, stream):
            key = record["article_id"]
            if key not in articles:
                order.append(key)
                articles[key] = record["article"]
                questions[key] = []
            questions[key].extend(record["questions"])
    lines = []
    for key in order:
        article = articles[key]
        article_id = hashlib.sha256(article.encode("utf-8")).hexdigest()[:16]
        lines.append(json.dumps({
            "article_id": article_id,
            "article": article,
            "questions": [{
                "question_unique_id": f"{article_id}_{index}",
                "question": question["question"],
                "options": [option.strip() for option in question["options"]],
                "gold_label": question["gold_label"],
                "difficult": bool(question["difficult"]),
            } for index, question in enumerate(questions[key])],
        }, ensure_ascii=False) + "\n")
    (root / relative).write_text("".join(lines), encoding="utf-8")
    verify(root, relative)
    raw.unlink()
    log("[data] QuALITY dev written")


def _serialize_conversation(conversation: dict) -> str:
    sessions = sorted(
        int(key.removeprefix("session_")) for key in conversation
        if key.startswith("session_")
        and key.removeprefix("session_").isdigit())
    blocks = []
    for session in sessions:
        date = conversation.get(f"session_{session}_date_time", "unknown date")
        lines = [f"DATE: {date}", "CONVERSATION:"]
        lines += [f"{turn['speaker']}: {turn.get('text', '')}"
                  for turn in conversation[f"session_{session}"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def prepare_locomo(root: Path, log=print) -> None:
    relative = "locomo/locomo10_flat.jsonl"
    if _ready(root, relative):
        log("[data] LoCoMo ready")
        return
    raw = root / "locomo" / "locomo10.json"
    if not _ready(root, "locomo/locomo10.json"):
        _download(LOCOMO_URL, raw)
    verify(root, "locomo/locomo10.json")
    rows = []
    for conversation_index, item in enumerate(
            json.loads(raw.read_text(encoding="utf-8"))):
        conversation = item["conversation"]
        context = _serialize_conversation(conversation)
        dialogue_ids = {
            str(turn["dia_id"]) for key in conversation
            if key.startswith("session_")
            and key.removeprefix("session_").isdigit()
            for turn in conversation[key]}
        for qa_index, qa in enumerate(item["qa"]):
            category = int(qa["category"])
            if category not in LOCOMO_ANSWERABLE:
                continue
            evidence = [str(value) for value in qa.get("evidence", [])]
            missing = [value for value in evidence if value not in dialogue_ids]
            rows.append({
                "_id": f"{item['sample_id']}::{qa_index}",
                "dataset": "locomo",
                "context": context,
                "input": str(qa["question"]),
                "question": str(qa["question"]),
                "answer": str(qa["answer"]),
                "answers": [str(qa["answer"])],
                "category": str(category),
                "evidence": evidence,
                "evidence_dialogue_ids": list(evidence),
                "sample_id": str(item["sample_id"]),
                "conversation_index": conversation_index,
                "qa_index": qa_index,
                "length": len(context),
                "all_classes": None,
                "missing_evidence_dialogue_ids_in_source": missing,
                "all_evidence_present_in_full_context": not missing,
            })
    if len(rows) != 1540:
        raise ValueError(f"expected 1540 answerable rows, found {len(rows)}")
    with (root / relative).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    verify(root, relative)
    log("[data] LoCoMo flattened (1,540 questions)")


def prepare_short_context(log=print) -> None:
    from datasets import load_dataset

    for name, (dataset, config, split) in SHORT_CONTEXT.items():
        rows = load_dataset(dataset, config, split=split)
        log(f"[data] {name}: {dataset} {config or ''} {split} n={len(rows)}")


def prepare_all(root: Path, log=print) -> None:
    root.mkdir(parents=True, exist_ok=True)
    prepare_longbench(root, log)
    prepare_quality(root, log)
    prepare_locomo(root, log)
    prepare_short_context(log)


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
