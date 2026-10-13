from __future__ import annotations

from pathlib import Path

from ..data.benchmarks import load_jsonl, verify
from .runtime import Progress, Runtime, candidate_scores

LABELS = ("A", "B", "C", "D")
ANSWER_CUE = "The correct option is "


def load_examples(data_dir: Path) -> list[dict]:
    path = verify(data_dir, "quality/quality_dev.jsonl")
    examples = []
    for record in load_jsonl(path):
        for question in record["questions"]:
            examples.append({
                "index": len(examples),
                "article_id": record["article_id"],
                "question_id": question["question_unique_id"],
                "article": record["article"],
                "question": question["question"],
                "options": question["options"],
                "gold": LABELS[int(question["gold_label"]) - 1],
                "hard": bool(question["difficult"]),
            })
    return examples


def format_prompt(example: dict) -> str:
    choices = "\n".join(f"{label}. {option}"
                        for label, option in zip(LABELS, example["options"]))
    return (
        "Read the following article and answer the multiple-choice question.\n\n"
        f"Article:\n{example['article'].strip()}\n\n"
        f"Question:\n{example['question'].strip()}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer with only the option label.")


def evaluate(runtime: Runtime, data_dir: Path, start: int, end: int) -> dict:
    examples = load_examples(data_dir)
    end = min(end, len(examples))
    suffix = runtime.assistant + ANSWER_CUE
    rows = []
    progress = Progress(f"quality:{runtime.mode}", end - start)
    for example in examples[start:end]:
        result = candidate_scores(
            runtime, format_prompt(example), LABELS, False, suffix)
        scores = result["scores"]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        prediction = LABELS[order[0]]
        rows.append({
            "index": example["index"], "article_id": example["article_id"],
            "question_id": example["question_id"], "hard": example["hard"],
            "gold": example["gold"], "prediction": prediction,
            "correct": int(prediction == example["gold"]),
            "margin": float(scores[order[0]] - scores[order[1]]), **result})
        progress.update(rows[-1]["correct"])
    return {"benchmark": "quality", "metric": "accuracy", "rows": rows,
            "score_key": "correct"}
