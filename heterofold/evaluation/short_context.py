from __future__ import annotations

import re

from ..data.benchmarks import SHORT_CONTEXT
from .runtime import Progress, Runtime, candidate_scores, generate, greedy_decode

BENCHMARKS = tuple(SHORT_CONTEXT)
GSM8K_MAX_NEW_TOKENS = 4096


def load(name: str):
    from datasets import load_dataset

    dataset, config, split = SHORT_CONTEXT[name]
    return load_dataset(dataset, config, split=split)


def format_item(name: str, example: dict):
    if name == "arc":
        labels = [str(value).strip() for value in example["choices"]["label"]]
        texts = [str(value).strip() for value in example["choices"]["text"]]
        choices = "\n".join(f"{label}. {text}"
                            for label, text in zip(labels, texts))
        prompt = (
            "Answer the following multiple-choice question. Select the single "
            "correct option.\n\n"
            f"Question: {example['question'].strip()}\n\nChoices:\n{choices}\n\n"
            "Answer with only the option label.")
        return prompt, labels, labels.index(
            str(example["answerKey"]).strip()), "arc_label", {}
    if name == "mmlu":
        labels = ["A", "B", "C", "D"]
        choices = "\n".join(f"{label}. {text}"
                            for label, text in zip(labels, example["choices"]))
        subject = example["subject"].replace("_", " ")
        prompt = (
            f"Answer the following multiple-choice question about {subject}.\n\n"
            f"Question: {example['question'].strip()}\n\nChoices:\n{choices}\n\n"
            "Answer with only the option label.")
        return (prompt, [" A", " B", " C", " D"], int(example["answer"]),
                "label", {"subject": example["subject"]})
    if name == "winogrande":
        before, after = example["sentence"].split("_", 1)
        prompt = ("Complete the sentence with the most plausible option.\n\n"
                  f"Sentence: {before}")
        candidates = [example["option1"] + after, example["option2"] + after]
        return prompt, candidates, int(example["answer"]) - 1, "continuation", {}
    if name == "hellaswag":
        prompt = ("Choose the most plausible continuation of the following "
                  f"context.\n\nContext: {example['ctx'].strip()}\n\n"
                  "Continuation:")
        candidates = [" " + value.strip() for value in example["endings"]]
        return prompt, candidates, int(example["label"]), "continuation", {}
    if name == "gsm8k":
        prompt = (
            "Solve the following math problem carefully. Show your reasoning "
            "and finish with the final numeric answer in \\boxed{...}.\n\n"
            f"Problem: {example['question'].strip()}")
        gold = example["answer"].split("####")[-1].strip()
        return prompt, None, gold, "generation", {}
    raise ValueError(name)


def normalize_number(value: str) -> str:
    value = value.strip().replace(",", "")
    value = value.replace("$", "").replace("%", "")
    if value.endswith("."):
        value = value[:-1]
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".12g")
    except ValueError:
        return value


def extract_numeric_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    marker = re.findall(r"####\s*([^\n]+)", text)
    if marker:
        return normalize_number(marker[-1])
    numbers = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[0-9]+)?%?", text)
    return normalize_number(numbers[-1]) if numbers else ""


ANSWER_CUE = {"arc_label": "The correct option is ", "label": "The answer is",
              "continuation": ""}


def evaluate(runtime: Runtime, name: str, start: int, end: int,
             max_new_tokens: int = GSM8K_MAX_NEW_TOKENS) -> dict:
    dataset = load(name)
    end = min(end, len(dataset))
    rows = []
    progress = Progress(f"{name}:{runtime.mode}", end - start)
    for index in range(start, end):
        prompt, candidates, gold, scoring, metadata = format_item(
            name, dataset[index])
        if scoring == "generation":
            if runtime.mode == "native":
                initial = runtime.native_ids(prompt, runtime.assistant)
                text, generated = greedy_decode(runtime, initial, max_new_tokens)
                cache_tokens = 0
            else:
                cache, cache_tokens, _ = runtime.transfer_cache(prompt)
                text, generated = generate(
                    runtime, runtime.suffix_ids(runtime.assistant),
                    max_new_tokens, past_key_values=cache,
                    past_length=cache_tokens)
                del cache
            prediction = extract_numeric_answer(text)
            gold = normalize_number(gold)
            row = {"index": index, "gold": gold, "prediction": prediction,
                   "correct": int(prediction == gold),
                   "generated_tokens": generated, "generation": text,
                   "transfer_cache_tokens": cache_tokens, **metadata}
        else:
            result = candidate_scores(
                runtime, prompt, candidates, scoring == "continuation",
                runtime.assistant + ANSWER_CUE[scoring])
            scores = result["scores"]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            row = {"index": index, "gold": gold, "prediction": prediction,
                   "correct": int(prediction == gold), **result, **metadata}
        rows.append(row)
        progress.update(row["correct"])
    return {"benchmark": name, "metric": "accuracy", "rows": rows,
            "score_key": "correct"}
