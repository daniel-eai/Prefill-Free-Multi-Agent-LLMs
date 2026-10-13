from __future__ import annotations

import re
import string
from collections import Counter
from pathlib import Path

from ..data.benchmarks import load_jsonl, verify
from .runtime import (
    Progress,
    Runtime,
    generate,
    strip_terminal_markers,
    terminal_markers,
)

PROMPTS = {
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nArticle: {context}\n\n Answer the question based on "
        "the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is "
        "a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\n"
        "Answer:"),
    "locomo": (
        "You are a helpful assistant whose job is to understand the following "
        "conversation and answer questions based on it. If the answer is unknown, "
        "do not invent information.\n\n{context}\n\n"
        "Based on the above conversations, write a short answer for the following "
        "question in a few words. Do not write complete and lengthy sentences. "
        "Answer with exact words from the conversations whenever possible.\n\n"
        "Question: {input}\nAnswer:"),
}
FILES = {"qasper": "longbench/qasper.jsonl",
         "hotpotqa": "longbench/hotpotqa.jsonl",
         "locomo": "locomo/locomo10_flat.jsonl"}
SETTINGS = {"qasper": (128, 0.6, 30000), "hotpotqa": (32, 0.6, 30000),
            "locomo": (64, 0.0, 100000)}
TOP_P, TOP_K, SEED = 0.95, 20, 1001
BENCHMARKS = tuple(PROMPTS)


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(character for character in value
                    if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def qa_f1(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    same = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
    if not same:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


_STEMMER = None


def _stem(word: str) -> str:
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()
    return _STEMMER.stem(word)


def _locomo_normalize(value: str) -> str:
    value = value.replace(",", "").lower()
    value = "".join(character for character in value
                    if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    return " ".join(value.split())


def _locomo_f1(prediction: str, answer: str) -> float:
    predicted = [_stem(word) for word in _locomo_normalize(prediction).split()]
    expected = [_stem(word) for word in _locomo_normalize(answer).split()]
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0 or not predicted or not expected:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def locomo_score(prediction: str, answer: str, category: int) -> float:
    if category == 3:
        answer = answer.split(";", 1)[0].strip()
    if category == 1:
        predictions = [value.strip() for value in prediction.split(",")]
        answers = [value.strip() for value in answer.split(",")]
        return sum(max(_locomo_f1(value, expected) for value in predictions)
                   for expected in answers) / len(answers)
    return _locomo_f1(prediction, answer)


def locomo_clean(value: str) -> str:
    value = value.replace('\\"', "'").strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    value = lines[0] if lines else ""
    value = re.sub(r"^\s*(?:\([ab]\)|[ab]\))\s*", "", value, flags=re.I)
    value = re.sub(r"^\s*answer\s*:\s*", "", value, flags=re.I)
    return value.strip()


def truncate_prompt(prompt: str, tokenizer, max_tokens: int):
    token_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if len(token_ids) <= max_tokens:
        return prompt, len(token_ids), False
    half = max_tokens // 2
    retained = token_ids[:half] + token_ids[-(max_tokens - half):]
    return tokenizer.decode(retained, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False), len(token_ids), True


def evaluate(runtime: Runtime, name: str, data_dir: Path, start: int,
             end: int) -> dict:
    source = load_jsonl(verify(data_dir, FILES[name]))
    end = min(end, len(source))
    max_new_tokens, temperature, max_prompt_tokens = SETTINGS[name]
    markers = terminal_markers(runtime)
    rows = []
    progress = Progress(f"{name}:{runtime.mode}", end - start, every=10)
    for index in range(start, end):
        item = source[index]
        prompt = PROMPTS[name].format(context=item["context"],
                                      input=item["input"])
        prompt, prompt_tokens, truncated = truncate_prompt(
            prompt, runtime.tokenizer, max_prompt_tokens)
        if name == "locomo" and truncated:
            raise ValueError("LoCoMo prompt unexpectedly truncated")
        sampling = dict(temperature=temperature, top_p=TOP_P, top_k=TOP_K,
                        seed=SEED + index)
        if runtime.mode == "native":
            text, generated = generate(
                runtime, runtime.native_ids(prompt, runtime.assistant),
                max_new_tokens, **sampling)
            cache_tokens = 0
        else:
            cache, cache_tokens, _ = runtime.transfer_cache(prompt)
            text, generated = generate(
                runtime, runtime.suffix_ids(runtime.assistant), max_new_tokens,
                past_key_values=cache, past_length=cache_tokens, **sampling)
            del cache
        prediction = strip_terminal_markers(text, markers)
        row = {"index": index, "receiver_prompt_tokens": prompt_tokens,
               "prompt_truncated": truncated,
               "transfer_cache_tokens": cache_tokens,
               "generated_tokens": generated}
        if name == "locomo":
            prediction = locomo_clean(prediction)
            row.update(
                answers=[item["answer"]], category=int(item["category"]),
                sample_id=item["sample_id"],
                f1=locomo_score(prediction, item["answer"],
                                int(item["category"])))
        else:
            answers = [str(value) for value in item["answers"]]
            row.update(answers=answers,
                       f1=max(qa_f1(prediction, answer) for answer in answers))
        row["prediction"] = prediction
        rows.append(row)
        progress.update(row["f1"])
    return {"benchmark": name, "metric": "token_f1", "rows": rows,
            "score_key": "f1"}
