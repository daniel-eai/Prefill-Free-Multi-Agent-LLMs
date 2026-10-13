from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import requests

SEED = 20260825
PROMPT_DIGEST = (
    "8a1f9d7684e32bc416a3c278bd41dcd470ad010f2e2dd5e8067d5f845b848113")
OPENR1_COUNT = 1000

OPENR1_REPO = "open-r1/OpenR1-Math-220k"
OPENR1_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
OPENR1_FILE = "data/train-00000-of-00010.parquet"
HOTPOT_REPO = "hotpotqa/hotpot_qa"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_FILE = "distractor/train-00000-of-00002.parquet"
OPENR1 = "Open-R1 prompt-only"
HOTPOT = "HotpotQA distractor train full-context"


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in value if character.isalnum())


def _download(repo: str, revision: str, filename: str, output: Path) -> None:
    if output.is_file() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(8 << 20):
                if chunk:
                    stream.write(chunk)
    os.replace(temporary, output)


def evaluation_exclusions(qasper: Path, hotpot: Path):
    from datasets import load_dataset

    def rows(path):
        return [json.loads(line) for line in path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    questions = {normalize(question) for question in arc["question"]}
    hotpot_rows = rows(hotpot)
    questions.update(normalize(row["input"]) for row in rows(qasper))
    questions.update(normalize(row["input"]) for row in hotpot_rows)
    title_pattern = re.compile(r"(?:^|\n)Passage\s+\d+:\n([^\n]+)\n")
    titles = {normalize(title) for row in hotpot_rows
              for title in title_pattern.findall(str(row["context"]))
              if normalize(title)}
    return questions, titles


def select_openr1(parquet: Path, excluded: set[str]) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["problem", "source", "generations"])
    first, rest = [], []
    for source_row, (problem, source, generations) in enumerate(zip(
            table.column("problem").to_pylist(),
            table.column("source").to_pylist(),
            table.column("generations").to_pylist())):
        if not generations or not problem:
            continue
        source = (source or "").lower()
        item = (source_row, (problem + "\n" + generations[0])[:9000])
        if "aime" in source:
            first.append(item)
        elif any(tag in source for tag in ("olympiad", "competition", "math")):
            rest.append(item)
        if (len(first) >= OPENR1_COUNT
                or len(first) + len(rest) >= 4 * OPENR1_COUNT):
            break
    chosen = (first + rest)[:OPENR1_COUNT]
    if len(chosen) < OPENR1_COUNT:
        raise ValueError(f"only {len(chosen)} eligible Open-R1 problems")
    candidates = []
    for index, (source_row, text) in enumerate(chosen):
        problem = text.split("<think>", 1)[0].strip()
        if not problem or normalize(problem) in excluded:
            raise ValueError(f"unexpected Open-R1 problem at row {source_row}")
        prompt = f"Problem: {problem}\n\nSolution:"
        start = len("Problem:")
        while prompt[start].isspace():
            start += 1
        end = prompt.rfind("\n\nSolution:")
        candidates.append({
            "dataset": OPENR1,
            "source_repo": OPENR1_REPO, "source_revision": OPENR1_REVISION,
            "source_file": OPENR1_FILE, "source_row": source_row,
            "identity": f"openr1:{index}",
            "prompt": prompt,
            "question": prompt[start:end],
            "question_sha256": digest_text(normalize(problem)),
            "question_char_start": start,
            "question_char_end": end,
        })
    validation = candidates[4::5][:200]
    held_out = {row["identity"] for row in validation}
    train = [row for row in candidates
             if row["identity"] not in held_out][:800]
    if len(train) != 800 or len(validation) != 200:
        raise ValueError("unexpected Open-R1 split")
    return ([dict(row, split="train") for row in train]
            + [dict(row, split="validation") for row in validation])


def _hotpot_prompt(titles, sentences, question):
    passages = [
        f"Passage {index}:\n{title}\n{''.join(map(str, body)).strip()}"
        for index, (title, body) in enumerate(zip(titles, sentences), start=1)]
    prefix = "\n\n".join(passages) + "\n\nQuestion: "
    return prefix + question + "\nAnswer:", len(prefix), len(prefix) + len(question)


def select_hotpot(parquet: Path, excluded_questions: set[str],
                  excluded_titles: set[str], seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["id", "question", "context"])
    candidates, seen = [], set()
    for source_row, (identifier, question, context) in enumerate(zip(
            table.column("id").to_pylist(),
            table.column("question").to_pylist(),
            table.column("context").to_pylist())):
        question = str(question or "").strip()
        key = normalize(question)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in excluded_questions:
            continue
        titles = [str(value) for value in context["title"]]
        title_keys = {normalize(value) for value in titles if normalize(value)}
        if title_keys & excluded_titles:
            continue
        sentences = [list(map(str, value)) for value in context["sentences"]]
        prompt, start, end = _hotpot_prompt(titles, sentences, question)
        candidates.append({
            "dataset": HOTPOT,
            "source_repo": HOTPOT_REPO, "source_revision": HOTPOT_REVISION,
            "source_file": HOTPOT_FILE, "source_row": source_row,
            "hotpot_id": str(identifier),
            "identity": f"hotpotqa:{identifier}",
            "document_titles": titles,
            "document_title_hashes": sorted(title_keys),
            "prompt": prompt, "question": question,
            "question_sha256": digest_text(key),
            "question_char_start": start, "question_char_end": end,
        })
    candidates.sort(key=lambda row: digest_text(
        f"{seed}:hotpot:select:{row['identity']}"))
    used: set[str] = set()

    def take(count: int, split: str, forbidden: set[str]) -> list[dict]:
        selected = []
        for row in candidates:
            if row["identity"] in used:
                continue
            if set(row["document_title_hashes"]) & forbidden:
                continue
            used.add(row["identity"])
            selected.append(dict(row, split=split))
            if len(selected) == count:
                return selected
        raise ValueError(f"only {len(selected)}/{count} HotpotQA rows for {split}")

    train = take(800, "train", set())
    train_titles = {title for row in train
                    for title in row["document_title_hashes"]}
    validation = take(200, "validation", train_titles)
    return train + validation


def build_manifest(out_path: Path, *, qasper: Path, hotpot: Path,
                   sources_dir: Path, log=print) -> dict:
    excluded_questions, excluded_titles = evaluation_exclusions(qasper, hotpot)
    openr1_path = sources_dir / "openr1-train-00000-of-00010.parquet"
    hotpot_path = sources_dir / "hotpot-distractor-train-00000-of-00002.parquet"
    _download(OPENR1_REPO, OPENR1_REVISION, OPENR1_FILE, openr1_path)
    _download(HOTPOT_REPO, HOTPOT_REVISION, HOTPOT_FILE, hotpot_path)
    openr1 = select_openr1(openr1_path, excluded_questions)
    hotpot_rows = select_hotpot(
        hotpot_path, excluded_questions, excluded_titles, SEED)
    rows = ([row for row in openr1 if row["split"] == "train"]
            + [row for row in hotpot_rows if row["split"] == "train"]
            + [row for row in openr1 if row["split"] == "validation"]
            + [row for row in hotpot_rows if row["split"] == "validation"])
    for index, row in enumerate(rows):
        row["index"] = index
    if len({row["identity"] for row in rows}) != 2000:
        raise ValueError("duplicate calibration identities")
    prompt_digest = digest_text("\n".join(
        digest_text(row["prompt"]) for row in rows))
    if prompt_digest != PROMPT_DIGEST:
        raise ValueError(
            "calibration prompts differ from the pinned set "
            f"({prompt_digest} != {PROMPT_DIGEST}); check the source "
            "parquet files and evaluation data")
    counts = Counter((row["dataset"], row["split"]) for row in rows)
    manifest = {
        "schema_version": 1,
        "calibration_id": "openr1_hotpot_n2000",
        "seed": SEED,
        "prompt_sha256_sequence": prompt_digest,
        "counts": {f"{dataset}:{split}": count
                   for (dataset, split), count in sorted(counts.items())},
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(out_path) + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    os.replace(temporary, out_path)
    log(f"[calibration-data] {len(rows)} prompts -> {out_path} "
        f"(digest {prompt_digest[:16]}...)")
    return manifest


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
