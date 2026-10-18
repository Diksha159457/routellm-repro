"""Benchmark items for the routing experiments: MMLU (multiple choice) and GSM8K (math).

Each item is {"id", "dataset", "subject", "question", "prompt", "gold"}.
`question` is what the routers embed; `prompt` is what the LLMs receive.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

LETTERS = "ABCD"

MMLU_PROMPT = (
    "Answer the following multiple-choice question. Think step by step briefly, "
    "then end your reply with a final line of the form 'Answer: X' where X is A, B, C or D.\n\n"
    "{question}\n{options}"
)
GSM8K_PROMPT = (
    "Solve the following math problem. Think step by step, then end your reply with a final "
    "line of the form 'Answer: N' where N is just the final number.\n\n{question}"
)


def load_mmlu(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    items = []
    for i in idx:
        r = ds[i]
        options = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(r["choices"]))
        items.append({
            "id": f"mmlu-{i}",
            "dataset": "mmlu",
            "subject": r["subject"],
            "question": r["question"],
            "prompt": MMLU_PROMPT.format(question=r["question"], options=options),
            "gold": LETTERS[r["answer"]],
        })
    return items


def load_gsm8k(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    items = []
    for i in idx:
        r = ds[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "id": f"gsm8k-{i}",
            "dataset": "gsm8k",
            "subject": "math",
            "question": r["question"],
            "prompt": GSM8K_PROMPT.format(question=r["question"]),
            "gold": gold,
        })
    return items


# ---------- grading ----------

_MMLU_STRICT = re.compile(r"Answer:\s*\**\(?([ABCD])\b")
_MMLU_LOOSE = re.compile(r"answer\s+is\s*:?\s*\**\(?([ABCD])\b", re.I)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def grade(item: dict, response: str | None) -> bool:
    if not response:
        return False
    if item["dataset"] == "mmlu":
        hits = _MMLU_STRICT.findall(response) or _MMLU_LOOSE.findall(response)
        return bool(hits) and hits[-1] == item["gold"]
    tail = response.split("Answer:")[-1] if "Answer:" in response else response
    nums = _NUM.findall(tail)
    if not nums:
        return False
    try:
        return abs(float(nums[-1].replace(",", "")) - float(item["gold"])) < 1e-6
    except ValueError:
        return False


def save_items(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n")


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
