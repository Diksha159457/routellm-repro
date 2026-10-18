"""Step 2: grade both models' responses and write data/labels.csv.

One row per item: weak_correct, strong_correct and token counts.
The routing target is "strong needed" = strong correct AND weak wrong.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from data import grade, load_items
from generate import DATA, safe_name


def load_responses(model: str) -> dict[str, dict]:
    path = DATA / "responses" / f"{safe_name(model)}.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("response") is not None:
                out[rec["id"]] = rec
    return out


def main() -> None:
    models = json.loads((DATA / "models.json").read_text())
    items = load_items(DATA / "items.jsonl")
    weak, strong = load_responses(models["weak"]), load_responses(models["strong"])

    rows, skipped = [], 0
    for it in items:
        w, s = weak.get(it["id"]), strong.get(it["id"])
        if w is None or s is None:
            skipped += 1
            continue
        rows.append({
            "id": it["id"],
            "dataset": it["dataset"],
            "subject": it["subject"],
            "question": it["question"],
            "weak_correct": int(grade(it, w["response"])),
            "strong_correct": int(grade(it, s["response"])),
            "weak_tokens": w.get("completion_tokens") or 0,
            "strong_tokens": s.get("completion_tokens") or 0,
        })

    out = DATA / "labels.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out} ({skipped} items missing a response)")
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        wa = sum(r["weak_correct"] for r in sub) / len(sub)
        sa = sum(r["strong_correct"] for r in sub) / len(sub)
        need = sum(r["strong_correct"] and not r["weak_correct"] for r in sub) / len(sub)
        print(f"  {ds}: n={len(sub)} weak={wa:.1%} strong={sa:.1%} strong-needed={need:.1%}")
    for role, resp in (("weak", weak), ("strong", strong)):
        cut = sum(1 for r in resp.values() if r.get("finish_reason") == "length")
        if cut:
            print(f"  WARNING: {cut} {role} responses hit max_tokens and were cut off; "
                  f"consider re-running generate.py with a larger --max_tokens")


if __name__ == "__main__":
    main()