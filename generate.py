"""Step 1: sample benchmark items and collect responses from a weak and a strong model.

Works with any OpenAI-compatible endpoint. Defaults target Groq (free tier):
weak = allam-2-7b, strong = openai/gpt-oss-120b (a reasoning model, run at low effort).
    export OPENAI_API_KEY=<your groq key>
    python generate.py --n_mmlu 1000 --n_gsm8k 500

Responses are cached in data/responses/<model>.jsonl and the script resumes
where it stopped, so rate-limit interruptions are safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data import load_gsm8k, load_items, load_mmlu, save_items

DATA = Path("data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"))
    p.add_argument("--weak", default="allam-2-7b")
    p.add_argument("--strong", default="openai/gpt-oss-120b")
    p.add_argument("--n_mmlu", type=int, default=1000)
    p.add_argument("--n_gsm8k", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=1536,
                   help="reasoning models spend part of this on hidden thinking, so keep it generous")
    p.add_argument("--reasoning_effort", default="low", choices=["low", "medium", "high", "none"],
                   help="sent only to reasoning models (see --reasoning_models); 'none' disables it")
    p.add_argument("--reasoning_models", default="gpt-oss,qwen3",
                   help="comma-separated name fragments that mark a model as a reasoning model")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def safe_name(model: str) -> str:
    return model.replace("/", "__")


def extra_params(model: str, args) -> dict:
    fragments = [f.strip() for f in args.reasoning_models.split(",") if f.strip()]
    if args.reasoning_effort != "none" and any(f in model for f in fragments):
        return {"reasoning_effort": args.reasoning_effort}
    return {}


def call_with_retry(client, model: str, prompt: str, max_tokens: int, extra: dict, tries: int = 8) -> dict:
    for attempt in range(tries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body=extra or None,
            )
            usage = r.usage
            choice = r.choices[0]
            return {
                # empty string (not None) when the token budget ran out: graded as wrong, not retried
                "response": choice.message.content or "",
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        except Exception as e:  # rate limits, timeouts, transient 5xx
            wait = min(60, 2 ** attempt) + random.random()
            print(f"[{model}] {type(e).__name__}: {e} -> retry in {wait:.0f}s")
            time.sleep(wait)
    return {"response": None, "error": "gave up after retries"}


def run_model(client, model: str, items: list[dict], args) -> None:
    out = DATA / "responses" / f"{safe_name(model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        # skip finished answers; failed calls and cut-off answers are retried (the newest line wins)
        recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        done = {r["id"] for r in recs if r.get("response") is not None and r.get("finish_reason") != "length"}
    todo = [x for x in items if x["id"] not in done]
    extra = extra_params(model, args)
    print(f"{model}: {len(done)} cached, {len(todo)} to go  extra params: {extra or 'none'}")
    lock = threading.Lock()
    with out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(call_with_retry, client, model, x["prompt"], args.max_tokens, extra): x for x in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = {"id": futures[fut]["id"], "model": model, **fut.result()}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if n % 50 == 0:
                print(f"{model}: {n}/{len(todo)}")


def main() -> None:
    from openai import OpenAI

    args = parse_args()
    items_path = DATA / "items.jsonl"
    if items_path.exists():
        items = load_items(items_path)
        print(f"Reusing {len(items)} items from {items_path}")
    else:
        items = load_mmlu(args.n_mmlu, args.seed) + load_gsm8k(args.n_gsm8k, args.seed)
        save_items(items, items_path)
    # written up front so build_labels.py works even if this run is interrupted
    (DATA / "models.json").write_text(json.dumps({"weak": args.weak, "strong": args.strong}, indent=2))
    client = OpenAI(base_url=args.base_url, api_key=os.environ["OPENAI_API_KEY"])
    for model in (args.weak, args.strong):
        run_model(client, model, items, args)


if __name__ == "__main__":
    main()