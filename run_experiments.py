"""Step 3: train every router, evaluate it with RouteLLM's metrics, and plot the curves.

    python run_experiments.py

Two experiments:
  in_domain     random 50/50 train/test split over all items, repeated over 5 seeds
  cross_domain  train on MMLU, test on GSM8K (does the router generalize to a new task?)

Writes results/results.md, results/results.json and results/pgr_curve_<exp>.png.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from routers import (ClassifierRouter, MatrixFactorizationRouter, OracleRouter, RandomRouter,
                     SWRankingRouter)
from routing_metrics import routing_curve, summarize

ROUTERS = ["random", "sw_ranking", "matrix_factorization", "classifier", "oracle"]


def load_labels(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        "questions": [r["question"] for r in rows],
        "dataset": np.array([r["dataset"] for r in rows]),
        "weak": np.array([int(r["weak_correct"]) for r in rows]),
        "strong": np.array([int(r["strong_correct"]) for r in rows]),
    }


def embed(questions: list[str], model_name: str, cache: Path) -> np.ndarray:
    if cache.exists():
        X = np.load(cache)
        if len(X) == len(questions):
            return X
    from sentence_transformers import SentenceTransformer

    X = SentenceTransformer(model_name).encode(questions, batch_size=64, show_progress_bar=True,
                                               normalize_embeddings=True)
    np.save(cache, X)
    return X


def make_router(name: str, seed: int, weak_test, strong_test):
    return {
        "random": lambda: RandomRouter(seed),
        "sw_ranking": lambda: SWRankingRouter(),
        "matrix_factorization": lambda: MatrixFactorizationRouter(seed=seed),
        "classifier": lambda: ClassifierRouter(seed=seed),
        "oracle": lambda: OracleRouter(weak_test, strong_test),
    }[name]()


def evaluate_split(X, weak, strong, tr, te, seed: int) -> dict:
    results = {}
    for name in ROUTERS:
        router = make_router(name, seed, weak[te], strong[te]).fit(X[tr], weak[tr], strong[tr])
        curve = routing_curve(router.score(X[te]), weak[te], strong[te], rng=np.random.default_rng(seed))
        results[name] = {"summary": summarize(curve), "curve": curve}
    return results


def splits(data: dict, experiment: str, seeds: list[int]):
    n = len(data["weak"])
    if experiment == "in_domain":
        for s in seeds:
            perm = np.random.default_rng(s).permutation(n)
            yield s, perm[: n // 2], perm[n // 2:]
    else:
        tr = np.nonzero(data["dataset"] == "mmlu")[0]
        te = np.nonzero(data["dataset"] == "gsm8k")[0]
        for s in seeds:
            yield s, tr, te


def mean_std(values: list) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    suffix = "" if len(vals) == len(values) else f" ({len(values) - len(vals)} never reached)"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}{suffix}"


def plot(per_router_curves: dict, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in per_router_curves.items():
        style = "--" if name in ("random", "oracle") else "-"
        ax.plot(100 * curve["frac_strong"], 100 * curve["accuracy"], style, label=name)
    any_curve = next(iter(per_router_curves.values()))
    ax.axhline(100 * any_curve["acc_weak"], color="grey", lw=0.8)
    ax.axhline(100 * any_curve["acc_strong"], color="grey", lw=0.8)
    ax.set_xlabel("% of queries sent to the strong model")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/labels.csv")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    data = load_labels(Path(args.labels))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = embed(data["questions"], args.embedder, Path(args.labels).with_suffix(".emb.npy"))

    report = {"n_items": len(data["weak"]), "acc_weak": float(data["weak"].mean()),
              "acc_strong": float(data["strong"].mean()), "experiments": {}}
    md = [f"# Routing results\n",
          f"Items: {report['n_items']}  |  weak acc: {report['acc_weak']:.1%}  |  "
          f"strong acc: {report['acc_strong']:.1%}\n"]

    for exp in ("in_domain", "cross_domain"):
        if exp == "cross_domain" and not {"mmlu", "gsm8k"} <= set(data["dataset"]):
            continue
        runs = [evaluate_split(X, data["weak"], data["strong"], tr, te, s)
                for s, tr, te in splits(data, exp, args.seeds)]
        table = {}
        md += [f"\n## {exp.replace('_', ' ')}  ({len(runs)} seeds)\n",
               "| Router | APGR ↑ | CPT(50%) ↓ | CPT(80%) ↓ | Acc @ 50% strong |",
               "|---|---|---|---|---|"]
        for name in ROUTERS:
            summ = [r[name]["summary"] for r in runs]
            table[name] = {k: [s[k] for s in summ] for k in summ[0]}
            md.append(f"| {name} | {mean_std(table[name]['APGR'])} | {mean_std(table[name]['CPT50'])} | "
                      f"{mean_std(table[name]['CPT80'])} | {mean_std(table[name]['acc_at_50pct_strong'])} |")
        report["experiments"][exp] = table
        plot({n: runs[0][n]["curve"] for n in ROUTERS}, f"Quality vs. strong-model calls ({exp})",
             out / f"pgr_curve_{exp}.png")

    (out / "results.json").write_text(json.dumps(report, indent=2))
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
