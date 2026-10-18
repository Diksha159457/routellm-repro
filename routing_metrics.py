"""RouteLLM evaluation metrics.

A router assigns each query a score; higher means "send to the strong model".
Sweeping how many queries go to the strong model traces a quality-vs-cost curve.

  PGR(c)  performance gap recovered at strong-call fraction c:
          (acc_router(c) - acc_weak) / (acc_strong - acc_weak)
  APGR    area under the PGR curve over c in [0, 1]. A random router scores ~0.5.
  CPT(x)  smallest strong-call fraction that reaches PGR >= x (lower is better).
"""
from __future__ import annotations

import numpy as np


def routing_curve(scores, weak_correct, strong_correct, n_points: int = 101, rng=None) -> dict:
    scores = np.asarray(scores, dtype=float)
    weak = np.asarray(weak_correct, dtype=float)
    strong = np.asarray(strong_correct, dtype=float)
    n = len(scores)
    rng = rng or np.random.default_rng(0)

    # Highest score routed to strong first; random tie-breaking so constant scores act like random.
    order = np.lexsort((rng.random(n), -scores))
    # gain[k] = accuracy change from sending the top-k queries to the strong model
    gain = np.concatenate([[0.0], np.cumsum((strong - weak)[order])])

    fracs = np.linspace(0.0, 1.0, n_points)
    ks = np.round(fracs * n).astype(int)
    acc = (weak.sum() + gain[ks]) / n

    acc_weak, acc_strong = weak.mean(), strong.mean()
    gap = acc_strong - acc_weak
    pgr = (acc - acc_weak) / gap if gap > 1e-12 else np.full_like(acc, np.nan)
    return {"frac_strong": fracs, "accuracy": acc, "pgr": pgr,
            "acc_weak": float(acc_weak), "acc_strong": float(acc_strong)}


def apgr(curve: dict) -> float:
    pgr, x = curve["pgr"], curve["frac_strong"]
    # trapezoid rule, written out to avoid numpy's trapz/trapezoid rename
    return float(np.sum((pgr[1:] + pgr[:-1]) / 2 * np.diff(x)))


def cpt(curve: dict, target: float) -> float | None:
    hit = np.nonzero(curve["pgr"] >= target - 1e-9)[0]
    return float(curve["frac_strong"][hit[0]]) if len(hit) else None


def summarize(curve: dict) -> dict:
    return {
        "APGR": apgr(curve),
        "CPT50": cpt(curve, 0.5),
        "CPT80": cpt(curve, 0.8),
        "acc_at_50pct_strong": float(np.interp(0.5, curve["frac_strong"], curve["accuracy"])),
    }
