import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import grade  # noqa: E402
from routers import ClassifierRouter, OracleRouter, RandomRouter, SWRankingRouter  # noqa: E402
from routing_metrics import apgr, cpt, routing_curve  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


def synthetic(n=2000, d=32, seed=0):
    """Queries whose 'difficulty' lies along one embedding direction.
    Hard queries: weak usually wrong, strong usually right."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    hard = X[:, 0] > 0.3
    weak = np.where(hard, rng.random(n) < 0.2, rng.random(n) < 0.85).astype(int)
    strong = np.where(hard, rng.random(n) < 0.85, rng.random(n) < 0.9).astype(int)
    return X, weak, strong


def run(router, X, weak, strong):
    tr, te = slice(0, 1000), slice(1000, None)
    router.fit(X[tr], weak[tr], strong[tr])
    return apgr(routing_curve(router.score(X[te]), weak[te], strong[te]))


def test_curve_endpoints():
    weak, strong = np.array([1, 0, 0, 1]), np.array([1, 1, 1, 0])
    c = routing_curve(np.zeros(4), weak, strong)
    assert c["accuracy"][0] == pytest.approx(weak.mean())
    assert c["accuracy"][-1] == pytest.approx(strong.mean())
    assert c["pgr"][0] == pytest.approx(0.0) and c["pgr"][-1] == pytest.approx(1.0)


def test_random_router_apgr_near_half():
    X, weak, strong = synthetic()
    vals = [run(RandomRouter(s), X, weak, strong) for s in range(10)]
    assert 0.4 < np.mean(vals) < 0.6


def test_learned_routers_beat_random_and_oracle_beats_all():
    X, weak, strong = synthetic()
    rand = np.mean([run(RandomRouter(s), X, weak, strong) for s in range(5)])
    sw = run(SWRankingRouter(gamma=5.0), X, weak, strong)
    clf = run(ClassifierRouter(), X, weak, strong)
    oracle = run(OracleRouter(weak[1000:], strong[1000:]), X, weak, strong)
    assert sw > rand + 0.1 and clf > rand + 0.1
    assert oracle > max(sw, clf)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_matrix_factorization_learns():
    from routers import MatrixFactorizationRouter

    X, weak, strong = synthetic()
    rand = np.mean([run(RandomRouter(s), X, weak, strong) for s in range(5)])
    assert run(MatrixFactorizationRouter(epochs=200), X, weak, strong) > rand + 0.1


def test_cpt_monotone_targets():
    X, weak, strong = synthetic()
    c = routing_curve(OracleRouter(weak, strong).score(X), weak, strong)
    assert cpt(c, 0.5) <= cpt(c, 0.8)


def test_grading():
    mmlu = {"dataset": "mmlu", "gold": "C"}
    assert grade(mmlu, "Reasoning...\nAnswer: C")
    assert grade(mmlu, "first Answer: A then corrected. Answer: **C**")
    assert not grade(mmlu, "the answer is a tricky one, Answer: B")
    assert not grade(mmlu, None)
    gsm = {"dataset": "gsm8k", "gold": "1200"}
    assert grade(gsm, "So 3 * 400 = 1,200\nAnswer: 1,200")
    assert grade(gsm, "total is 1200.0")
    assert not grade(gsm, "Answer: 120")
