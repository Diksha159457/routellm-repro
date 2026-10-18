"""Routers from RouteLLM (Ong et al., 2024), adapted to a correctness-labelled setting.

Every router learns from query embeddings X and per-query correctness of the
weak and strong models, and returns score(X) = estimated P(strong model needed).

  RandomRouter         baseline; APGR ~ 0.5 in expectation
  SWRankingRouter      similarity-weighted ranking: a Bradley-Terry fit where training
                       comparisons are weighted by embedding similarity to the query.
                       With two models this reduces to a similarity-weighted win rate.
  MatrixFactorization  bilinear scoring s(m, q) = w^T (v_m * (W q + b)),
                       P(strong wins) = sigmoid(s(strong, q) - s(weak, q)). PyTorch.
  ClassifierRouter     logistic regression on embeddings (stand-in for the paper's BERT classifier)
  OracleRouter         uses the true test labels; upper bound, not a real router
"""
from __future__ import annotations

import numpy as np


def _decisive(weak, strong):
    """Comparisons where exactly one model is right; label 1 if the strong one won."""
    weak, strong = np.asarray(weak), np.asarray(strong)
    mask = weak != strong
    return mask, strong[mask].astype(float)


def _unit(X):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


class RandomRouter:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, weak, strong):
        return self

    def score(self, X):
        return self.rng.random(len(X))


class OracleRouter:
    name = "oracle"

    def __init__(self, weak_test, strong_test):
        self.needed = (np.asarray(strong_test) == 1) & (np.asarray(weak_test) == 0)
        self.hurts = (np.asarray(strong_test) == 0) & (np.asarray(weak_test) == 1)

    def fit(self, X, weak, strong):
        return self

    def score(self, X):
        return self.needed.astype(float) - self.hurts.astype(float)


class SWRankingRouter:
    name = "sw_ranking"

    def __init__(self, gamma: float = 10.0):
        self.gamma = gamma

    def fit(self, X, weak, strong):
        mask, self.wins = _decisive(weak, strong)
        self.Xd = _unit(X)[mask]
        self.prior = self.wins.mean() if len(self.wins) else 0.5
        return self

    def score(self, X):
        if len(self.wins) == 0:
            return np.full(len(X), self.prior)
        sims = _unit(X) @ self.Xd.T                       # cosine similarity, shape (n_test, n_train)
        w = np.exp(self.gamma * (sims - sims.max(axis=1, keepdims=True)))
        return (w @ self.wins + self.prior) / (w.sum(axis=1) + 1.0)  # +1 pseudo-count toward the prior


class ClassifierRouter:
    name = "classifier"

    def __init__(self, C: float = 1.0, seed: int = 0):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, X, weak, strong):
        y = ((np.asarray(strong) == 1) & (np.asarray(weak) == 0)).astype(int)
        self.constant = None
        if y.min() == y.max():
            self.constant = float(y[0])
        else:
            self.clf.fit(_unit(X), y)
        return self

    def score(self, X):
        if self.constant is not None:
            return np.full(len(X), self.constant)
        return self.clf.predict_proba(_unit(X))[:, 1]


class MatrixFactorizationRouter:
    name = "matrix_factorization"

    def __init__(self, dim: int = 64, epochs: int = 300, lr: float = 3e-3,
                 weight_decay: float = 1e-4, seed: int = 0):
        self.dim, self.epochs, self.lr, self.wd, self.seed = dim, epochs, lr, weight_decay, seed

    def _build(self, in_dim: int):
        import torch
        from torch import nn

        class MF(nn.Module):
            def __init__(self, in_dim, dim):
                super().__init__()
                self.proj = nn.Linear(in_dim, dim)               # W q + b
                self.models = nn.Embedding(2, dim)               # v_weak, v_strong
                self.head = nn.Linear(dim, 1, bias=False)        # w

            def forward(self, q):
                h = self.proj(q)
                s_weak = self.head(self.models.weight[0] * h).squeeze(-1)
                s_strong = self.head(self.models.weight[1] * h).squeeze(-1)
                return s_strong - s_weak                         # logit of P(strong wins)

        torch.manual_seed(self.seed)
        return MF(in_dim, self.dim)

    def fit(self, X, weak, strong):
        import torch

        mask, wins = _decisive(weak, strong)
        X = torch.tensor(_unit(X)[mask])
        y = torch.tensor(wins, dtype=torch.float32)
        self.model = self._build(X.shape[1])
        if len(y) == 0:
            return self
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(self.epochs):  # full-batch: the decisive set is small
            opt.zero_grad()
            loss = loss_fn(self.model(X), y)
            loss.backward()
            opt.step()
        return self

    def score(self, X):
        import torch

        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(_unit(X)))).numpy()
