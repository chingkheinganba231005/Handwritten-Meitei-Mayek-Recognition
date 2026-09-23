"""Combining members, pair specialists and error bookkeeping."""

import numpy as np
import pandas as pd
import torch


def accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean())


def fit_weights(probs, labels, steps=400):
    """Ensemble weights that minimise log-loss (softmax-parametrised, so they sum to 1)."""
    P = torch.tensor(np.stack(probs))
    y = torch.as_tensor(labels)
    w = torch.zeros(len(probs), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(steps):
        mix = (w.softmax(0).view(-1, 1, 1) * P).sum(0)
        loss = -torch.log(mix[torch.arange(len(y)), y] + 1e-9).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.softmax(0).detach().numpy()


def combine(probs, weights=None):
    """Weighted (or plain) average of a list of probability arrays."""
    weights = np.full(len(probs), 1 / len(probs)) if weights is None else np.asarray(weights)
    return sum(w * p for w, p in zip(weights, probs))


def apply_specialists(probs, spec, beta):
    """Re-split the probability of a pair when it holds the top two places.

    spec maps (a, b) -> the specialist's P(a). beta = 0 ignores the
    specialists, beta = 1 trusts them fully.
    """
    out = probs.copy()
    if not beta:
        return out
    top2 = np.argsort(-probs, 1)[:, :2]
    for (a, b), q in spec.items():
        hit = ((top2[:, 0] == a) & (top2[:, 1] == b)) | ((top2[:, 0] == b) & (top2[:, 1] == a))
        mass = probs[hit, a] + probs[hit, b]
        ra = probs[hit, a] / mass
        na = q[hit] ** beta * ra ** (1 - beta)
        nb = (1 - q[hit]) ** beta * (1 - ra) ** (1 - beta)
        out[hit, a], out[hit, b] = mass * na / (na + nb), mass * nb / (na + nb)
    return out


def confusions(probs, labels, k=None):
    """Directed confusions (true -> predicted) with counts, most frequent first."""
    pred = probs.argmax(1)
    wrong = pred != labels
    pairs = pd.Series(list(zip(labels[wrong], pred[wrong]))).value_counts()
    if k:
        pairs = pairs.head(k)
    return [(int(a), int(b), int(n)) for (a, b), n in pairs.items()]


def per_class(probs, labels):
    """(accuracy, count) for every class."""
    ok = probs.argmax(1) == labels
    return {int(c): (float(ok[labels == c].mean()), int((labels == c).sum())) for c in np.unique(labels)}


def wilson(correct, n, z=1.959964):
    """95% Wilson score interval for a proportion."""
    p = correct / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)
