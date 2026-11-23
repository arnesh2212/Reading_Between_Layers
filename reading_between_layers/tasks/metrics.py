"""Shared metric implementations and evaluation helpers across task types.

These are thin wrappers around scikit-learn (classification, regression,
clustering) and ``seqeval`` (token classification) plus small native
implementations for ranking metrics (MRR, Recall@k, nDCG) that we use during
the in-batch retrieval evaluation in phase 1.

The wider HF ``evaluate`` library wraps the same backends; phase 1 calls them
directly to avoid the dataset-download side effect.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """Detach and convert a tensor or numpy array to a 1-D or 2-D numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def accuracy(predictions: Tensor | np.ndarray, labels: Tensor | np.ndarray) -> float:
    """Plain top-1 accuracy."""
    p = _to_numpy(predictions)
    y = _to_numpy(labels)
    return float((p == y).mean()) if len(y) else 0.0


def macro_f1(predictions: Tensor | np.ndarray, labels: Tensor | np.ndarray) -> float:
    """Unweighted mean of per-class F1, computed via scikit-learn."""
    from sklearn.metrics import f1_score

    p = _to_numpy(predictions)
    y = _to_numpy(labels)
    return float(f1_score(y, p, average="macro", zero_division=0))


def expected_calibration_error(
    probabilities: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    num_bins: int = 15,
) -> float:
    """Standard binned expected calibration error.

    ``probabilities`` may be shape ``(n,)`` (binary) or ``(n, num_classes)``
    (multi-class). Returns the absolute deviation between bin confidence and
    bin accuracy, averaged with bin frequency weights.
    """
    probs = _to_numpy(probabilities)
    y = _to_numpy(labels)
    if probs.ndim == 1:
        confidences = probs
        preds = (probs >= 0.5).astype(int)
    else:
        confidences = probs.max(axis=1)
        preds = probs.argmax(axis=1)
    accuracies = (preds == y).astype(float)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        in_bin = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(confidences[in_bin].mean())
        bin_acc = float(accuracies[in_bin].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
    return float(ece)


def pearson(predictions: Tensor | np.ndarray, labels: Tensor | np.ndarray) -> float:
    """Pearson product-moment correlation."""
    from scipy.stats import pearsonr

    p = _to_numpy(predictions).astype(np.float64)
    y = _to_numpy(labels).astype(np.float64)
    if len(p) < 2 or np.allclose(p, p[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(pearsonr(p, y).statistic)


def spearman(predictions: Tensor | np.ndarray, labels: Tensor | np.ndarray) -> float:
    """Spearman rank correlation."""
    from scipy.stats import spearmanr

    p = _to_numpy(predictions).astype(np.float64)
    y = _to_numpy(labels).astype(np.float64)
    if len(p) < 2 or np.allclose(p, p[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(spearmanr(p, y).statistic)


def mean_squared_error(
    predictions: Tensor | np.ndarray, labels: Tensor | np.ndarray
) -> float:
    """Plain MSE."""
    p = _to_numpy(predictions).astype(np.float64)
    y = _to_numpy(labels).astype(np.float64)
    if len(y) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def v_measure(
    cluster_assignments: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
) -> float:
    """V-measure from scikit-learn."""
    from sklearn.metrics import v_measure_score

    return float(v_measure_score(_to_numpy(labels), _to_numpy(cluster_assignments)))


def normalized_mutual_info(
    cluster_assignments: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
) -> float:
    """Normalized mutual information from scikit-learn."""
    from sklearn.metrics import normalized_mutual_info_score

    return float(
        normalized_mutual_info_score(_to_numpy(labels), _to_numpy(cluster_assignments))
    )


def adjusted_rand_index(
    cluster_assignments: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
) -> float:
    """Adjusted Rand index from scikit-learn."""
    from sklearn.metrics import adjusted_rand_score

    return float(
        adjusted_rand_score(_to_numpy(labels), _to_numpy(cluster_assignments))
    )


def token_classification_f1(
    predictions: list[list[str]],
    labels: list[list[str]],
) -> float:
    """Sequence-level F1 from ``seqeval`` (BIO-aware).

    Raises:
        ImportError: if ``seqeval`` is not installed; the dependency is listed
            in ``pyproject.toml`` and is expected in the project environment.
    """
    try:
        from seqeval.metrics import f1_score as seqeval_f1
    except ImportError as exc:
        raise ImportError(
            "token_classification_f1 requires the seqeval package"
        ) from exc
    return float(seqeval_f1(labels, predictions))


def mrr_at_k(scores: Tensor | np.ndarray, k: int = 10) -> float:
    """Mean reciprocal rank at ``k`` assuming the diagonal is the true positive.

    ``scores`` has shape ``(n, n)``. For each row, we rank columns by score and
    report the reciprocal rank of column ``i`` if it falls within the top ``k``,
    else zero, averaged over rows.
    """
    s = _to_numpy(scores)
    n = s.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-s, axis=1)
    ranks = np.zeros(n, dtype=np.float64)
    for i in range(n):
        match = np.where(order[i] == i)[0]
        if len(match) and match[0] < k:
            ranks[i] = 1.0 / (match[0] + 1)
    return float(ranks.mean())


def recall_at_k(scores: Tensor | np.ndarray, k: int = 100) -> float:
    """Recall at ``k`` over the diagonal-positive matrix."""
    s = _to_numpy(scores)
    n = s.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-s, axis=1)
    hits = 0
    for i in range(n):
        if i in order[i, :k]:
            hits += 1
    return float(hits / n)


def ndcg_at_k(scores: Tensor | np.ndarray, k: int = 10) -> float:
    """Normalized DCG at ``k`` with a single relevant document per query."""
    s = _to_numpy(scores)
    n = s.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-s, axis=1)
    total = 0.0
    for i in range(n):
        match = np.where(order[i] == i)[0]
        if len(match) and match[0] < k:
            total += 1.0 / math.log2(match[0] + 2)
    return float(total / n)


def perplexity(log_probs: Tensor | np.ndarray) -> float:
    """Perplexity of a sequence of per-token log probabilities."""
    lp = _to_numpy(log_probs).astype(np.float64)
    if len(lp) == 0:
        return float("inf")
    return float(math.exp(-float(lp.mean())))


def exact_match(predictions: list[str], labels: list[str]) -> float:
    """Fraction of exact string matches."""
    if not labels:
        return 0.0
    return float(sum(p == y for p, y in zip(predictions, labels, strict=True)) / len(labels))


def qa_token_f1(predictions: list[str], labels: list[str]) -> float:
    """Token-overlap F1 averaged over the dataset (SQuAD-style)."""
    if not labels:
        return 0.0
    f1s: list[float] = []
    for pred, gold in zip(predictions, labels, strict=True):
        pred_tokens = pred.split()
        gold_tokens = gold.split()
        if not pred_tokens or not gold_tokens:
            f1s.append(float(pred_tokens == gold_tokens))
            continue
        common: dict[str, int] = {}
        for tok in pred_tokens:
            common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
        num_same = sum(common.values())
        if num_same == 0:
            f1s.append(0.0)
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1s.append(2 * precision * recall / (precision + recall))
    return float(sum(f1s) / len(f1s))
