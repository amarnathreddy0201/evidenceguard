"""Evaluation metrics, in pure Python (no numpy needed)."""
import random


def gold_recall_at_k(ranked_urls, gold_urls, ks=(8, 50, 200)):
    """How many gold URLs appear among the URLs of the top-k passages.

    ranked_urls: URL of each retrieved passage, in rank order (repeats allowed).
    Returns {k: (found, total)} so counts can be pooled across claims.
    """
    out = {}
    for k in ks:
        top = set(ranked_urls[:k])
        out[k] = (len(top & gold_urls), len(gold_urls))
    return out


def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def confusion(y_true, y_pred, labels):
    """Rows are gold labels, columns are predictions; an extra column holds unparsed (None) predictions."""
    cols = list(labels) + [None]
    m = {t: {c: 0 for c in cols} for t in labels}
    for t, p in zip(y_true, y_pred):
        m[t][p if p in labels else None] += 1
    return m


def prf(y_true, y_pred, label):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def per_class(y_true, y_pred, labels):
    out = {}
    for lab in labels:
        p, r, f = prf(y_true, y_pred, lab)
        out[lab] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                    "support": sum(1 for t in y_true if t == lab)}
    return out


def macro_f1(y_true, y_pred, labels):
    present = [lab for lab in labels if any(t == lab for t in y_true)]
    return sum(prf(y_true, y_pred, lab)[2] for lab in present) / len(present) if present else 0.0


def bootstrap_ci(y_true, y_pred, metric, n=1000, seed=42, alpha=0.05):
    """Percentile bootstrap interval of metric(y_true, y_pred), resampling claims."""
    rng = random.Random(seed)
    idx = list(range(len(y_true)))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(metric([y_true[i] for i in s], [y_pred[i] for i in s]))
    vals.sort()
    return vals[int(n * alpha / 2)], vals[int(n * (1 - alpha / 2)) - 1]


def cohen_kappa(a, b):
    """Cohen's kappa between two label lists of equal length."""
    n = len(a)
    if n == 0:
        return 0.0
    cats = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
