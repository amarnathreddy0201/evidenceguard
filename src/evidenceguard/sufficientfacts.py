"""Loader for the SufficientFacts dataset (second check of sufficiency detection).

Files live in DATA/sufficient_facts/{fever,vitaminc,hover}_sufficient_facts.jsonl.
Binary task: label_after == "NOT ENOUGH" -> insufficient, both "ENOUGH ..." labels -> sufficient.
"""
import json
import random

from .config import DATA

SOURCES = ("fever", "vitaminc", "hover")


def load_sufficientfacts(sources=SOURCES):
    rows = []
    for src in sources:
        path = DATA / "sufficient_facts" / f"{src}_sufficient_facts.jsonl"
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                evidence = " ".join(" ".join(pair) if isinstance(pair, list) else str(pair)
                                    for pair in r["evidence"])
                rows.append({
                    "id": f"{src}-{i}", "source": src, "claim": r["claim"], "evidence_text": evidence,
                    "sufficient": not r["label_after"].upper().startswith("NOT ENOUGH"),
                    "label_after": r["label_after"], "label_before": r["label_before"],
                })
    return rows


def balanced_sample(rows, n_per_class=300, seed=42):
    """Equal numbers of sufficient and insufficient examples (the full set is about 74% insufficient)."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["sufficient"]]
    neg = [r for r in rows if not r["sufficient"]]
    k = min(n_per_class, len(pos), len(neg))
    out = rng.sample(pos, k) + rng.sample(neg, k)
    rng.shuffle(out)
    return out
