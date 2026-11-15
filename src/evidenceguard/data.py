import json
import random
from collections import Counter
from .config import DATA

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]


def load_split(split: str):
    """Load an AVeriTeC split ('train' or 'dev') from DATA/averitec/."""
    path = DATA / "averitec" / f"{split}.json"
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_counts(rows):
    return dict(Counter(r["label"] for r in rows))


def profile(rows):
    """Per-label size of the question/answer annotation and answer types."""
    out = {}
    for lab in LABELS:
        sub = [r for r in rows if r["label"] == lab]
        if not sub:
            continue
        nq = [len(r["questions"]) for r in sub]
        na = [sum(len(q["answers"]) for q in r["questions"]) for r in sub]
        out[lab] = {
            "claims": len(sub),
            "avg_questions": round(sum(nq) / len(sub), 2),
            "avg_answers": round(sum(na) / len(sub), 2),
        }
    out["answer_types"] = dict(Counter(
        a["answer_type"] for r in rows for q in r["questions"] for a in q["answers"]))
    return out


def annotation_urls(row):
    """URLs the annotators used for this claim (source_url and archived cached_source_url)."""
    return {a[k] for q in row["questions"] for a in q["answers"]
            for k in ("source_url", "cached_source_url") if a.get(k)}


def select_dev_subset(rows, seed=42, n_nei=35, n_cp=38, n_other=27):
    """Pick claims: all NEI, all Conflicting, and n_other Supported/Refuted (proportional).

    Returns a list of {"claim_id": index in dev.json, "label": label}, sorted by claim_id.
    """
    rng = random.Random(seed)
    by = {lab: [i for i, r in enumerate(rows) if r["label"] == lab] for lab in LABELS}
    nei, cp = by["Not Enough Evidence"], by["Conflicting Evidence/Cherrypicking"]
    assert len(nei) >= n_nei and len(cp) >= n_cp
    pick = rng.sample(nei, n_nei) + rng.sample(cp, n_cp)
    ref, sup = by["Refuted"], by["Supported"]
    n_ref = round(n_other * len(ref) / (len(ref) + len(sup)))
    pick += rng.sample(ref, n_ref) + rng.sample(sup, n_other - n_ref)
    return [{"claim_id": i, "label": rows[i]["label"]} for i in sorted(pick)]
