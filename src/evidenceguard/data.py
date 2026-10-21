import json
from collections import Counter
from .config import DATA

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]


def load_split(split: str):
    """Load an AVeriTeC split ('train' or 'dev') from DATA/averitec/."""
    path = DATA / "averitec" / f"{split}.json"
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
