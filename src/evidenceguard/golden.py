"""Golden evaluation set built from AVeriTeC dev.

Two layers:
  * dev_split: every dev claim is assigned to "tune" or "test" (stratified by label, fixed seed).
    Thresholds, prompts and the round budget are chosen on "tune"; "test" is scored once at the end.
  * golden150: 150 dev claims (all NEI, all Conflicting, plus Supported/Refuted) for hand labelling of
    evidence sufficiency and conflict. Derived labels are weak until a human has checked them.
"""
import random
from .data import LABELS

SCHEMA_LABEL = {
    "Supported": "SUPPORTED",
    "Refuted": "REFUTED",
    "Not Enough Evidence": "NOT_ENOUGH_EVIDENCE",
    "Conflicting Evidence/Cherrypicking": "CONFLICTING_EVIDENCE",
}
# Weak mapping from the verdict label to the Auditor's labels (to be checked by hand).
DERIVED = {
    "Supported": ("SUFFICIENT", "NONE", "VERDICT"),
    "Refuted": ("SUFFICIENT", "NONE", "VERDICT"),
    "Not Enough Evidence": ("INSUFFICIENT", "NONE", "ABSTAIN"),
    "Conflicting Evidence/Cherrypicking": ("CONFLICTING", "MATERIAL", "ABSTAIN"),
}


def split_dev(rows, seed=42, test_fraction=0.5):
    """Assign each dev claim index to 'tune' or 'test', stratified by label."""
    rng = random.Random(seed)
    out = {}
    for lab in LABELS:
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        rng.shuffle(idx)
        n_test = round(len(idx) * test_fraction)
        for j, i in enumerate(idx):
            out[i] = "test" if j < n_test else "tune"
    return out


def pick_golden_ids(rows, week2_ids, seed=42, n_total=150):
    """All NEI + all Conflicting + Supported/Refuted up to n_total (week-2 claims first)."""
    rng = random.Random(seed)
    by = {lab: [i for i, r in enumerate(rows) if r["label"] == lab] for lab in LABELS}
    chosen = set(by["Not Enough Evidence"]) | set(by["Conflicting Evidence/Cherrypicking"])
    sr = [i for i in by["Refuted"] + by["Supported"]]
    chosen |= {i for i in week2_ids if rows[i]["label"] in ("Refuted", "Supported")}
    pool = [i for i in sr if i not in chosen]
    rng.shuffle(pool)
    need = n_total - len(chosen)
    chosen |= set(pool[:max(need, 0)])
    return sorted(chosen)


def evidence_items(row, max_chars=600):
    """Annotator answers as evidence items: (question, answer, answer_type, url)."""
    items = []
    for qi, q in enumerate(row["questions"]):
        for ai, a in enumerate(q["answers"]):
            items.append({
                "evidence_id": f"q{qi}a{ai}",
                "question": q["question"],
                "quote": (a.get("answer") or "")[:max_chars],
                "answer_type": a.get("answer_type"),
                "source_url": a.get("source_url"),
                "cached_source_url": a.get("cached_source_url"),
            })
    return items


def coverage(row):
    """Share of annotator questions that have at least one answer that is not 'Unanswerable'."""
    qs = row["questions"]
    if not qs:
        return 0.0
    ok = sum(1 for q in qs if any(a.get("answer_type") != "Unanswerable" for a in q["answers"]))
    return round(ok / len(qs), 3)


def golden_record(i, row, split):
    suff, conf, action = DERIVED[row["label"]]
    return {
        "schema_version": "1.0",
        "example_id": f"avdev-{i}",
        "dev_index": i,
        "is_synthetic": False,
        "split": split,
        "claim": row["claim"],
        "claim_date": row.get("claim_date"),
        "speaker": row.get("speaker"),
        "gold_label": SCHEMA_LABEL[row["label"]],
        "gold_label_source": "averitec_annotators",
        "evidence_sufficiency": suff,
        "evidence_conflict": conf,
        "expected_action": action,
        "evidence_coverage": coverage(row),
        "sufficiency_label_source": "derived_from_verdict",
        "human_verified": False,
        "evidence": evidence_items(row),
        "gold_rationale": row.get("justification"),
    }
