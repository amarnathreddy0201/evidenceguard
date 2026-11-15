"""Merge hand labels from the review sheet into the golden set and report agreement.

  python scripts/merge_labels.py                                  # your labels
  python scripts/merge_labels.py --second second_reviewer.csv     # also Cohen's kappa against a second labeller

Reads  outputs/golden/review_sheet.csv (your filled copy) and golden_dev150.jsonl.
Writes outputs/golden/golden_dev150_verified.jsonl and golden_label_report.json.
Rows with a blank or invalid label stay unverified.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.config import OUT
from evidenceguard.io import save_json, save_jsonl
from evidenceguard.metrics import cohen_kappa

SUFF = {"SUFFICIENT", "INSUFFICIENT", "CONFLICTING"}
CONF = {"NONE", "MINOR", "MATERIAL"}
ACT = {"VERDICT", "ABSTAIN"}
COLS = {"suff": "evidence_sufficiency", "conf": "evidence_conflict", "act": "expected_action"}


def read_sheet(path):
    """Return {example_id: {suff, conf, act, reviewer}} for rows with all three labels valid."""
    out, invalid = {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            def cell(prefix):
                key = next(k for k in row if k.startswith(prefix))
                return (row[key] or "").strip().upper()
            suff, conf, act = cell("evidence_sufficiency"), cell("evidence_conflict"), cell("expected_action")
            if not (suff or conf or act):
                continue
            if suff in SUFF and conf in CONF and act in ACT:
                out[row["example_id"]] = {"suff": suff, "conf": conf, "act": act, "reviewer": (row.get("reviewer") or "").strip()}
            else:
                invalid.append(row["example_id"])
    return out, invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(OUT / "golden" / "review_sheet.csv"))
    ap.add_argument("--second", default=None, help="a second reviewer's filled copy of the sheet")
    a = ap.parse_args()

    golden = [json.loads(l) for l in open(OUT / "golden" / "golden_dev150.jsonl", encoding="utf-8")]
    mine, invalid = read_sheet(a.sheet)
    merged, vs_derived = [], Counter()
    for r in golden:
        lab = mine.get(r["example_id"])
        if lab:
            vs_derived[(r["evidence_sufficiency"], lab["suff"])] += 1
            r = dict(r, derived_evidence_sufficiency=r["evidence_sufficiency"], derived_expected_action=r["expected_action"],
                     evidence_sufficiency=lab["suff"], evidence_conflict=lab["conf"], expected_action=lab["act"],
                     sufficiency_label_source="human", human_verified=True, reviewer=lab["reviewer"])
        merged.append(r)
    save_jsonl(merged, "golden_dev150_verified.jsonl", "golden")

    report = {
        "labelled": len(mine), "total": len(golden), "invalid_rows": invalid,
        "derived_vs_human_sufficiency": {f"derived={d} human={h}": n for (d, h), n in sorted(vs_derived.items())},
        "agreement_with_derived": round(sum(n for (d, h), n in vs_derived.items() if d == h) / max(sum(vs_derived.values()), 1), 3),
    }
    if a.second:
        other, inv2 = read_sheet(a.second)
        common = sorted(set(mine) & set(other))
        report["second_reviewer"] = {"claims_in_common": len(common), "invalid_rows": inv2}
        for key, name in COLS.items():
            if common:
                report["second_reviewer"][f"kappa_{name}"] = round(cohen_kappa([mine[i][key] for i in common],
                                                                                [other[i][key] for i in common]), 3)
    save_json(report, "golden_label_report.json", "golden")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
