"""Build the golden evaluation set from AVeriTeC dev.

Run from the repo root:  python scripts/build_golden.py
Writes to <ROOT>/outputs/golden/:
  dev_split.json         every dev claim -> tune/test (stratified, seed 42)
  golden_dev150.jsonl    150 records in the EvidenceGuard schema (derived labels, human_verified=false)
  review_sheet.csv       blind sheet for hand labelling (no verdict, no justification shown)
  golden_summary.json    counts
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.config import OUT, SEED, ensure_dirs
from evidenceguard.data import load_split
from evidenceguard.golden import evidence_items, golden_record, pick_golden_ids, split_dev
from evidenceguard.io import save_json, save_jsonl


def main():
    ensure_dirs()
    dev = load_split("dev")
    split = split_dev(dev, seed=SEED)
    save_json({"seed": SEED, "test_fraction": 0.5,
               "split": {str(i): s for i, s in sorted(split.items())}}, "dev_split.json", "golden")

    w2 = OUT / "week2" / "dev100_ids.json"
    week2_ids = [c["claim_id"] for c in json.loads(w2.read_text(encoding="utf-8"))["claims"]] if w2.exists() else []
    ids = pick_golden_ids(dev, week2_ids, seed=SEED)
    records = [golden_record(i, dev[i], split[i]) for i in ids]
    save_jsonl(records, "golden_dev150.jsonl", "golden")

    # Blind review sheet: the reviewer sees the claim and the annotators' answers, not the verdict.
    sheet = OUT / "golden" / "review_sheet.csv"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "claim", "claim_date", "speaker", "evidence (question | answer | type | url)",
                    "evidence_sufficiency (SUFFICIENT/INSUFFICIENT/CONFLICTING)",
                    "evidence_conflict (NONE/MINOR/MATERIAL)", "expected_action (VERDICT/ABSTAIN)",
                    "reviewer", "notes"])
        for i in ids:
            row = dev[i]
            # "•" not "-": Excel treats a cell starting with "-" or "=" as a formula and refuses edits.
            ev = "\n".join(f"• {e['question']} | {e['quote']} | {e['answer_type']} | {e['source_url']}"
                           for e in evidence_items(row, 400))
            w.writerow([f"avdev-{i}", row["claim"], row.get("claim_date"), row.get("speaker"), ev, "", "", "", "", ""])

    summary = {
        "seed": SEED,
        "dev_split_counts": {s: dict(Counter(dev[i]["label"] for i, v in split.items() if v == s))
                             for s in ("tune", "test")},
        "golden150_counts": dict(Counter(r["gold_label"] for r in records)),
        "golden150_by_split": dict(Counter(r["split"] for r in records)),
        "golden150_by_split_label": {s: dict(Counter(r["gold_label"] for r in records if r["split"] == s))
                                     for s in ("tune", "test")},
        "week2_claims_included": len(set(week2_ids) & set(ids)),
    }
    save_json(summary, "golden_summary.json", "golden")
    print(json.dumps(summary, indent=1))
    print("wrote to", OUT / "golden")


if __name__ == "__main__":
    main()
