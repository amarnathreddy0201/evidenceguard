"""Score a baseline run against the hand-labelled golden set: does it abstain when it should?

  python scripts/eval_abstention.py --name baseline_q3b

Uses outputs/golden/golden_dev150_verified.jsonl (expected_action = VERDICT or ABSTAIN) and the baseline
predictions. The baseline "abstains" when it predicts Not Enough Evidence (or returns nothing parseable).
Only claims present in both files are scored. Reports all claims, then tune and test separately.
Writes outputs/baseline/<name>_abstention.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.config import OUT
from evidenceguard.io import save_json

NEI = "Not Enough Evidence"


def rate(num, den):
    return round(num / den, 3) if den else None


def stats(items):
    """items: list of (should_abstain, abstained, correct_when_answered)."""
    sa = [i for i in items if i[0]]
    sv = [i for i in items if not i[0]]
    abst = [i for i in items if i[1]]
    ans = [i for i in items if not i[1]]
    caught = sum(1 for i in sa if i[1])
    bad = sum(1 for i in ans if i[0] or not i[2])  # answered when it should abstain, or answered wrongly
    return {
        "claims": len(items), "should_abstain": len(sa), "should_verdict": len(sv),
        "abstain_recall": rate(caught, len(sa)),                          # caught the claims it should refuse
        "abstain_precision": rate(caught, len(abst)),                     # its refusals that were right
        "false_abstain_rate": rate(sum(1 for i in sv if i[1]), len(sv)),  # refused a claim it could decide
        "confident_error_rate": rate(sum(1 for i in sa if not i[1]), len(sa)),  # answered when it should refuse
        "coverage": rate(len(ans), len(items)),
        "risk_on_answered": rate(bad, len(ans)),                          # answered claims that were wrong or should not have been answered
        "verdict_accuracy_on_answered_verdict_claims": rate(sum(1 for i in sv if not i[1] and i[2]), sum(1 for i in sv if not i[1])),
    }


def bootstrap(items, key, n=2000, seed=42):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = stats([items[rng.randrange(len(items))] for _ in items])[key]
        if v is not None:
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    preds = {r["claim_id"]: r for r in (json.loads(l) for l in open(OUT / "baseline" / f"{a.name}.jsonl", encoding="utf-8") if l.strip())}
    golden = {r["example_id"]: r for r in (json.loads(l) for l in open(OUT / "golden" / "golden_dev150_verified.jsonl", encoding="utf-8"))}
    sheet_ok = [g for g in golden.values() if g.get("human_verified")]
    if len(sheet_ok) != len(golden):
        print(f"warning: only {len(sheet_ok)} of {len(golden)} golden rows are human-verified")

    by_split = {"all": [], "tune": [], "test": []}
    for cid, p in preds.items():
        g = golden.get(f"avdev-{cid}")
        if not g:
            continue
        abstained = p["pred_label"] in (None, NEI)
        item = (g["expected_action"] == "ABSTAIN", abstained, p["pred_label"] == p["gold_label"])  # baseline file uses AVeriTeC names; golden file uses upper-case schema names
        by_split["all"].append(item)
        by_split[g["split"]].append(item)

    result = {"run": a.name, "baseline_claims": len(preds), "scored_claims": len(by_split["all"]),
              "abstain_rule": "baseline abstains when it predicts Not Enough Evidence or returns no parseable label"}
    for k, items in by_split.items():
        if items:
            result[k] = stats(items)
            result[k]["abstain_recall_ci95"] = bootstrap(items, "abstain_recall")
            result[k]["confident_error_rate_ci95"] = bootstrap(items, "confident_error_rate")
    save_json(result, f"{a.name}_abstention.json", "baseline")

    print(f"{result['scored_claims']} of {len(preds)} baseline claims are in the golden set")
    for k in ("all", "tune", "test"):
        if k in result:
            d = result[k]
            print(f"\n== {k}: {d['claims']} claims ({d['should_abstain']} should abstain, {d['should_verdict']} should give a verdict)")
            print(f"   abstain recall {d['abstain_recall']} {d['abstain_recall_ci95']}   abstain precision {d['abstain_precision']}")
            print(f"   confident-error rate {d['confident_error_rate']} {d['confident_error_rate_ci95']}   false-abstain rate {d['false_abstain_rate']}")
            print(f"   coverage {d['coverage']}   risk on answered {d['risk_on_answered']}   "
                  f"verdict accuracy (answered, should-verdict) {d['verdict_accuracy_on_answered_verdict_claims']}")


if __name__ == "__main__":
    main()
