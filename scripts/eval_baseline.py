"""Score a baseline run: accuracy, per-class F1, confusion matrix, bootstrap intervals, retrieval recall.

  python scripts/eval_baseline.py --name baseline_q3b
Reports all claims, then the tune and test halves separately (see outputs/golden/dev_split.json).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.config import OUT
from evidenceguard.data import LABELS, annotation_urls, load_split
from evidenceguard.io import save_json
from evidenceguard.metrics import accuracy, bootstrap_ci, confusion, macro_f1, per_class


def score(rows, dev):
    yt = [r["gold_label"] for r in rows]
    yp = [r["pred_label"] for r in rows]
    acc_ci = bootstrap_ci(yt, yp, accuracy)
    f1_ci = bootstrap_ci(yt, yp, lambda a, b: macro_f1(a, b, LABELS))
    found = total = 0
    for r in rows:
        gold = annotation_urls(dev[r["claim_id"]])
        urls = {p["url"] for p in r["passages"]}
        found += len(gold & urls)
        total += len(gold)
    return {
        "claims": len(rows),
        "unparsed_predictions": sum(1 for p in yp if p is None),
        "accuracy": round(accuracy(yt, yp), 3), "accuracy_ci95": [round(x, 3) for x in acc_ci],
        "macro_f1": round(macro_f1(yt, yp, LABELS), 3), "macro_f1_ci95": [round(x, 3) for x in f1_ci],
        "per_class": per_class(yt, yp, LABELS),
        "confusion_rows_gold_cols_pred": {k: {str(c): v for c, v in d.items()} for k, d in confusion(yt, yp, LABELS).items()},
        "final_passages_gold_url_recall": round(found / total, 3) if total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(OUT / "baseline" / f"{a.name}.jsonl", encoding="utf-8") if l.strip()]
    dev = load_split("dev")
    split = json.loads((OUT / "golden" / "dev_split.json").read_text("utf-8"))["split"]
    result = {"run": a.name, "llm": rows[0].get("llm") if rows else None, "reranker": rows[0].get("reranker") if rows else None,
              "all": score(rows, dev)}
    for s in ("tune", "test"):
        sub = [r for r in rows if split[str(r["claim_id"])] == s]
        if sub:
            result[s] = score(sub, dev)
    save_json(result, f"{a.name}_scores.json", "baseline")
    for k in ("all", "tune", "test"):
        if k in result:
            d = result[k]
            print(f"\n== {k}: {d['claims']} claims | accuracy {d['accuracy']} {d['accuracy_ci95']} | "
                  f"macro-F1 {d['macro_f1']} {d['macro_f1_ci95']} | unparsed {d['unparsed_predictions']} | "
                  f"final-passage gold recall {d['final_passages_gold_url_recall']}")
            for lab, m in d["per_class"].items():
                print(f"   {lab:38s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f}  (n={m['support']})")
    print("\nsaved", OUT / "baseline" / f"{a.name}_scores.json")


if __name__ == "__main__":
    main()
