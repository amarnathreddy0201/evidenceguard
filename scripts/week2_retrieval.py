"""Week 2: pick 100 dev claims, build BM25 over their knowledge-store files, measure gold-URL recall.

Run from the repo root:  python scripts/week2_retrieval.py
Outputs go to <ROOT>/outputs/week2/.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.config import OUT, SEED, STORE_DEV, ensure_dirs, set_seed
from evidenceguard.data import annotation_urls, load_split, select_dev_subset
from evidenceguard.io import save_json, save_jsonl
from evidenceguard.metrics import gold_recall_at_k
from evidenceguard.retrieve import BM25, build_passages

KS = (8, 50, 200)
TOP_SAVE = 50
MAX_WORDS = 150


def main():
    ensure_dirs()
    set_seed(SEED)
    dev = load_split("dev")
    subset = select_dev_subset(dev, seed=SEED)
    save_json({"seed": SEED, "n": len(subset), "claims": subset}, "dev100_ids.json", "week2")
    print("selected", len(subset), "claims; store:", STORE_DEV)

    rows = []
    pooled = defaultdict(lambda: {k: [0, 0] for k in KS})
    t0 = time.time()
    for n, item in enumerate(subset, 1):
        cid, label = item["claim_id"], item["label"]
        row = dev[cid]
        passages, store_urls = build_passages(STORE_DEV / f"{cid}.json", MAX_WORDS)
        bm25 = BM25([p["text"] for p in passages])
        hits = bm25.search(row["claim"], k=max(KS))
        ranked_urls = [passages[i]["url"] for i, _ in hits]

        gold_all = annotation_urls(row)       # gold = URLs the annotators used (not the store "type")
        gold = gold_all & store_urls          # only gold pages that exist in the store can be found
        rec = gold_recall_at_k(ranked_urls, gold, KS)
        for k in KS:
            for grp in (label, "ALL"):
                pooled[grp][k][0] += rec[k][0]
                pooled[grp][k][1] += rec[k][1]

        rows.append({
            "claim_id": cid, "label": label, "claim": row["claim"],
            "n_documents": len(store_urls), "n_passages": len(passages),
            "gold_urls": sorted(gold), "gold_urls_not_in_store": sorted(gold_all - store_urls),
            "recall": {str(k): list(rec[k]) for k in KS},
            "passages": [{"rank": r + 1, "score": round(s, 3), "url": passages[i]["url"],
                          "text": passages[i]["text"]} for r, (i, s) in enumerate(hits[:TOP_SAVE])],
        })
        if n % 10 == 0 or n == len(subset):
            print(f"{n}/{len(subset)} done, {time.time() - t0:.0f}s", flush=True)

    save_jsonl(rows, "retrieval_dev100.jsonl", "week2")

    summary = {"seed": SEED, "max_words": MAX_WORDS, "ks": list(KS),
               "seconds": round(time.time() - t0), "by_label": {}}
    for grp, d in pooled.items():
        sel = [r for r in rows if grp == "ALL" or r["label"] == grp]
        elig = [r for r in sel if r["gold_urls"]]
        entry = {
            "claims": len(sel),
            "claims_without_gold_in_store": len(sel) - len(elig),
            "gold_urls": d[KS[0]][1],
            "avg_passages": round(sum(r["n_passages"] for r in sel) / len(sel)),
        }
        for k in KS:
            entry[f"recall@{k}"] = round(d[k][0] / d[k][1], 3) if d[k][1] else None
        for k in KS:
            hit = sum(1 for r in elig if r["recall"][str(k)][0] > 0)
            entry[f"any_gold@{k}"] = round(hit / len(elig), 3) if elig else None
        summary["by_label"][grp] = entry
    save_json(summary, "retrieval_recall.json", "week2")
    print(json.dumps(summary["by_label"], indent=1))
    print("outputs in", OUT / "week2")


if __name__ == "__main__":
    main()
