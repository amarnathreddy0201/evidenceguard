"""Run the one-pass baseline on the week-2 dev claims (or the golden set) and save one JSON line per claim.

Examples (from the repo root):
  python scripts/run_baseline.py --llm mock --limit 3          # dry run, no model needed
  python scripts/run_baseline.py --llm ollama --model qwen2.5:3b-instruct --name baseline_q3b
Results are appended, so a rerun resumes where it stopped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.baseline import run_claim
from evidenceguard.config import OUT, STORE_DEV, ensure_dirs, set_seed
from evidenceguard.data import load_split
from evidenceguard.llm import get_client
from evidenceguard.rerank import try_embedder


def claim_ids(which, split):
    gdir = OUT / "golden"
    if which == "week2":
        ids = [c["claim_id"] for c in json.loads((OUT / "week2" / "dev100_ids.json").read_text("utf-8"))["claims"]]
    else:
        ids = [json.loads(l)["dev_index"] for l in open(gdir / "golden_dev150.jsonl", encoding="utf-8")]
    if split != "all":
        sp = json.loads((gdir / "dev_split.json").read_text("utf-8"))["split"]
        ids = [i for i in ids if sp[str(i)] == split]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", choices=["week2", "golden"], default="week2")
    ap.add_argument("--split", choices=["all", "tune", "test"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new claims (0 = all)")
    ap.add_argument("--llm", choices=["ollama", "mock"], default="ollama")
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--no-rerank", action="store_true", help="skip the embedding rerank")
    ap.add_argument("--questions", type=int, default=4)
    ap.add_argument("--name", default=None, help="output file name (without extension)")
    a = ap.parse_args()

    ensure_dirs()
    set_seed()
    dev = load_split("dev")
    ids = claim_ids(a.claims, a.split)
    name = a.name or f"baseline_{a.llm}_{a.claims}_{a.split}"
    out_path = OUT / "baseline" / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["claim_id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    todo = [i for i in ids if i not in done]
    if a.limit:
        todo = todo[:a.limit]

    llm = get_client(a.llm, a.model)
    embedder = None if a.no_rerank else try_embedder()
    print(f"claims: {len(ids)} total, {len(done)} done, {len(todo)} to run | llm={llm.name} | "
          f"reranker={'embedding' if embedder else 'none (sentence-transformers not installed or disabled)'}")
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for n, cid in enumerate(todo, 1):
            res = run_claim(llm, dev[cid], STORE_DEV / f"{cid}.json", embedder, n_questions=a.questions)
            res.update({"claim_id": cid, "claim": dev[cid]["claim"], "llm": llm.name, "run": name})
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] claim {cid}: gold={res['gold_label']} pred={res['pred_label']} "
                  f"({res['seconds']['total']}s, elapsed {time.time() - t0:.0f}s)", flush=True)
    print("saved", out_path)


if __name__ == "__main__":
    main()
