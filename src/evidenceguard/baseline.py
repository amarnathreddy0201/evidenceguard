"""One-pass baseline: questions -> search -> top passages -> one verdict. It always answers."""
import re
import time

from .data import LABELS
from .llm import parse_json
from .retrieve import BM25, build_passages

SYSTEM = ("You are a careful fact-checker. Use only the evidence passages you are given. "
          "Reply with a single JSON object and nothing else.")

QUESTION_PROMPT = """Write {n} short, specific questions that must be answered to check this claim.
Use names, dates and numbers from the claim. Each question should be a good web search query.

Claim: {claim}
Claim date: {date}
Speaker: {speaker}

Return JSON like {{"questions": ["...", "..."]}}"""

VERDICT_PROMPT = """Decide the verdict for the claim using only the numbered evidence passages.

Labels:
- Supported: the evidence supports the claim.
- Refuted: the evidence contradicts the claim.
- Not Enough Evidence: the evidence is not enough to support or refute the claim.
- Conflicting Evidence/Cherrypicking: the evidence conflicts, or the claim is technically true but misleading.

Claim: {claim}
Claim date: {date}
Speaker: {speaker}

Evidence passages:
{evidence}

Return JSON like {{"label": "<one of the four labels>", "reason": "<one or two sentences>", "evidence_ids": [1, 2]}}"""

_CANON = {
    "supported": "Supported", "support": "Supported", "true": "Supported",
    "refuted": "Refuted", "refute": "Refuted", "false": "Refuted",
    "not enough evidence": "Not Enough Evidence", "not enough information": "Not Enough Evidence",
    "nei": "Not Enough Evidence", "not enough": "Not Enough Evidence",
    "conflicting evidence/cherrypicking": "Conflicting Evidence/Cherrypicking",
    "conflicting evidence": "Conflicting Evidence/Cherrypicking", "conflicting": "Conflicting Evidence/Cherrypicking",
    "cherrypicking": "Conflicting Evidence/Cherrypicking", "cherry-picking": "Conflicting Evidence/Cherrypicking",
    "conflicting/cherrypicking": "Conflicting Evidence/Cherrypicking",
}


def normalize_label(text):
    """Map a model's label text to one of the four dataset labels, or None."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"\s+", " ", text.strip().lower())
    if t in _CANON:
        return _CANON[t]
    for key in sorted(_CANON, key=len, reverse=True):
        if key in t:
            return _CANON[key]
    return None


def generate_questions(llm, row, n=4):
    prompt = QUESTION_PROMPT.format(n=n, claim=row["claim"], date=row.get("claim_date") or "unknown",
                                    speaker=row.get("speaker") or "unknown")
    raw = llm.generate(prompt, system=SYSTEM, max_tokens=200)
    obj = parse_json(raw) or {}
    qs = obj.get("questions") if isinstance(obj.get("questions"), list) else []
    qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()][:n]
    return qs, raw


def retrieve_for_claim(bm25, passages, queries, embedder=None, per_query=30, pool=100, k_final=8,
                       per_url_cap=2, rrf_k=60):
    """Reciprocal-rank fusion over several BM25 queries, optional embedding rerank, URL-diverse top-k."""
    fused = {}
    for q in queries:
        for rank, (pid, _) in enumerate(bm25.search(q, k=per_query), 1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(fused, key=lambda i: -fused[i])[:pool]
    if embedder is not None and ranked:
        sims = embedder.scores(queries, [passages[i]["text"] for i in ranked])
        order = sorted(range(len(ranked)), key=lambda j: -sims[j])
        scored = [(ranked[j], float(sims[j])) for j in order]
    else:
        scored = [(i, fused[i]) for i in ranked]
    out, per_url = [], {}
    for pid, s in scored:
        u = passages[pid]["url"]
        if per_url.get(u, 0) >= per_url_cap:
            continue
        per_url[u] = per_url.get(u, 0) + 1
        out.append({"rank": len(out) + 1, "score": round(s, 4), "url": u, "text": passages[pid]["text"]})
        if len(out) == k_final:
            break
    return out


def _trim(text, words=170):
    w = text.split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def format_evidence(passages):
    return "\n".join(f"[{p['rank']}] ({p['url']}) {_trim(p['text'])}" for p in passages)


def decide_verdict(llm, row, passages):
    prompt = VERDICT_PROMPT.format(claim=row["claim"], date=row.get("claim_date") or "unknown",
                                   speaker=row.get("speaker") or "unknown", evidence=format_evidence(passages))
    raw = llm.generate(prompt, system=SYSTEM, max_tokens=250)
    obj = parse_json(raw) or {}
    return {"label": normalize_label(obj.get("label")), "label_raw": obj.get("label"),
            "reason": obj.get("reason"), "evidence_ids": obj.get("evidence_ids"), "raw": raw}


def run_claim(llm, row, store_path, embedder=None, n_questions=4, k_final=8, max_words=150):
    """Run the whole one-pass baseline for one dev claim. Returns a result dict."""
    t0 = time.time()
    passages, store_urls = build_passages(store_path, max_words)
    bm25 = BM25([p["text"] for p in passages])
    questions, q_raw = generate_questions(llm, row, n_questions)
    queries = [row["claim"]] + questions
    top = retrieve_for_claim(bm25, passages, queries, embedder, k_final=k_final)
    t_ret = time.time()
    verdict = decide_verdict(llm, row, top)
    return {
        "gold_label": row["label"], "pred_label": verdict["label"], "label_raw": verdict["label_raw"],
        "reason": verdict["reason"], "evidence_ids": verdict["evidence_ids"],
        "questions": questions, "queries_used": queries,
        "passages": top, "n_passages_indexed": len(passages), "n_documents": len(store_urls),
        "reranker": "embedding" if embedder is not None else "none",
        "raw_questions": q_raw, "raw_verdict": verdict["raw"],
        "seconds": {"index_and_retrieval": round(t_ret - t0, 1), "total": round(time.time() - t0, 1)},
    }
