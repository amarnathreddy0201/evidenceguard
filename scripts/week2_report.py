"""Build the Week 2 HTML report from the saved retrieval outputs.

Run after week2_retrieval.py:  python scripts/week2_report.py
Writes <ROOT>/week2_report.html
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidenceguard.config import OUT, ROOT

W2 = OUT / "week2"
summary = json.loads((W2 / "retrieval_recall.json").read_text(encoding="utf-8"))
ids = json.loads((W2 / "dev100_ids.json").read_text(encoding="utf-8"))
rows = [json.loads(l) for l in open(W2 / "retrieval_dev100.jsonl", encoding="utf-8")]

ORDER = ["ALL", "Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
SHORT = {"ALL": "All 100", "Supported": "Supported", "Refuted": "Refuted",
         "Not Enough Evidence": "Not Enough Evidence", "Conflicting Evidence/Cherrypicking": "Conflicting / Cherrypicking"}
ks = summary["ks"]
by = summary["by_label"]
e = html.escape


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def table_recall():
    head = "".join(f"<th>Recall@{k}</th>" for k in ks)
    body = ""
    for g in ORDER:
        d = by[g]
        cells = "".join(f"<td>{pct(d[f'recall@{k}'])}</td>" for k in ks)
        body += f"<tr><td>{SHORT[g]}</td><td>{d['claims']}</td><td>{d['gold_urls']}</td>{cells}</tr>"
    return f"<table><thead><tr><th>Group</th><th>Claims</th><th>Gold URLs</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def table_anygold():
    head = "".join(f"<th>Any gold @{k}</th>" for k in ks)
    body = ""
    for g in ORDER:
        d = by[g]
        cells = "".join(f"<td>{pct(d[f'any_gold@{k}'])}</td>" for k in ks)
        body += (f"<tr><td>{SHORT[g]}</td><td>{d['claims'] - d['claims_without_gold_in_store']}</td>"
                 f"<td>{d['claims_without_gold_in_store']}</td>{cells}</tr>")
    return (f"<table><thead><tr><th>Group</th><th>Claims with gold in store</th>"
            f"<th>Claims with none</th>{head}</tr></thead><tbody>{body}</tbody></table>")


def misses(label, n=3, k="50"):
    out = []
    for r in rows:
        if r["label"] != label or not r["gold_urls"]:
            continue
        found, total = r["recall"][k]
        if found < total:
            out.append((total - found, r))
    out.sort(key=lambda x: -x[0])
    items = ""
    for _, r in out[:n]:
        found, total = r["recall"][k]
        items += (f"<li><strong>Claim {r['claim_id']}</strong>: {e(r['claim'])}<br>"
                  f"<span class=\"small\">gold pages found in top {k}: {found} of {total}; "
                  f"{r['n_passages']:,} passages searched</span></li>")
    return f"<ul>{items}</ul>" if items else "<p class=\"small\">None.</p>"


counts = {g: by[g]["claims"] for g in ORDER if g != "ALL"}
avg_pass = by["ALL"]["avg_passages"]

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvidenceGuard Week 2 Report</title>
<style>
:root {{ --ink:#172033; --muted:#536176; --paper:#fff; --canvas:#f3f6fb; --line:#dce4ef; --blue:#2457a7;
  --blue-soft:#eaf1fc; --teal-soft:#e8f6f3; --teal:#087e75; --amber-soft:#fff5df; --amber:#d39a2c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--canvas); color:var(--ink); font:16px/1.6 "Segoe UI",Arial,sans-serif; }}
header {{ padding:48px 24px 36px; background:linear-gradient(130deg,#14284a,#2457a7 65%,#347aa2); color:#fff; }}
.wrap {{ width:min(100% - 40px, 980px); margin:0 auto; }}
h1 {{ margin:0 0 8px; font-size:clamp(1.8rem,4vw,2.6rem); }}
header p {{ margin:0; color:#e4edfb; }}
main {{ padding:28px 0 48px; }}
section {{ margin:0 0 20px; padding:24px 28px; background:var(--paper); border:1px solid var(--line); border-radius:14px; }}
h2 {{ margin:0 0 12px; font-size:1.3rem; }}
h3 {{ margin:18px 0 6px; font-size:1.02rem; }}
table {{ width:100%; border-collapse:collapse; font-size:.93rem; margin:10px 0; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:#f5f7fa; }}
.tw {{ overflow-x:auto; }}
.small {{ color:var(--muted); font-size:.9rem; }}
.callout {{ margin:14px 0 0; padding:14px 16px; border-left:4px solid var(--teal); background:var(--teal-soft); border-radius:8px; }}
.caution {{ border-left-color:var(--amber); background:var(--amber-soft); }}
code {{ background:#edf1f7; padding:1px 5px; border-radius:4px; font:.88em Consolas,monospace; }}
pre {{ background:#f5f7fa; border:1px solid var(--line); border-radius:8px; padding:12px; overflow-x:auto; font:.84rem/1.5 Consolas,monospace; }}
li {{ margin:6px 0; }}
</style>
</head>
<body>
<header><div class="wrap">
<h1>Week 2: retrieval baseline</h1>
<p>EvidenceGuard, evidence-aware abstention for automated fact-checking. BM25 retrieval over 100 AVeriTeC dev claims, no LLM yet.</p>
</div></header>
<main class="wrap">

<section>
<h2>1. What week 2 set out to do</h2>
<ul>
<li>Choose the 100 dev claims for the baseline (all Not Enough Evidence, all Conflicting/Cherrypicking, plus Supported and Refuted).</li>
<li>Read each claim's knowledge-store file, split the pages into passages, and index them with BM25.</li>
<li>Measure how often the pages the annotators used (gold URLs) appear in the top 8, 50 and 200 passages, split by label.</li>
</ul>
<p>Running on the laptop CPU only: Python, no GPU, no LLM.</p>
</section>

<section>
<h2>2. What was done</h2>
<h3>Claim selection (seed {summary['seed']})</h3>
<p>{ids['n']} claims from <code>dev.json</code>: {counts['Not Enough Evidence']} Not Enough Evidence,
{counts['Conflicting Evidence/Cherrypicking']} Conflicting/Cherrypicking, {counts['Refuted']} Refuted and
{counts['Supported']} Supported. The two rare classes are fully included, so the sample is not representative of the
real label mix (Refuted is about 61% of dev). IDs are in <code>outputs/week2/dev100_ids.json</code>.</p>
<h3>Data format found</h3>
<p>Each <code>output_dev/&lt;n&gt;.json</code> is JSON Lines, one document per line with <code>claim_id</code>,
<code>type</code>, <code>query</code>, <code>url</code> and <code>url2text</code> (a list of sentences). File number <em>n</em> matches
<code>dev.json[n]</code>, checked through the gold URLs on 12 files.</p>
<h3>Retrieval pipeline</h3>
<ol>
<li>Read the store file and keep one copy of each URL (the same page appears under several search types).</li>
<li>Group consecutive sentences into passages of about 150 words; an average claim has about {avg_pass:,} passages.</li>
<li>Index with Okapi BM25 (k1 = 1.5, b = 0.75, simple stop-word list). The query is the claim text only.</li>
<li>Keep the top 200 for scoring and the top 50 in the saved file.</li>
</ol>
<div class="callout caution"><strong>Leak guard:</strong> the store's <code>type</code> field marks the gold documents. The retrieval code
never reads it. Gold pages come from the annotators' answer URLs (<code>source_url</code> and <code>cached_source_url</code> in <code>dev.json</code>).</div>
</section>

<section>
<h2>3. Results</h2>
<h3>Gold-URL recall</h3>
<p class="small">Share of gold URLs that appear among the URLs of the top-k passages. Only gold pages that exist in the store are counted.</p>
<div class="tw">{table_recall()}</div>
<h3>At least one gold page found</h3>
<p class="small">Share of claims where any gold page appears in the top k (claims with no gold page in the store are excluded).</p>
<div class="tw">{table_anygold()}</div>
<p class="small">Run time: {summary['seconds']//60} minutes for {ids['n']} claims on the laptop CPU (indexing about {avg_pass:,} passages per claim in pure Python).</p>
</section>

<section>
<h2>4. What to read in these numbers</h2>
<ul>
<li><strong>Recall is low everywhere.</strong> Only {pct(by['ALL']['recall@8'])} of gold pages are in the top 8 passages, {pct(by['ALL']['recall@50'])} in the top 50 and {pct(by['ALL']['recall@200'])} in the top 200. At least one gold page reaches the top 8 for only {pct(by['ALL']['any_gold@8'])} of claims. Claim text alone is a weak BM25 query over about {avg_pass:,} passages per claim.</li>
<li><strong>Reranking the BM25 list cannot fix this.</strong> Recall@200 is the ceiling for any reranker working on this list, and it is only {pct(by['ALL']['recall@200'])}. Week 3 has to improve the first-stage search (questions as queries, more candidates, dense retrieval), not just rerank.</li>
<li><strong>The four labels look similar.</strong> The Not Enough Evidence group ({pct(by['Not Enough Evidence']['recall@50'])} at 50) and the Conflicting group ({pct(by['Conflicting Evidence/Cherrypicking']['recall@50'])} at 50) are not clearly different from Refuted ({pct(by['Refuted']['recall@50'])}) or Supported ({pct(by['Supported']['recall@50'])}). With 8 to 38 claims per group, differences of a few points are noise.</li>
<li><strong>Gold-URL recall is a strict measure.</strong> It only credits the exact pages the annotators cited. In claim 9 (a GDP claim) BM25 ranks the transcript of the speech and a CNN fact-check on top, which are on topic, but the annotators' page was a different article, so recall counts it as a miss. Read some top passages by hand before concluding the retriever is bad.</li>
<li>For Not Enough Evidence claims, annotators may have found little usable evidence, so low recall there can be partly expected. Do not tune the retriever on those claims alone.</li>
</ul>
<h3>Largest misses in the top 50 for Not Enough Evidence claims</h3>
{misses('Not Enough Evidence')}
<h3>Largest misses in the top 50 for Conflicting/Cherrypicking claims</h3>
{misses('Conflicting Evidence/Cherrypicking')}
</section>

<section>
<h2>5. Limits of this baseline</h2>
<ul>
<li>Claim text is the only query. The papers generate questions with an LLM and search with them; that comes in week 3.</li>
<li>BM25 only, no dense reranking yet. The top-8 list comes straight from BM25.</li>
<li>Gold is defined by URL, not by passage. A gold page counts as found when any of its passages is in the top k, even if the answering sentence is not.</li>
<li>Some annotator URLs may be missing from the store; those are excluded and counted above.</li>
<li>One run, one seed, 100 claims. Per-label groups are small (27 to 38 claims), so differences of a few points are noise.</li>
</ul>
</section>

<section>
<h2>6. Files produced</h2>
<table>
<tr><th>File</th><th>Content</th></tr>
<tr><td><code>outputs/week2/dev100_ids.json</code></td><td>The 100 selected claim IDs with labels</td></tr>
<tr><td><code>outputs/week2/retrieval_dev100.jsonl</code></td><td>Per claim: gold URLs, recall, top 50 passages with score and URL</td></tr>
<tr><td><code>outputs/week2/retrieval_recall.json</code></td><td>The recall tables in this report</td></tr>
<tr><td><code>evidenceguard/scripts/week2_retrieval.py</code></td><td>Runs the whole experiment</td></tr>
<tr><td><code>evidenceguard/scripts/week2_report.py</code></td><td>Regenerates this page</td></tr>
</table>
<p class="small">To reproduce: <code>python scripts/week2_retrieval.py</code>, then <code>python scripts/week2_report.py</code>, from <code>evidenceguard/</code>.</p>
</section>

<section>
<h2>7. Next: week 3</h2>
<ol>
<li>Generate a few questions per claim with a small local LLM (Qwen2.5-3B in Ollama) and search with them as well as the claim.</li>
<li>Rerank the BM25 top 50 with a small embedding model on CPU and keep the top 8.</li>
<li>Give the top passages to the LLM and have it output one of the four labels as JSON.</li>
<li>Evaluate accuracy and per-class F1, and save one line per claim for the week-4 error analysis.</li>
</ol>
<p class="small">Before week 3: read the passages for a handful of Not Enough Evidence and Conflicting claims and note what kind of evidence is missing.</p>
</section>

</main>
</body>
</html>
"""
dest = ROOT / "week2_report.html"
dest.write_text(page, encoding="utf-8")
print("wrote", dest)
