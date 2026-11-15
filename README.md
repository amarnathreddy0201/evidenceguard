# EvidenceGuard

**Evidence-Aware Abstention for Reliable Automated Fact-Checking** (M.Tech project)

A fact-checker that searches again when its evidence is missing and refuses to answer when the evidence
stays insufficient or conflicting. The planned system is a small agentic loop (Planner, Retriever, Evidence
Auditor, Judge) on one local language model, compared against a one-pass baseline that always answers.

## Status
| Part | State |
|---|---|
| Data profile, dev knowledge store, BM25 retrieval recall (week 2) | Done (recall@8 0.039, @50 0.131, @200 0.233 on 100 claims) |
| Golden evaluation set (150 claims) and tune/test split | Built; sufficiency and conflict labels are **derived guesses, not yet hand-labelled** (0 of 150 labelled) |
| One-pass baseline (questions, search, verdict) | **Done:** run on 100 claims with Qwen2.5-3B; accuracy 0.24, macro-F1 0.244 (see "Baseline results") |
| Scoring, label merging, SufficientFacts loader | Done |
| Review sheet and per-claim text files for labelling | Done (`review_sheet.csv`, `review_txt/`) |
| Hand labelling, then `merge_labels.py` | Not done |
| Evidence Auditor and agent loop | Not started |

## Layout
```
evidenceguard/
  src/evidenceguard/
    config.py          paths and seed (EG_ROOT, EG_STORE_DEV override them)
    data.py            AVeriTeC loading, dev subset, annotator URLs
    retrieve.py        passage building and BM25 (pure Python)
    rerank.py          optional embedding rerank (needs sentence-transformers)
    llm.py             Ollama client, mock client, on-disk reply cache
    baseline.py        one-pass baseline
    metrics.py         accuracy, F1, bootstrap intervals, Cohen's kappa
    golden.py          golden set and tune/test split
    sufficientfacts.py SufficientFacts loader
  scripts/             runnable steps (see "What to run")
  tests/test_core.py   plain-assert tests
  notebooks/           Colab setup notebooks
```
Data and results live next to the repo, in `../data/` and `../outputs/`. They are git-ignored.

## Setup (local, CPU)
1. `pip install -e .` (the core code uses only the Python standard library; Python 3.10 or newer).
2. `pip install -r requirements.txt` for data download (`huggingface_hub`) and the optional embedding
   rerank (`sentence-transformers`, which pulls in PyTorch). Skip it to run the baseline with BM25 only.
3. Install Ollama from https://ollama.com, then `ollama pull qwen2.5:3b-instruct`.
4. Data, under `../data/`:
   - `averitec/train.json` and `averitec/dev.json`
   - the unzipped dev knowledge store at
     `averitec_store/data_store/knowledge_store/dev_knowledge_store/output_dev/` (one JSON Lines file per
     claim; file *n* belongs to `dev.json[n]`)
   - `sufficient_facts/{fever,vitaminc,hover}_sufficient_facts.jsonl` for the SufficientFacts second check

Check the install with `python tests/test_core.py`.

## What to run
Run these from this folder. Steps 1 to 5 are already done and their outputs are in `../outputs/`. Steps 6 and 7
are for labelling.

| # | Command | What it does | Output (in `../outputs/`) |
|---|---|---|---|
| 1 | `python scripts/week2_retrieval.py` | BM25 over 100 dev claims; gold-URL recall at 8, 50, 200 | `week2/` |
| 2 | `python scripts/build_golden.py` | Golden set (150 claims), tune/test split, blind review sheet | `golden/` |
| 3 | `python scripts/run_baseline.py --llm ollama --limit 3 --name try1` | Test the baseline on 3 claims | `baseline/try1.jsonl` |
| 4 | `python scripts/run_baseline.py --llm ollama --name baseline_q3b` | Baseline on all 100 claims (run overnight; resumes if stopped) | `baseline/baseline_q3b.jsonl` |
| 5 | `python scripts/eval_baseline.py --name baseline_q3b` | Accuracy, per-class F1, confusion matrix, bootstrap intervals, for all claims and for tune and test | `baseline/baseline_q3b_scores.json` |
| 6 | `python scripts/export_review_txt.py` | Split the review sheet into one text file per claim (claim, evidence, instruction, three blank label lines). Skips existing files; `--force` rewrites them | `golden/review_txt/<example_id>.txt` |
| 7 | `python scripts/merge_labels.py` | Merge your hand labels from the CSV into the golden set; add `--second other.csv` for kappa | `golden/golden_dev150_verified.jsonl`, `golden/golden_label_report.json` |

`merge_labels.py` reads only the CSV, not the text files. Running it with no labels filled writes a "verified" file
that is identical to the derived one (`labelled: 0`); do not cite it as verified.

Options for `run_baseline.py`: `--llm mock` runs without a model for a dry run, `--model` picks the Ollama model,
`--split tune|test|all`, `--claims week2|golden`, `--no-rerank`, `--questions N`.

After step 3, open `try1.jsonl` and check that `questions`, `pred_label` and `reason` look sensible. After step 5,
the `unparsed_predictions` count should be small; a high count means the model is not returning the JSON format
(it was 0 of 100 in the real run).

Each line of a baseline `.jsonl` is one claim: the claim, gold label, predicted label and reason, the generated
questions, the 8 retrieved passages with URLs, the raw model replies and timings.

**Warning:** do not rerun `build_golden.py` after you start labelling. It rewrites `golden/review_sheet.csv` and
would overwrite your labels. Keep a copy of the sheet as you work.

## Behaviour to know about
- `run_baseline.py` appends one line per claim, so a stopped run resumes.
- Model replies are cached in `../outputs/cache/llm/`. A rerun with the same prompts and model costs nothing.
  Delete that folder if you want to force fresh replies.
- Generation is greedy (temperature 0, seed 42). `set_seed()` fixes the seed for Python and, when installed,
  numpy and torch.
- Indexing a claim's knowledge store in pure Python takes about 20 to 36 seconds; the model call comes on top.
  Measured: about 54 seconds per claim in total, so 100 claims took roughly 1.5 hours.
- The review sheet's evidence lines start with a bullet (`•`), not `-`: Excel treats a cell that begins with `-` or
  `=` as a formula and refuses edits.

## Baseline results
One-pass baseline, Qwen2.5-3B-Instruct, 100 week-2 claims (35 Not Enough Evidence, 38 Conflicting, 19 Refuted,
8 Supported). Brackets are 95% bootstrap intervals.

| Set | Claims | Accuracy | Macro-F1 |
|---|---|---|---|
| All | 100 | 0.24 [0.16, 0.32] | 0.244 [0.158, 0.322] |
| Tune | 48 | 0.146 [0.062, 0.25] | 0.151 |
| Test | 52 | 0.327 [0.192, 0.442] | 0.328 |

| Class (all 100) | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Supported | 8 | 0.21 | 0.88 | 0.33 |
| Refuted | 19 | 0.22 | 0.32 | 0.26 |
| Not Enough Evidence | 35 | 0.19 | 0.09 | 0.12 |
| Conflicting / Cherrypicking | 38 | 0.35 | 0.21 | 0.26 |

What it shows: the baseline almost never refuses when it should (3 of 35 Not Enough Evidence claims caught),
over-predicts Supported (34 predictions for 8 true claims), and saw the annotator-cited pages in its top 8 passages
only 4.4% of the time (a strict measure: relevant pages the annotators did not cite count as misses).
The tune/test gap is noise (about 50 claims each, overlapping intervals), so use the all-claims row as the baseline.

## Evaluation data
- **AVeriTeC dev (500 claims):** gold verdicts and evidence URLs. `golden/dev_split.json` assigns every claim to
  `tune` or `test` (stratified by label, seed 42).
- **Golden set (150 claims):** all Not Enough Evidence and Conflicting claims, plus Supported and Refuted.
  The verdict is gold. The sufficiency, conflict and expected-action fields are derived from the verdict
  (`human_verified: false`) and become gold only after you fill `golden/review_sheet.csv` and run step 7.
  The sheet hides the verdict and justification so you judge the evidence only. Label a sample first (about 25
  tune and 25 test claims) to see how often the derived labels agree with yours.
- **SufficientFacts (3,000 examples, MIT licence):** second check of insufficiency detection. Use
  `sufficientfacts.load_sufficientfacts()` and `balanced_sample()`; about 74% of the examples are insufficient.

## Rules for clean evaluation
- The knowledge store has a `type` field that marks the gold pages. Retrieval never reads it. The annotators'
  URLs in `dev.json` are used only to score results.
- Tune prompts, thresholds and the round budget on the `tune` half only. Score the `test` half once.
- Keep synthetic examples out of the final test and report them separately.
- Small groups (about 18 to 19 claims per rare class in the test half) need confidence intervals.
- Say plainly which labels are derived and which were checked by hand.

## Licences and data
AVeriTeC is CC BY-NC 4.0 (non-commercial): cite it, do not redistribute the data. SufficientFacts is MIT.
This repository does not contain any dataset files.

## Colab
`notebooks/01_setup_and_profile.ipynb` mounts Drive, pulls this repo and profiles the data. With `EG_ROOT`
unset, the code uses the Drive folder `EvidenceGuard/` on Colab and the folder above the repo locally.

## Project documents
The abstract, architecture diagram, evaluation-data page, week 2 report and project summary are HTML files in
the folder above this repository (`project_abstract.html`, `architecture_diagram.html`, `evaluation_data.html`,
`week2_report.html`, `project_summary.html`, `analysis.html`), plus `viva_guide.html` (what to explain in the viva,
with diagrams).
