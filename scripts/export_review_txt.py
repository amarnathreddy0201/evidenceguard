"""Split the review sheet into one text file per claim, named by example_id (e.g. avdev-9.txt).

  python scripts/export_review_txt.py
  python scripts/export_review_txt.py --sheet path/to/review_sheet.csv --out path/to/folder

Each file has the claim, the evidence lines, then three blank label lines to fill in.
Files that already exist are skipped, so labels you typed are never overwritten
(use --force to rewrite them).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.config import OUT

INSTRUCTION = (
    "Analyse the above information, check the links, extract realworld content. "
    "Based on information give me answers for the below questions."
)

LABEL_LINES = [
    "evidence_sufficiency (SUFFICIENT/INSUFFICIENT/CONFLICTING): ",
    "evidence_conflict (NONE/MINOR/MATERIAL): ",
    "expected_action (VERDICT/ABSTAIN): ",
]


def column(row, prefix):
    """Value of the column whose header starts with prefix."""
    key = next(k for k in row if k.startswith(prefix))
    return (row[key] or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(OUT / "golden" / "review_sheet.csv"))
    ap.add_argument("--out", default=str(OUT / "golden" / "review_txt"))
    ap.add_argument("--force", action="store_true", help="overwrite existing text files")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(a.sheet, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = out / f"{row['example_id']}.txt"
            if path.exists() and not a.force:
                skipped += 1
                continue
            text = "\n".join([
                "claim:",
                column(row, "claim"),
                "",
                "evidence (question | answer | type | url):",
                column(row, "evidence"),
                "",
                INSTRUCTION,
                *LABEL_LINES,
            ])
            path.write_text(text + "\n", encoding="utf-8")
            written += 1
    print(f"wrote {written} files, skipped {skipped} existing, in {out}")


if __name__ == "__main__":
    main()
