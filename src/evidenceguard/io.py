import json
from datetime import datetime
from .config import OUT

def save_jsonl(rows, name, subdir=""):
    path = OUT / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

def save_json(obj, name, subdir=""):
    path = OUT / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def run_tag(prefix):
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M}"
