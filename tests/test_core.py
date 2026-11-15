"""Plain-assert tests. Run:  python tests/test_core.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidenceguard.baseline import normalize_label
from evidenceguard.llm import parse_json
from evidenceguard.metrics import accuracy, cohen_kappa, confusion, macro_f1, prf
from evidenceguard.retrieve import BM25, chunk_sentences, tokenize

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]


def test_parse_json():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"label": "Refuted"} done') == {"label": "Refuted"}
    assert parse_json("no json here") is None
    assert parse_json("") is None
    assert parse_json("[1, 2]") is None


def test_normalize_label():
    assert normalize_label("Supported") == "Supported"
    assert normalize_label("refuted.") == "Refuted"
    assert normalize_label("Not Enough Information") == "Not Enough Evidence"
    assert normalize_label("NEI") == "Not Enough Evidence"
    assert normalize_label("Conflicting Evidence/Cherrypicking") == "Conflicting Evidence/Cherrypicking"
    assert normalize_label("cherry-picking") == "Conflicting Evidence/Cherrypicking"
    assert normalize_label("banana") is None
    assert normalize_label(None) is None


def test_metrics():
    yt = ["Supported", "Refuted", "Refuted", "Not Enough Evidence"]
    yp = ["Supported", "Refuted", "Supported", None]
    assert accuracy(yt, yp) == 0.5
    p, r, f = prf(yt, yp, "Refuted")
    assert (p, r) == (1.0, 0.5)
    m = confusion(yt, yp, LABELS)
    assert m["Not Enough Evidence"][None] == 1 and m["Refuted"]["Supported"] == 1
    assert 0 < macro_f1(yt, yp, LABELS) < 1
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9


def test_retrieval():
    assert "the" not in tokenize("The cat sat")
    chunks = chunk_sentences(["one two three", "four five six", "seven eight nine"], max_words=6)
    assert chunks == ["one two three four five six", "seven eight nine"]
    texts = ["apple banana fruit", "car engine road", "banana smoothie recipe"]
    top = BM25(texts).search("banana smoothie", k=2)
    assert top[0][0] == 2 and len(top) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
