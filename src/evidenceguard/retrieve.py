"""Passage building and BM25 retrieval over an AVeriTeC knowledge-store file.

Store lines carry a "type" field that marks gold documents. That field leaks the answer.
Nothing in this module reads it: retrieval sees only "url" and "url2text".
"""
import json
import math
import re
from collections import defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an and are as at be been by for from has have he her his i in is it its of on or
that the their this to was were will with would not no but if so than then there they we you who what which
when where while about after before into over under up out also more most such can could should may might""".split())


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def read_store_documents(path):
    """Yield (url, sentences) from a knowledge-store file, once per unique URL. Ignores "type"."""
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            url = o.get("url")
            if url in seen:
                continue
            seen.add(url)
            yield url, o.get("url2text") or []


def chunk_sentences(sentences, max_words=150):
    """Group consecutive sentences into passages of roughly max_words words."""
    out, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def build_passages(path, max_words=150):
    """Return (passages, urls_in_store). Each passage is {"url", "text"}."""
    passages, urls = [], set()
    for url, sents in read_store_documents(path):
        urls.add(url)
        for text in chunk_sentences(sents, max_words):
            passages.append({"url": url, "text": text})
    return passages, urls


class BM25:
    """Okapi BM25 over a list of passage texts (pure Python, inverted index)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)          # term -> [(doc_id, term_frequency)]
        self.doc_len = []
        for i, t in enumerate(texts):
            toks = tokenize(t)
            self.doc_len.append(len(toks))
            tf = defaultdict(int)
            for tok in toks:
                tf[tok] += 1
            for tok, c in tf.items():
                self.postings[tok].append((i, c))
        self.n = len(texts)
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 1.0

    def search(self, query, k=50):
        """Return [(doc_id, score)] for the top-k passages."""
        scores = defaultdict(float)
        for tok in set(tokenize(query)):
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
