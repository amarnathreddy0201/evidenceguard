"""Small LLM clients with an on-disk cache.

OllamaClient talks to a local Ollama server over HTTP (no extra packages).
MockClient is a deterministic stand-in for tests and dry runs.
Every call is cached by (model, system, prompt, options), so a rerun after a crash is free.
"""
import hashlib
import json
import re
import time
import urllib.request

from .config import OUT, SEED


def parse_json(text):
    """Return the first JSON object found in a model reply, or None."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(t[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class _CachedClient:
    name = "base"

    def __init__(self, cache=True):
        self.cache_dir = OUT / "cache" / "llm" if cache else None
        self.calls = 0
        self.cache_hits = 0

    def _key(self, system, prompt, options):
        raw = json.dumps([self.name, system, prompt, options], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, prompt, system="", max_tokens=300, json_mode=True):
        options = {"max_tokens": max_tokens, "json": json_mode, "seed": SEED}
        key = self._key(system, prompt, options)
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path is not None and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        t0 = time.time()
        text = self._call(prompt, system, max_tokens, json_mode)
        self.calls += 1
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text, "seconds": round(time.time() - t0, 1)},
                                       ensure_ascii=False), encoding="utf-8")
        return text

    def _call(self, prompt, system, max_tokens, json_mode):
        raise NotImplementedError


class OllamaClient(_CachedClient):
    def __init__(self, model="qwen2.5:3b-instruct", host="http://localhost:11434",
                 num_ctx=4096, timeout=900, cache=True):
        super().__init__(cache)
        self.model, self.host, self.num_ctx, self.timeout = model, host.rstrip("/"), num_ctx, timeout
        self.name = f"ollama:{model}:ctx{num_ctx}"

    def _call(self, prompt, system, max_tokens, json_mode):
        body = {
            "model": self.model, "prompt": prompt, "system": system, "stream": False,
            "options": {"temperature": 0, "seed": SEED, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(self.host + "/api/generate", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("response", "")


class MockClient(_CachedClient):
    """Deterministic fake model: writes questions from the claim and picks a label from a hash."""
    name = "mock"

    def __init__(self):
        super().__init__(cache=False)

    def _call(self, prompt, system, max_tokens, json_mode):
        claim = re.search(r"Claim: (.*)", prompt)
        claim = claim.group(1).strip() if claim else prompt[:80]
        if "questions" in prompt.lower().split("return json")[-1][:80]:
            words = [w for w in re.findall(r"[A-Za-z]{5,}", claim)][:6]
            qs = [f"What is known about {w.lower()} in this claim?" for w in words[:3]] or ["Is the claim true?"]
            return json.dumps({"questions": qs})
        labels = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
        h = int(hashlib.sha256(claim.encode("utf-8")).hexdigest(), 16)
        return json.dumps({"label": labels[h % 4], "reason": "mock", "evidence_ids": [1]})


def get_client(kind="ollama", model="qwen2.5:3b-instruct", **kw):
    if kind == "mock":
        return MockClient()
    if kind == "ollama":
        return OllamaClient(model=model, **kw)
    raise ValueError(f"unknown LLM backend: {kind}")
