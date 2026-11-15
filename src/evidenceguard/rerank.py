"""Optional embedding rerank. Needs sentence-transformers; the baseline runs without it."""


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily on purpose
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts, batch_size=64):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)

    def scores(self, queries, passages):
        """Max cosine similarity of each passage to any query. Returns a list of floats."""
        q = self.encode(queries)
        p = self.encode(passages)
        sims = p @ q.T                     # (n_passages, n_queries), both L2-normalised
        return sims.max(axis=1).tolist()


def try_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Return an Embedder, or None if sentence-transformers is not installed."""
    try:
        return Embedder(model_name)
    except ImportError:
        return None
