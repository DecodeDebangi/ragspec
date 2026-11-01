"""Reference RAG pipelines over BEIR, used to benchmark retrieval strategies with ragspec.

ragspec config usage:

    pipeline:
      target: pipelines:make
      options: {dataset: scifact, retriever: hybrid, k: 10}

Retrievers:
  bm25    bm25s lexical (Lucene-style BM25, English stemming + stopwords)
  dense   bge-small-en-v1.5 embeddings in pgvector (HNSW, cosine)
  hybrid  bm25 top-100 + dense top-100 fused with Reciprocal Rank Fusion
  rerank  hybrid top-N rescored by a cross-encoder (`reranker`: minilm or bge)

With `generate: true` the pipeline also answers with Claude, citing the retrieved documents.
"""

from __future__ import annotations

import json
import threading
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
DSN = "postgresql://ragspec:ragspec@localhost:5433/ragspec"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}
CANDIDATES = 100
RERANK_MAX_TOKENS = 256
RRF_K = 60

# Torch models are shared across the runner's worker threads; serialize calls into them.
_model_lock = threading.Lock()
_local = threading.local()


@cache
def corpus(dataset: str) -> dict[str, dict]:
    with (HERE / "data" / dataset / "corpus.jsonl").open() as f:
        return {d["_id"]: d for d in map(json.loads, f)}


def text_of(dataset: str, doc_id: str) -> str:
    d = corpus(dataset)[doc_id]
    return f"{d.get('title') or ''}\n{d['text']}".strip()


@cache
def bm25_index(dataset: str):
    import bm25s
    import Stemmer

    retriever = bm25s.BM25.load(HERE / "data" / dataset / "bm25", load_corpus=True)
    return retriever, Stemmer.Stemmer("english")


@cache
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@cache
def cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKERS[name], max_length=RERANK_MAX_TOKENS)


def connection():
    if getattr(_local, "conn", None) is None:
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(DSN, autocommit=True)
        register_vector(conn)
        conn.execute("SET hnsw.ef_search = 200")
        _local.conn = conn
    return _local.conn


def search_bm25(dataset: str, query: str, n: int) -> list[tuple[str, float]]:
    import bm25s

    retriever, stemmer = bm25_index(dataset)
    tokens = bm25s.tokenize([query], stopwords="en", stemmer=stemmer, show_progress=False)
    if not tokens.vocab:
        return []
    docs, scores = retriever.retrieve(tokens, k=n, show_progress=False, n_threads=1)
    return [(d["id"], float(s)) for d, s in zip(docs[0], scores[0], strict=True)]


def search_dense(dataset: str, query: str, n: int) -> list[tuple[str, float]]:
    with _model_lock:
        vec = embedder().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    rows = connection().execute(
        f"SELECT id, 1 - (embedding <=> %s) FROM docs_{dataset} "
        f"ORDER BY embedding <=> %s LIMIT %s",
        (vec, vec, n),
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def rrf(*rankings: list[tuple[str, float]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def search_hybrid(dataset: str, query: str, n: int) -> list[tuple[str, float]]:
    lexical = search_bm25(dataset, query, CANDIDATES)
    return rrf(lexical, search_dense(dataset, query, CANDIDATES))[:n]


def search_rerank(dataset: str, query: str, n: int, reranker: str = "minilm",
                  depth: int = 30) -> list[tuple[str, float]]:
    candidates = search_hybrid(dataset, query, depth)
    pairs = [(query, text_of(dataset, doc_id)[:2000]) for doc_id, _ in candidates]
    with _model_lock:
        scores = cross_encoder(reranker).predict(pairs, batch_size=depth,
                                                 show_progress_bar=False)
    ranked = sorted(zip((d for d, _ in candidates), scores, strict=True),
                    key=lambda kv: kv[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:n]]


RETRIEVERS = {"bm25": search_bm25, "dense": search_dense, "hybrid": search_hybrid,
              "rerank": search_rerank}

ANSWER_SYSTEM = """\
You answer questions using only the provided sources. The user's message is either a \
question or a claim; for a claim, say whether the sources support it, refute it, or don't \
address it, and explain why.

Rules:
- Use only facts stated in the sources. Do not add background knowledge.
- After every sentence that states a fact, cite its source(s) by id in square brackets, \
e.g. [12345] or [12345, 67890].
- If the sources do not contain the answer, say so plainly instead of guessing.
- Be concise: at most 5 sentences."""


@cache
def claude():
    import anthropic

    return anthropic.Anthropic(max_retries=6)


def generate_answer(question: str, contexts: list[dict], model: str, effort: str) -> dict:
    sources = "\n\n".join(f'<source id="{c["id"]}">\n{c["text"]}\n</source>' for c in contexts)
    response = claude().beta.messages.create(
        model=model,
        max_tokens=4000,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user",
                   "content": f"<sources>\n{sources}\n</sources>\n\n{question}"}],
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("generator refused")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return {"answer": text, "usage": {"input_tokens": response.usage.input_tokens,
                                      "output_tokens": response.usage.output_tokens}}


def make(dataset: str, retriever: str = "hybrid", k: int = 10, generate: bool = False,
         model: str = "claude-opus-5-5", effort: str = "low", reranker: str = "minilm",
         rerank_depth: int = 30):
    if retriever == "rerank":
        def search(ds, q, n):
            return search_rerank(ds, q, n, reranker, rerank_depth)
    else:
        search = RETRIEVERS[retriever]
    # Load indexes and models up front so the first question's latency isn't inflated.
    corpus(dataset)
    if retriever in ("bm25", "hybrid", "rerank"):
        bm25_index(dataset)
    if retriever in ("dense", "hybrid", "rerank"):
        embedder()
        search_dense(dataset, "warm up", 1)
    if retriever == "rerank":
        cross_encoder(reranker).predict([("warm", "up")])

    def answer(question: str) -> dict:
        hits = search(dataset, question, k)
        contexts = [{"id": d, "text": text_of(dataset, d), "score": s} for d, s in hits]
        out: dict = {"contexts": contexts}
        if generate:
            gen = generate_answer(question, contexts, model, effort)
            out["answer"] = gen["answer"]
            out["meta"] = {"usage": gen["usage"]}
        return out

    return answer
