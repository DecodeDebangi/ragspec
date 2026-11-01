"""Download BEIR datasets and build the indexes the benchmark pipelines query.

    uv run python bench/beir.py scifact nfcorpus fiqa

For each dataset this writes:
  bench/data/<name>/              raw BEIR files (corpus, queries, qrels)
  bench/evals/<name>.jsonl        test queries in ragspec format, with graded qrels
  bench/data/<name>/bm25/         bm25s lexical index
  Postgres table docs_<name>      pgvector embeddings with an HNSW index
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import httpx

HERE = Path(__file__).parent
DATA = HERE / "data"
EVALS = HERE / "evals"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DSN = "postgresql://ragspec:ragspec@localhost:5433/ragspec"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


def download(name: str) -> Path:
    target = DATA / name
    if (target / "corpus.jsonl").exists():
        return target
    print(f"[{name}] downloading")
    resp = httpx.get(BEIR_URL.format(name=name), follow_redirects=True, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.namelist():
            rel = Path(member).relative_to(name) if member.startswith(name + "/") else None
            if rel is None or member.endswith("/"):
                continue
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(member))
    return target


def load_corpus(path: Path) -> list[dict]:
    with (path / "corpus.jsonl").open() as f:
        return [json.loads(line) for line in f]


def doc_text(doc: dict) -> str:
    title = doc.get("title") or ""
    return f"{title}\n{doc['text']}".strip()


def write_eval(name: str, path: Path) -> int:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (path / "qrels" / "test.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    queries = {}
    with (path / "queries.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    EVALS.mkdir(parents=True, exist_ok=True)
    n = 0
    with (EVALS / f"{name}.jsonl").open("w") as out:
        for qid in sorted(qrels, key=lambda x: (len(x), x)):
            if qid in queries and any(g > 0 for g in qrels[qid].values()):
                out.write(json.dumps({"id": qid, "question": queries[qid],
                                      "relevant": qrels[qid], "tags": [name]}) + "\n")
                n += 1
    return n


def build_bm25(name: str, corpus: list[dict]) -> None:
    import bm25s
    import Stemmer

    out = DATA / name / "bm25"
    if out.exists():
        return
    print(f"[{name}] bm25 index over {len(corpus):,} docs")
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize([doc_text(d) for d in corpus], stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save(out, corpus=[{"id": d["_id"]} for d in corpus])


def build_vectors(name: str, corpus: list[dict]) -> None:
    import psycopg
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    table = f"docs_{name}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]
        if exists and conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == len(corpus):
            return
        print(f"[{name}] embedding {len(corpus):,} docs with {EMBED_MODEL}")
        model = SentenceTransformer(EMBED_MODEL)
        texts = [doc_text(d) for d in corpus]
        vectors = model.encode(texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} (id text PRIMARY KEY, title text, body text, "
                     f"embedding vector({EMBED_DIM}))")
        with conn.cursor().copy(
            f"COPY {table} (id, title, body, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["text", "text", "text", "vector"])
            for doc, vec in zip(corpus, vectors, strict=True):
                copy.write_row((doc["_id"], doc.get("title") or "", doc["text"], vec))
        print(f"[{name}] building HNSW index")
        conn.execute(f"CREATE INDEX ON {table} USING hnsw (embedding vector_cosine_ops) "
                     f"WITH (m = 16, ef_construction = 128)")
        conn.execute(f"ANALYZE {table}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    for name in args.datasets:
        path = download(name)
        corpus = load_corpus(path)
        n = write_eval(name, path)
        print(f"[{name}] {len(corpus):,} docs, {n} test queries -> bench/evals/{name}.jsonl")
        build_bm25(name, corpus)
        build_vectors(name, corpus)


if __name__ == "__main__":
    main()
