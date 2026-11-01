"""A deliberately small RAG app over a fictional product's help center.

Retrieval is a pure-Python BM25 over corpus.jsonl; answers come from Claude, from any hosted
OpenAI-compatible API with provider="openai", or from Ollama with provider="ollama". The two
`prompt` settings let the quickstart show ragspec catching a regression:

  grounded  answer only from the sources and cite them (the baseline)
  concise   a "simplified" prompt from a hypothetical PR: shorter, no citation rule,
            and it lets the model fill gaps from general knowledge
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from functools import cache
from pathlib import Path

import anthropic
import httpx

HERE = Path(__file__).parent

PROMPTS = {
    "grounded": """\
You are the support assistant for Nimbus, a project management tool. Answer the customer's \
question using only the help-center sources provided. After each sentence that states a fact, \
cite the source id in square brackets, e.g. [billing-refunds]. If the sources don't cover the \
question, say you don't know and suggest contacting support. Keep it to 3 sentences.""",
    "concise": """\
You are the friendly support assistant for Nimbus, a project management tool. Answer the \
customer's question in 2 sentences. Be helpful and confident.""",
}


@cache
def corpus() -> list[dict]:
    with (HERE / "corpus.jsonl").open() as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@cache
def index():
    docs = [tokenize(d["title"] + " " + d["text"]) for d in corpus()]
    df = Counter(t for doc in docs for t in set(doc))
    avg_len = sum(map(len, docs)) / len(docs)
    return docs, df, avg_len


def bm25(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> list[tuple[dict, float]]:
    docs, df, avg_len = index()
    n = len(docs)
    scores = []
    for doc, tokens in zip(corpus(), docs, strict=True):
        tf = Counter(tokens)
        score = 0.0
        for term in set(tokenize(query)):
            if term in tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * (k1 + 1) / (
                    tf[term] + k1 * (1 - b + b * len(tokens) / avg_len))
        scores.append((doc, score))
    return sorted(scores, key=lambda s: s[1], reverse=True)[:k]


@cache
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def generate_claude(model: str, system: str, user: str) -> str:
    response = client().beta.messages.create(
        model=model,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def generate_ollama(model: str, system: str, user: str, url: str) -> str:
    try:
        response = httpx.post(f"{url.rstrip('/')}/api/chat", timeout=600, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        })
    except httpx.ConnectError:
        raise RuntimeError(f"cannot reach Ollama at {url} (is the Ollama app running?)") from None
    if response.status_code == 404:
        raise RuntimeError(f"model {model} not found; run: ollama pull {model}")
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def generate_openai(model: str, system: str, user: str, base_url: str, api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"set {api_key_env} to your API key, in .env or your shell")
    for attempt in range(6):
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions", timeout=300,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
        )
        # Free tiers rate-limit hard, so back off and retry.
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 5:
            break
        # A daily quota won't reset within any backoff.
        if "per day" in response.text.lower():
            break
        try:
            wait = float(response.headers["retry-after"])
        except (KeyError, ValueError):
            wait = 5.0 * 2**attempt
        time.sleep(min(wait, 60.0))
    if response.status_code != 200:
        raise RuntimeError(f"llm api error {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def make(top_k: int = 4, prompt: str = "grounded", model: str = "claude-opus-5-5",
         provider: str = "anthropic", ollama_url: str = "http://localhost:11434",
         base_url: str | None = None, api_key_env: str = "LLM_API_KEY"):
    def answer(question: str) -> dict:
        hits = bm25(question, top_k)
        contexts = [{"id": d["id"], "text": f"{d['title']}: {d['text']}", "score": s}
                    for d, s in hits]
        sources = "\n".join(f'<source id="{c["id"]}">{c["text"]}</source>' for c in contexts)
        user = f"{sources}\n\nCustomer: {question}"
        if provider == "ollama":
            text = generate_ollama(model, PROMPTS[prompt], user, ollama_url)
        elif provider == "openai":
            text = generate_openai(model, PROMPTS[prompt], user, base_url, api_key_env)
        else:
            text = generate_claude(model, PROMPTS[prompt], user)
        return {"answer": text, "contexts": contexts}

    return answer
