"""LLM-as-judge: split an answer into claims and check each against the retrieved contexts.

One judge call per answer returns claim-level verdicts. Faithfulness and citation scores are
then computed deterministically from those verdicts (see metrics.generation_metrics), so the
judge never outputs a number directly.

Verdicts are cached on disk, keyed by judge model + prompt version + question + contexts +
answer. Re-running an unchanged pipeline costs nothing and yields identical scores, which is
what makes run-to-run comparisons in CI meaningful.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import anthropic
import httpx

from .citations import extract_markers, resolve
from .models import Claim, Context, JudgeResult

if TYPE_CHECKING:
    from .config import JudgeConfig

# Bump when the prompt or the post-processing in finalize() changes; it invalidates the cache.
PROMPT_VERSION = "2026-11-01.3"

# USD per million tokens (input, output). Used only for the cost line in reports.
PRICING = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM_PROMPT = """\
You audit answers produced by a retrieval-augmented generation (RAG) system. You judge only \
whether the answer is backed by the retrieved contexts it was given, not whether it is true \
in the world. Use no outside knowledge: a claim that is correct but absent from the contexts \
is unsupported.

The contexts come from one knowledge base about one product or organization, and they may not \
name it. Treat that name and a first-person voice ("we", "our") as given: a claim is never \
unsupported just because it names the product or speaks for it, and being part of that product \
is not a claim of its own ("the projects are Nimbus projects"). Every other fact still needs \
backing.

Work through the answer as follows.

1. Split the answer into atomic claims: short, self-contained factual statements, each \
checkable on its own. Resolve pronouns so every claim stands alone. Skip filler, hedges, \
restatements of the question, statements about the answer itself, and statements about \
what the contexts do or do not contain ("the sources don't mention X"). List each fact once.
2. For each claim, set cited_ids to the context ids the answer cites for it: the bracketed \
markers on the sentence that makes the claim (or on the clause, if a sentence carries several \
markers for different parts). Only use ids that appear in the answer's markers. Leave the list \
empty when the claim has no citation.
3. For each claim, set supporting_ids to every context whose text, read on its own, states or \
directly entails the claim. Partial support is not support: every part of the claim, \
including numbers, entities, direction of effect and qualifiers, must be backed. A number \
calculated from a context is entailed only if the calculation is correct: work it out yourself, \
applying every rule the context states, such as caps. If your result matches the claim, the \
verdict is "supported" and that context goes in supporting_ids, even though the context never \
states the result. If it doesn't match, the claim is not supported.
4. Give a short reason, then the verdict: "supported" if at least one context supports the \
claim, "contradicted" if a context states the opposite, otherwise "unsupported".

Set abstained to true only when the answer gives no substantive answer: it declines, or says \
the contexts don't cover the question. An answer that gives some facts and says the rest is \
unknown is not an abstention; list its claims as usual."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "cited_ids": {"type": "array", "items": {"type": "string"}},
                    "supporting_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "unsupported", "contradicted"],
                    },
                },
                "required": ["text", "cited_ids", "supporting_ids", "reason", "verdict"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["abstained", "claims"],
    "additionalProperties": False,
}


class Judge(Protocol):
    model: str

    async def judge(
        self, question: str, contexts: list[Context], answer: str, cited_ids: list[str]
    ) -> JudgeResult: ...


def canonicalize_citations(
    answer: str, contexts: list[Context], pattern: str, style: str
) -> tuple[str, list[str]]:
    """Rewrite markers so they name context ids ("[2]" -> "[doc_17]") and list the cited ids."""
    ids = [c.id for c in contexts]
    mapping = resolve(extract_markers(answer, pattern), ids, style)

    def rewrite(match: re.Match[str]) -> str:
        parts = re.split(r"\s*[,;]\s*", match.group(1).strip())
        return "[" + ", ".join(mapping.get(p.strip().strip("\"'"), p) for p in parts) + "]"

    cited: list[str] = []
    for doc_id in mapping.values():
        if doc_id not in cited:
            cited.append(doc_id)
    return re.sub(pattern, rewrite, answer), cited


def build_prompt(question: str, contexts: list[Context], answer: str, max_chars: int) -> str:
    blocks = "\n".join(
        f'<context id="{c.id}">\n{c.text[:max_chars]}\n</context>' for c in contexts
    )
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<contexts>\n{blocks or '(no contexts were retrieved)'}\n</contexts>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICING.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


class ClaudeJudge:
    def __init__(self, model: str, effort: str = "medium", max_context_chars: int = 6000):
        self.model = model
        self.effort = effort
        self.max_context_chars = max_context_chars
        self.client = anthropic.AsyncAnthropic(max_retries=6)

    async def judge(
        self, question: str, contexts: list[Context], answer: str, cited_ids: list[str]
    ) -> JudgeResult:
        prompt = build_prompt(question, contexts, answer, self.max_context_chars)
        try:
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
                # Re-run declined requests on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.BadRequestError as e:
            return JudgeResult(model=self.model, error=f"bad request: {e.message}")
        except anthropic.APIStatusError as e:
            return JudgeResult(model=self.model, error=f"api error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return JudgeResult(model=self.model, error=f"connection error: {e}")

        usage = dict(input_tokens=response.usage.input_tokens,
                     output_tokens=response.usage.output_tokens)
        if response.stop_reason == "refusal":
            return JudgeResult(model=self.model, error="judge refused", **usage)
        if response.stop_reason == "max_tokens":
            return JudgeResult(model=self.model, error="judge hit max_tokens", **usage)

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return JudgeResult(model=self.model, error="judge returned invalid JSON", **usage)
        return finalize(self.model, data, contexts, cited_ids, answer, **usage)


class OllamaJudge:
    """The same judge prompt and schema, run on a local model served by Ollama."""

    def __init__(self, model: str, url: str = "http://localhost:11434",
                 max_context_chars: int = 6000, timeout_s: float = 900.0):
        self.model = model
        self.url = url.rstrip("/")
        self.max_context_chars = max_context_chars
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def judge(
        self, question: str, contexts: list[Context], answer: str, cited_ids: list[str]
    ) -> JudgeResult:
        prompt = build_prompt(question, contexts, answer, self.max_context_chars)
        try:
            response = await self.client.post(f"{self.url}/api/chat", json={
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}],
                # Ollama constrains decoding to this JSON schema.
                "format": OUTPUT_SCHEMA,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            })
        except httpx.ConnectError:
            return JudgeResult(model=self.model,
                               error=f"cannot reach Ollama at {self.url} (is it running?)")
        except httpx.HTTPError as e:
            return JudgeResult(model=self.model, error=f"ollama error: {e!r}")
        if response.status_code != 200:
            detail = response.text[:200]
            return JudgeResult(model=self.model,
                               error=f"ollama error {response.status_code}: {detail}")

        body = response.json()
        usage = dict(input_tokens=body.get("prompt_eval_count", 0),
                     output_tokens=body.get("eval_count", 0))
        if body.get("done_reason") == "length":
            return JudgeResult(model=self.model, error="judge hit max_tokens", **usage)
        try:
            data = json.loads(body["message"]["content"])
        except (KeyError, json.JSONDecodeError):
            return JudgeResult(model=self.model, error="judge returned invalid JSON", **usage)
        return finalize(self.model, data, contexts, cited_ids, answer, **usage)


class OpenAICompatibleJudge:
    """The same judge prompt and schema, run on a hosted OpenAI-compatible chat API.

    Free tiers rate-limit hard, so 429s and transient 5xx errors back off and retry. Not every
    model accepts a JSON schema, so a 400 falls back once to plain JSON mode with the schema in
    the prompt; finalize() then treats malformed output as a judge error, never as a score.
    """

    RETRYABLE = {429, 500, 502, 503, 504}
    ATTEMPTS = 6

    def __init__(self, model: str, base_url: str, api_key: str, max_context_chars: int = 6000,
                 timeout_s: float = 300.0, backoff_s: float = 5.0):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.max_context_chars = max_context_chars
        self.backoff_s = backoff_s
        self.use_schema = True
        self.client = httpx.AsyncClient(
            timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"}
        )

    def _body(self, prompt: str) -> dict:
        system = SYSTEM_PROMPT
        if self.use_schema:
            response_format = {"type": "json_schema", "json_schema": {
                "name": "verdicts", "schema": OUTPUT_SCHEMA}}
        else:
            response_format = {"type": "json_object"}
            system += f"\n\nRespond with JSON matching this schema:\n{json.dumps(OUTPUT_SCHEMA)}"
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": response_format,
        }

    async def _post(self, prompt: str) -> httpx.Response:
        for attempt in range(self.ATTEMPTS):
            response = await self.client.post(self.url, json=self._body(prompt))
            if response.status_code == 400 and self.use_schema:
                self.use_schema = False
                continue
            if response.status_code not in self.RETRYABLE or attempt == self.ATTEMPTS - 1:
                return response
            # A daily quota won't reset within any backoff, so report it right away.
            if "per day" in response.text.lower():
                return response
            try:
                wait = float(response.headers["retry-after"])
            except (KeyError, ValueError):
                wait = self.backoff_s * 2**attempt
            await asyncio.sleep(min(wait, 60.0))
        return response

    async def judge(
        self, question: str, contexts: list[Context], answer: str, cited_ids: list[str]
    ) -> JudgeResult:
        prompt = build_prompt(question, contexts, answer, self.max_context_chars)
        try:
            response = await self._post(prompt)
        except httpx.HTTPError as e:
            return JudgeResult(model=self.model, error=f"llm api error: {e!r}")
        if response.status_code != 200:
            detail = response.text[:200]
            return JudgeResult(model=self.model,
                               error=f"llm api error {response.status_code}: {detail}")

        body = response.json()
        tokens = body.get("usage") or {}
        usage = dict(input_tokens=tokens.get("prompt_tokens", 0),
                     output_tokens=tokens.get("completion_tokens", 0))
        try:
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                return JudgeResult(model=self.model, error="judge hit max_tokens", **usage)
            data = json.loads(choice["message"]["content"])
            return finalize(self.model, data, contexts, cited_ids, answer, **usage)
        except (KeyError, IndexError, TypeError, ValueError):
            return JudgeResult(model=self.model, error="judge returned invalid JSON", **usage)


def make_judge(cfg: JudgeConfig) -> Judge:
    if cfg.provider == "ollama":
        return OllamaJudge(cfg.model, cfg.ollama_url, cfg.max_context_chars)
    if cfg.provider == "openai":
        if not cfg.base_url:
            raise ValueError("judge.base_url is required with provider: openai")
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise ValueError(f"set {cfg.api_key_env} to your API key, in .env or your shell")
        return OpenAICompatibleJudge(cfg.model, cfg.base_url, api_key, cfg.max_context_chars)
    return ClaudeJudge(cfg.model, cfg.effort, cfg.max_context_chars)


_WORD = re.compile(r"[a-z0-9$%.]+")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?](?=\s|$)|$)(?:\s*\[[^\[\]]+\])*")


def _words(text: str) -> set[str]:
    return {w.strip(".") for w in _WORD.findall(text.lower())} - {""}


def citations_by_sentence(answer: str) -> list[tuple[set[str], list[str]]]:
    """Split a canonicalized answer into sentences, each with its words and cited ids."""
    out = []
    for sentence in _SENTENCE.findall(answer):
        ids = [i.strip() for group in re.findall(r"\[([^\[\]]+)\]", sentence)
               for i in re.split(r"\s*[,;]\s*", group)]
        out.append((_words(re.sub(r"\[[^\[\]]+\]", " ", sentence)), ids))
    return out


def _sentence_citations(claim: str, sentences: list[tuple[set[str], list[str]]]) -> list[str]:
    """Citations on the answer sentence a claim was taken from, if one clearly matches."""
    words = _words(claim)
    if not words:
        return []
    best, score = [], 0.0
    for sentence_words, ids in sentences:
        overlap = len(words & sentence_words) / len(words)
        if overlap > score:
            best, score = ids, overlap
    return best if score >= 0.6 else []


def finalize(
    model: str,
    data: dict,
    contexts: list[Context],
    cited_ids: list[str],
    answer: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> JudgeResult:
    """Clamp judge output to ids that actually exist so it cannot invent citations.

    Also repairs two bookkeeping slips that weaker (local) judges make: a claim whose
    citation was left blank gets the markers of the answer sentence it came from, and an
    answer with checkable claims is not an abstention.
    """
    context_ids = {c.id for c in contexts}
    cited = set(cited_ids)
    claims = [
        Claim(
            text=c["text"],
            verdict=c["verdict"],
            reason=c.get("reason", ""),
            cited_ids=[i for i in dict.fromkeys(c.get("cited_ids", [])) if i in cited],
            supporting_ids=[i for i in dict.fromkeys(c.get("supporting_ids", []))
                            if i in context_ids],
        )
        for c in data.get("claims", [])
    ]
    sentences = citations_by_sentence(answer) if cited_ids else []
    for claim in claims:
        if not claim.cited_ids and sentences:
            claim.cited_ids = [i for i in dict.fromkeys(_sentence_citations(claim.text, sentences))
                               if i in cited]
        # A "supported" verdict needs a supporting context; keep the two fields consistent.
        if claim.verdict == "supported" and not claim.supporting_ids:
            claim.verdict = "unsupported"
    return JudgeResult(
        model=model,
        claims=claims,
        abstained=bool(data.get("abstained")) and not claims,
        cited_ids=cited_ids,
        invalid_citations=[i for i in cited_ids if i not in context_ids],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class CachedJudge:
    """Wraps a judge with a content-addressed on-disk cache."""

    def __init__(self, inner: Judge, cache_dir: Path):
        self.inner = inner
        self.model = inner.model
        self.cache_dir = cache_dir

    def _key(self, question: str, contexts: list[Context], answer: str) -> str:
        payload = json.dumps(
            {
                "v": PROMPT_VERSION,
                "model": self.model,
                "q": question,
                "ctx": [[c.id, c.text] for c in contexts],
                "a": answer,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def judge(
        self, question: str, contexts: list[Context], answer: str, cited_ids: list[str]
    ) -> JudgeResult:
        key = self._key(question, contexts, answer)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            result = JudgeResult.model_validate_json(path.read_text())
            # Cached hits cost nothing this run.
            return result.model_copy(update={"cached": True, "input_tokens": 0,
                                             "output_tokens": 0})
        result = await self.inner.judge(question, contexts, answer, cited_ids)
        if result.error is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result.model_dump_json())
        return result
