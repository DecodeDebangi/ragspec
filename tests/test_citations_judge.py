from ragspec.citations import extract_markers, resolve
from ragspec.judge import CachedJudge, canonicalize_citations, finalize
from ragspec.models import Context, JudgeResult


def test_extract_markers_order_and_splitting():
    assert extract_markers("A [d1]. B [d2, d3]; C [d1].") == ["d1", "d2", "d3"]


def test_resolve_ids_and_positions():
    assert resolve(["b"], ["a", "b"]) == {"b": "b"}
    assert resolve(["2"], ["a", "b"]) == {"2": "b"}
    # Out-of-range position stays unresolved and will be flagged invalid.
    assert resolve(["9"], ["a", "b"]) == {"9": "9"}
    # Numeric ids that exist win over positions in auto mode.
    assert resolve(["2"], ["1", "2"]) == {"2": "2"}
    assert resolve(["2"], ["1", "2"], style="position") == {"2": "2"}
    assert resolve(["1"], ["7", "8"], style="position") == {"1": "7"}


def test_canonicalize_rewrites_positional_markers():
    ctx = [Context(id="doc-a", text=""), Context(id="doc-b", text="")]
    text, cited = canonicalize_citations("X [2]. Y [1, 2].", ctx, r"\[([^\[\]]+)\]", "auto")
    assert text == "X [doc-b]. Y [doc-a, doc-b]."
    assert cited == ["doc-b", "doc-a"]


def test_finalize_clamps_invented_ids():
    ctx = [Context(id="a", text=""), Context(id="b", text="")]
    data = {"abstained": False, "claims": [
        {"text": "x", "verdict": "supported", "reason": "", "cited_ids": ["a", "zzz"],
         "supporting_ids": ["a", "nope"]},
        {"text": "y", "verdict": "supported", "reason": "", "cited_ids": [],
         "supporting_ids": ["nope"]},
    ]}
    r = finalize("m", data, ctx, cited_ids=["a", "ghost"])
    assert r.claims[0].cited_ids == ["a"]
    assert r.claims[0].supporting_ids == ["a"]
    # "supported" with no real supporting context is downgraded.
    assert r.claims[1].verdict == "unsupported"
    assert r.invalid_citations == ["ghost"]


def test_finalize_repairs_blank_citations_and_false_abstention():
    ctx = [Context(id="plans", text=""), Context(id="faq", text="")]
    answer = "Team costs $12 per seat per month [plans]. It includes a free trial [faq]."
    data = {"abstained": True, "claims": [
        {"text": "Team costs $12 per seat per month", "verdict": "supported", "reason": "",
         "cited_ids": [], "supporting_ids": ["plans"]},
        {"text": "Team includes a free trial", "verdict": "unsupported", "reason": "",
         "cited_ids": [], "supporting_ids": []},
        {"text": "Nimbus has an Android app", "verdict": "unsupported", "reason": "",
         "cited_ids": [], "supporting_ids": []},
    ]}
    r = finalize("m", data, ctx, ["plans", "faq"], answer)
    assert [c.cited_ids for c in r.claims] == [["plans"], ["faq"], []]
    assert r.abstained is False
    # Without citations in the answer there is nothing to recover.
    r = finalize("m", data, ctx, [], "Team costs $12 per seat per month.")
    assert r.claims[0].cited_ids == []


class CountingJudge:
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def judge(self, question, contexts, answer, cited_ids):
        self.calls += 1
        return JudgeResult(model=self.model, input_tokens=10, output_tokens=5)


async def test_cached_judge(tmp_path):
    inner = CountingJudge()
    judge = CachedJudge(inner, tmp_path)
    ctx = [Context(id="a", text="t")]
    first = await judge.judge("q", ctx, "ans", [])
    second = await judge.judge("q", ctx, "ans", [])
    third = await judge.judge("q", ctx, "different", [])
    assert inner.calls == 2
    assert not first.cached and second.cached and not third.cached
    assert second.input_tokens == 0


def test_ollama_judge_parses_response_and_reports_connection_errors():
    import asyncio
    import json

    import httpx

    from ragspec.judge import OllamaJudge

    sent = {}
    verdicts = {"abstained": False, "claims": [
        {"text": "x", "cited_ids": ["a"], "supporting_ids": ["a"], "reason": "", "verdict":
         "supported"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(verdicts)},
                                         "prompt_eval_count": 120, "eval_count": 30})

    judge = OllamaJudge("qwen2.5:3b")
    judge.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = asyncio.run(judge.judge("q", [Context(id="a", text="t")], "x [a]", ["a"]))
    assert sent["model"] == "qwen2.5:3b" and sent["format"]["required"] == ["abstained", "claims"]
    assert result.error is None and result.claims[0].verdict == "supported"
    assert (result.input_tokens, result.output_tokens) == (120, 30)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    judge.client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    result = asyncio.run(judge.judge("q", [], "x", []))
    assert "is it running" in result.error


def _hosted_judge(handler):
    import httpx

    from ragspec.judge import OpenAICompatibleJudge

    judge = OpenAICompatibleJudge("m", "https://llm.example/v1/", "secret", backoff_s=0)
    judge.client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                     headers={"Authorization": "Bearer secret"})
    return judge


def _hosted_reply(content, finish_reason="stop"):
    import httpx

    return httpx.Response(200, json={
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    })


VERDICTS = (
    '{"abstained": false, "claims": [{"text": "x", "cited_ids": ["a"], "supporting_ids": ["a"], '
    '"reason": "", "verdict": "supported"}]}'
)


async def test_hosted_judge_retries_rate_limits_and_parses():
    import json

    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return _hosted_reply(VERDICTS)

    result = await _hosted_judge(handler).judge("q", [Context(id="a", text="t")], "x [a]", ["a"])
    assert len(calls) == 3
    assert str(calls[0].url) == "https://llm.example/v1/chat/completions"
    assert calls[0].headers["authorization"] == "Bearer secret"
    assert json.loads(calls[0].content)["response_format"]["type"] == "json_schema"
    assert result.error is None and result.claims[0].verdict == "supported"
    assert (result.input_tokens, result.output_tokens) == (12, 3)


async def test_hosted_judge_falls_back_to_json_mode_and_reports_bad_output():
    import json

    import httpx

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if bodies[-1]["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": "schema not supported"})
        return _hosted_reply('{"claims": [{"text": "x", "verdict": "maybe"}]}')

    judge = _hosted_judge(handler)
    result = await judge.judge("q", [Context(id="a", text="t")], "x", [])
    assert [b["response_format"]["type"] for b in bodies] == ["json_schema", "json_object"]
    assert "Respond with JSON" in bodies[1]["messages"][0]["content"]
    # A verdict outside the schema is a judge error, never a score.
    assert result.error == "judge returned invalid JSON"


async def test_hosted_judge_surfaces_auth_errors_without_retrying():
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="bad key")

    result = await _hosted_judge(handler).judge("q", [], "x", [])
    assert len(calls) == 1 and "401" in result.error


async def test_hosted_judge_does_not_retry_a_daily_quota():
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, text="Generate content requests per day limit exceeded")

    result = await _hosted_judge(handler).judge("q", [], "x", [])
    assert len(calls) == 1 and "per day" in result.error


def test_make_judge_requires_url_and_key(monkeypatch):
    import pytest

    from ragspec.config import JudgeConfig
    from ragspec.judge import OpenAICompatibleJudge, make_judge

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = JudgeConfig(provider="openai", model="m")
    with pytest.raises(ValueError, match="base_url"):
        make_judge(cfg)
    cfg.base_url = "https://llm.example/v1"
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        make_judge(cfg)
    monkeypatch.setenv("LLM_API_KEY", "secret")
    assert isinstance(make_judge(cfg), OpenAICompatibleJudge)


async def test_product_reaches_the_prompt_and_the_cache_key(tmp_path):
    from ragspec.judge import build_prompt

    assert build_prompt("q", [], "a", 100, "Nimbus").startswith("<product>Nimbus</product>")
    assert "<product>" not in build_prompt("q", [], "a", 100)

    inner = CountingJudge()
    judge = CachedJudge(inner, tmp_path)
    await judge.judge("q", [], "a", [])
    inner.product = "Nimbus"
    await judge.judge("q", [], "a", [])
    assert inner.calls == 2


async def test_hosted_judge_retries_network_failures_then_reports_them():
    import httpx

    calls = []

    def flaky(request):
        calls.append(request)
        if len(calls) < 3:
            raise httpx.ConnectError("no route")
        return _hosted_reply(VERDICTS)

    result = await _hosted_judge(flaky).judge("q", [Context(id="a", text="t")], "x [a]", ["a"])
    assert len(calls) == 3 and result.error is None

    def down(request):
        raise httpx.ConnectError("no route")

    result = await _hosted_judge(down).judge("q", [], "x", [])
    assert "llm api error" in result.error
