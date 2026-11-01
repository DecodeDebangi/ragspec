import json

from typer.testing import CliRunner

from ragspec.adapters import _fill, _get_path, coerce_output
from ragspec.cli import app
from ragspec.config import Config, PipelineConfig
from ragspec.judge import finalize
from ragspec.models import Case
from ragspec.runner import load_dataset, run_eval, select_cases

CORPUS = {"d1": "Paris is the capital of France.", "d2": "Berlin is the capital of Germany."}


class KeywordJudge:
    """Marks a claim supported if any context shares a word with it."""

    model = "fake"

    async def judge(self, question, contexts, answer, cited_ids):
        claims = []
        for sentence in filter(None, (s.strip() for s in answer.split("."))):
            words = set(sentence.lower().split())
            supporting = [c.id for c in contexts if words & set(c.text.lower().rstrip(".").split())]
            cited = [i for i in cited_ids if f"[{i}]" in sentence]
            claims.append({"text": sentence, "reason": "", "cited_ids": cited,
                           "supporting_ids": supporting,
                           "verdict": "supported" if supporting else "unsupported"})
        return finalize(self.model, {"abstained": False, "claims": claims}, contexts, cited_ids)


async def pipeline(case: Case):
    doc = "d1" if "france" in case.question.lower() else "d2"
    return coerce_output({"answer": f"paris [{doc}]. unicorns exist",
                          "contexts": [{"id": doc, "text": CORPUS[doc]}]})


async def test_run_eval_end_to_end():
    cfg = Config(dataset="x", pipeline=PipelineConfig(target="t:f"), k=[1])
    cases = [Case(id="1", question="Capital of France?", relevant=["d1"]),
             Case(id="2", question="Capital of Spain?", relevant=["d9"])]
    run = await run_eval(cfg, cases, pipeline, KeywordJudge(), label="t")
    assert run.summary["recall@1"].mean == 0.5
    # Case 1: 1 of 2 claims supported. Case 2: the retrieved doc (Berlin) supports neither.
    assert run.summary["faithfulness"].mean == 0.25
    by_id = {c.case_id: c for c in run.cases}
    assert by_id["1"].metrics["citation_precision"] == 1.0
    assert by_id["2"].metrics["citation_precision"] == 0.0  # cites d2, which doesn't say paris
    assert run.latency_ms["p50"] is not None
    assert run.errors == 0


async def test_pipeline_errors_are_recorded():
    async def broken(case):
        raise RuntimeError("boom")

    cfg = Config(dataset="x", pipeline=PipelineConfig(target="t:f"), k=[1])
    run = await run_eval(cfg, [Case(id="1", question="q")], broken, None)
    assert run.errors == 1 and "boom" in run.cases[0].error


def test_select_cases_sample_is_seeded():
    cases = [Case(id=str(i), question="q", tags=["a" if i % 2 else "b"]) for i in range(50)]
    assert [c.id for c in select_cases(cases, sample=5, seed=1)] == \
           [c.id for c in select_cases(cases, sample=5, seed=1)]
    assert all("a" in c.tags for c in select_cases(cases, tags=["a"]))


def test_http_helpers():
    case = Case(id="7", question="hi")
    assert _fill({"q": "{question}", "n": [{"id": "{id}"}], "k": 3}, case) == \
           {"q": "hi", "n": [{"id": "7"}], "k": 3}
    assert _get_path({"a": {"b": [{"c": 1}]}}, "a.b.0.c") == 1


def test_coerce_output_accepts_strings():
    out = coerce_output({"answer": "x", "contexts": ["first", "second"]})
    assert [c.id for c in out.contexts] == ["1", "2"]


def test_cli_run_and_compare(tmp_path):
    (tmp_path / "pipe.py").write_text(
        "def make(good=True):\n"
        "    def answer(question):\n"
        "        ids = ['d1', 'd2'] if good else ['d2', 'd1']\n"
        "        return {'contexts': [{'id': i, 'text': ''} for i in ids]}\n"
        "    return answer\n"
    )
    rows = [{"id": str(i), "question": f"q{i}", "relevant": ["d1"]} for i in range(30)]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for name, good in [("good", True), ("bad", False)]:
        (tmp_path / f"{name}.yaml").write_text(
            f"dataset: data.jsonl\npipeline: {{target: 'pipe:make', options: {{good: {good}}}}}\n"
            "k: [1]\njudge: {enabled: false}\n"
        )
    runner = CliRunner()
    r1 = runner.invoke(app, ["run", "-c", str(tmp_path / "good.yaml"), "--out",
                             str(tmp_path / "base.json")])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["run", "-c", str(tmp_path / "bad.yaml"), "--out",
                             str(tmp_path / "head.json"), "--baseline", str(tmp_path / "base.json"),
                             "--markdown", str(tmp_path / "c.md"),
                             "--report", str(tmp_path / "r.html")])
    assert r2.exit_code == 1, r2.output
    assert "recall@1 dropped" in (tmp_path / "c.md").read_text()
    assert (tmp_path / "r.html").stat().st_size > 1000
    assert load_dataset(tmp_path / "data.jsonl")[0].relevant == {"d1": 1.0}


def test_load_config_reads_dotenv_and_real_env_wins(tmp_path, monkeypatch):
    from ragspec.config import load_config

    # setenv first so monkeypatch restores these even though .env sets them again.
    monkeypatch.setenv("FROM_DOTENV", "x")
    monkeypatch.delenv("FROM_DOTENV")
    monkeypatch.setenv("FROM_SHELL", "shell")
    (tmp_path / ".env").write_text("FROM_DOTENV=dotenv\nFROM_SHELL=dotenv\n")
    (tmp_path / "sub").mkdir()
    config = tmp_path / "sub" / "ragspec.yaml"
    config.write_text(
        "dataset: q.jsonl\n"
        "pipeline:\n  type: http\n  url: http://x\n"
        "  headers: {a: '${FROM_DOTENV}', b: '${FROM_SHELL}'}\n"
    )
    monkeypatch.chdir(tmp_path / "sub")
    assert load_config(config).pipeline.headers == {"a": "dotenv", "b": "shell"}
