import json
import threading

import httpx
import pytest

from ragspec.models import CaseResult, MetricSummary, Run, RunMeta
from ragspec.runner import save_run
from ragspec.ui import discover_configs, make_server, saved

CONFIG = """\
# A tiny test config.
dataset: questions.jsonl
pipeline:
  target: ui_test_rag:answer
k: [1]
judge:
  enabled: false
"""


@pytest.fixture
def client(tmp_path):
    (tmp_path / "ragspec.yaml").write_text(CONFIG)
    (tmp_path / "notes.yaml").write_text("just: some other yaml\n")
    (tmp_path / "other.json").write_text('{"not": "a run"}')
    (tmp_path / "questions.jsonl").write_text(
        json.dumps({"id": "q1", "question": "Where?", "relevant": ["d1"]}) + "\n"
    )
    (tmp_path / "ui_test_rag.py").write_text(
        "def answer(question):\n"
        "    return {'answer': 'Here. <b>' + question, 'contexts': [{'id': 'd1', 'text': 't'}]}\n"
    )
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(base_url=f"http://localhost:{server.server_address[1]}") as c:
        yield c
    server.shutdown()
    server.server_close()


def test_setup_lists_only_real_configs_and_questions(client):
    data = client.get("/api/setup").json()
    assert [c["name"] for c in data["configs"]] == ["ragspec.yaml"]
    assert data["configs"][0]["summary"] == "A tiny test config."
    assert data["configs"][0]["label"] == "ragspec.yaml"
    assert data["configs"][0]["missing_key"] is None
    assert data["questions"][0]["answerable"] is True
    assert data["questions"][0]["id"] == "q1"
    assert data["runs"] == []


def test_ask_scores_a_labelled_and_a_typed_question(client):
    labelled = client.post("/api/ask", json={"question": "Where?", "case_id": "q1",
                                             "configs": ["ragspec.yaml"]}).json()
    assert labelled["ragspec.yaml"]["result"]["metrics"]["recall@1"] == 1.0
    typed = client.post("/api/ask", json={"question": "Why?", "configs": ["ragspec.yaml"]}).json()
    result = typed["ragspec.yaml"]["result"]
    assert result["answer"] == "Here. <b>Why?"
    assert result["metrics"]["recall@1"] is None


def test_ask_rejects_bad_input_and_cross_site_requests(client):
    assert client.post("/api/ask", json={"question": "x", "configs": ["../secret.yaml"]}
                       ).status_code == 400
    assert client.post("/api/ask", json={"question": "", "configs": ["ragspec.yaml"]}
                       ).status_code == 400
    assert client.post("/api/ask", content='{"question": "x"}',
                       headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.get("/api/setup", headers={"Host": "evil.example"}).status_code == 403


def test_report_only_serves_discovered_runs(client):
    assert client.get("/report", params={"run": "../../etc/passwd"}).status_code == 404
    assert client.get("/report", params={"run": "other.json"}).status_code == 404


def test_configs_list_hosted_first_and_baseline_before_pr(tmp_path):
    for name in ["a.yaml", "a.hosted.pr.yaml", "a.hosted.yaml"]:
        (tmp_path / name).write_text(CONFIG)
    assert list(discover_configs(tmp_path)) == ["a.hosted.yaml", "a.hosted.pr.yaml", "a.yaml"]


def _run(label, faithfulness):
    meta = RunMeta(label=label, created_at="2026-01-01T00:00:00+00:00", ragspec_version="0",
                   pipeline="p", dataset="d", dataset_sha="x", n_cases=20, k=[1])
    cases = [CaseResult(case_id=f"q{i}", question="q", metrics={"faithfulness": faithfulness})
             for i in range(20)]
    return Run(meta=meta, cases=cases,
               summary={"faithfulness": MetricSummary(mean=faithfulness, n=20)})


def test_saved_results_compare_two_runs_without_model_calls(tmp_path):
    save_run(_run("before", 1.0), tmp_path / "a.json")
    save_run(_run("after", 0.5), tmp_path / "b.json")
    data = saved(tmp_path, "a.json", "b.json")
    assert data["base"]["label"] == "before"
    assert data["comparison"]["metrics"][0]["status"] == "regressed"
    assert len(data["comparison"]["failures"]) == 1
    with pytest.raises(ValueError):
        saved(tmp_path, "a.json", "../../secret.json")
