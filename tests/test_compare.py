from ragspec.compare import bootstrap_ci, compare_runs, to_markdown
from ragspec.config import GateConfig
from ragspec.models import CaseResult, MetricSummary, Run, RunMeta


def make_run(values: dict[str, list[float]], label="r", dataset_sha="abc") -> Run:
    n = len(next(iter(values.values())))
    cases = [
        CaseResult(case_id=f"q{i}", question=f"question {i}",
                   metrics={m: v[i] for m, v in values.items()})
        for i in range(n)
    ]
    summary = {m: MetricSummary(mean=sum(v) / len(v), n=len(v)) for m, v in values.items()}
    meta = RunMeta(label=label, created_at="2026-01-01T00:00:00+00:00", ragspec_version="0",
                   pipeline="p", dataset="d", dataset_sha=dataset_sha, n_cases=n, k=[5])
    return Run(meta=meta, summary=summary, cases=cases)


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.0, 0.2] * 10)
    assert lo < 0.16 < hi


def test_consistent_drop_is_regression():
    base = make_run({"faithfulness": [1.0] * 40})
    head = make_run({"faithfulness": [1.0] * 30 + [0.5] * 10})
    cmp = compare_runs(base, head)
    md = cmp.metrics[0]
    assert md.status == "regressed"
    assert md.n_worse == 10
    assert not cmp.passed
    assert cmp.worst_cases[0].delta == -0.5


def test_single_noisy_question_is_not_regression():
    base = make_run({"recall@5": [0.5, 1.0] * 20})
    head = make_run({"recall@5": [0.5, 1.0] * 19 + [0.5, 0.0]})
    assert compare_runs(base, head).metrics[0].status == "unchanged"


def test_small_drop_within_tolerance_passes():
    base = make_run({"recall@5": [1.0] * 100})
    head = make_run({"recall@5": [1.0] * 99 + [0.5]})
    cmp = compare_runs(base, head, GateConfig(default_max_drop=0.01))
    assert cmp.passed


def test_improvement_and_ungated_metrics():
    base = make_run({"recall@5": [0.0] * 30, "mrr": [1.0] * 30})
    head = make_run({"recall@5": [1.0] * 30, "mrr": [0.0] * 30})
    cmp = compare_runs(base, head, GateConfig(metrics=["recall@5"]))
    status = {m.metric: m.status for m in cmp.metrics}
    assert status == {"recall@5": "improved", "mrr": "regressed"}
    assert cmp.passed  # mrr is not gated


def test_min_threshold_and_markdown():
    base = make_run({"faithfulness": [0.7] * 10}, label="main")
    head = make_run({"faithfulness": [0.7] * 10}, label="pr-1", dataset_sha="other")
    cmp = compare_runs(base, head, GateConfig(min={"faithfulness": 0.8}))
    assert not cmp.passed
    md = to_markdown(cmp)
    assert "ragspec-report" in md and "`pr-1` vs `main`" in md
    assert "below the minimum" in md
    assert "dataset differs" in md


def test_unpaired_questions_ignored():
    base = make_run({"mrr": [1.0, 1.0]})
    head = make_run({"mrr": [1.0, 1.0, 0.0]})
    md = compare_runs(base, head).metrics[0]
    assert md.n_paired == 2 and md.delta == 0
