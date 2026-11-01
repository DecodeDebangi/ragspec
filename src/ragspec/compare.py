"""Compare two runs question by question and decide whether the head run regressed.

Deltas are computed on *paired* questions (same case id, metric present in both runs), and a
paired bootstrap gives a confidence interval on the mean delta. A metric counts as regressed
only when the whole interval sits below zero and the drop exceeds the tolerated `max_drop`,
so one noisy question can't fail a PR but a consistent small drop can.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel

from .config import GateConfig
from .metrics import GENERATION_METRICS
from .models import Run

# Reported but never gated: higher is not better for these.
INFORMATIONAL = {"abstention_rate"}

Status = Literal["regressed", "improved", "unchanged", "new", "missing"]


class MetricDelta(BaseModel):
    metric: str
    base: float | None
    head: float | None
    delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_paired: int = 0
    n_better: int = 0
    n_worse: int = 0
    status: Status = "unchanged"
    gated: bool = True
    below_min: bool = False


class CaseChange(BaseModel):
    case_id: str
    question: str
    metric: str
    base: float
    head: float

    @property
    def delta(self) -> float:
        return self.head - self.base


class Comparison(BaseModel):
    base_label: str
    head_label: str
    dataset_changed: bool
    n_base: int
    n_head: int
    metrics: list[MetricDelta]
    worst_cases: list[CaseChange]
    best_cases: list[CaseChange]
    latency: dict[str, dict[str, float | None]]
    failures: list[str]
    head_cost_usd: float | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


def bootstrap_ci(
    deltas: list[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of paired deltas."""
    n = len(deltas)
    if n == 1:
        return deltas[0], deltas[0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(resamples))
    alpha = (1 - confidence) / 2
    return means[int(alpha * resamples)], means[min(resamples - 1, int((1 - alpha) * resamples))]


def compare_runs(base: Run, head: Run, gate: GateConfig | None = None) -> Comparison:
    gate = gate or GateConfig()
    base_cases = {c.case_id: c for c in base.cases}
    head_cases = {c.case_id: c for c in head.cases}
    shared_ids = sorted(base_cases.keys() & head_cases.keys())

    names = list(head.summary) + [m for m in base.summary if m not in head.summary]
    deltas: list[MetricDelta] = []
    changes: list[CaseChange] = []
    for name in names:
        b = base.summary.get(name)
        h = head.summary.get(name)
        md = MetricDelta(
            metric=name,
            base=b.mean if b else None,
            head=h.mean if h else None,
            gated=name not in INFORMATIONAL and (not gate.metrics or name in gate.metrics),
        )
        if name in gate.min and md.head is not None and md.head < gate.min[name]:
            md.below_min = True
        if b is None:
            md.status = "new"
        elif h is None:
            md.status = "missing"
        else:
            pairs = []
            for cid in shared_ids:
                bv = base_cases[cid].metrics.get(name)
                hv = head_cases[cid].metrics.get(name)
                if bv is not None and hv is not None:
                    pairs.append((cid, bv, hv))
                    if name not in INFORMATIONAL and hv != bv:
                        changes.append(CaseChange(case_id=cid, question=head_cases[cid].question,
                                                  metric=name, base=bv, head=hv))
            if pairs:
                diffs = [hv - bv for _, bv, hv in pairs]
                md.n_paired = len(pairs)
                md.delta = sum(diffs) / len(diffs)
                md.ci_low, md.ci_high = bootstrap_ci(diffs, gate.confidence)
                md.n_better = sum(1 for d in diffs if d > 0)
                md.n_worse = sum(1 for d in diffs if d < 0)
                tolerance = gate.max_drop.get(name, gate.default_max_drop)
                if md.ci_high < 0 and md.delta < -tolerance:
                    md.status = "regressed"
                elif md.ci_low > 0:
                    md.status = "improved"
        deltas.append(md)

    failures = []
    for md in deltas:
        if not md.gated:
            continue
        if md.status == "regressed":
            failures.append(
                f"{md.metric} dropped {md.delta:+.3f} "
                f"(95% CI {md.ci_low:+.3f} to {md.ci_high:+.3f}, n={md.n_paired})"
            )
        if md.below_min:
            floor = gate.min[md.metric]
            failures.append(f"{md.metric} = {md.head:.3f} is below the minimum {floor}")

    worst = _one_per_case([c for c in changes if c.delta < 0], names, worst_first=True)[:10]
    best = _one_per_case([c for c in changes if c.delta > 0], names, worst_first=False)[:5]

    return Comparison(
        base_label=base.meta.label,
        head_label=head.meta.label,
        dataset_changed=base.meta.dataset_sha != head.meta.dataset_sha,
        n_base=len(base.cases),
        n_head=len(head.cases),
        metrics=deltas,
        worst_cases=worst,
        best_cases=best,
        latency={k: {"base": base.latency_ms.get(k), "head": head.latency_ms.get(k)}
                 for k in ("p50", "p95")},
        failures=failures,
        head_cost_usd=head.cost.get("judge_usd"),
    )


def _one_per_case(changes: list[CaseChange], names: list[str],
                  worst_first: bool) -> list[CaseChange]:
    """Keep each question's biggest move, preferring answer-level metrics over retrieval ones.

    Metrics are correlated (a question that loses recall@1 also loses ndcg@1), so listing
    every metric would repeat the same question several times.
    """
    def priority(metric: str) -> int:
        if metric in GENERATION_METRICS:
            return GENERATION_METRICS.index(metric)
        deep = sorted((m for m in names if "@" in m), key=lambda m: -int(m.split("@")[1]))
        return len(GENERATION_METRICS) + (deep.index(metric) if metric in deep else len(deep))

    sign = 1 if worst_first else -1
    ranked = sorted(changes, key=lambda c: (sign * c.delta, priority(c.metric)))
    seen: set[str] = set()
    out = []
    for c in ranked:
        if c.case_id not in seen:
            seen.add(c.case_id)
            out.append(c)
    return out


def check_thresholds(run: Run, gate: GateConfig) -> list[str]:
    """Absolute gate for a run with no baseline."""
    failures = []
    for name, floor in gate.min.items():
        s = run.summary.get(name)
        if s is not None and s.mean is not None and s.mean < floor:
            failures.append(f"{name} = {s.mean:.3f} is below the minimum {floor}")
    return failures


MARKER = "<!-- ragspec-report -->"
_ICON = {"regressed": "🔴", "improved": "🟢", "unchanged": "⚪", "new": "🆕", "missing": "➖"}


def _fmt(v: float | None, digits: int = 3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _signed(v: float | None) -> str:
    return "–" if v is None else f"{v:+.3f}"


def to_markdown(cmp: Comparison, report_url: str | None = None) -> str:
    verdict = (
        "✅ **No significant regressions**"
        if cmp.passed
        else f"❌ **{len(cmp.failures)} gate failure{'s' if len(cmp.failures) != 1 else ''}**"
    )
    lines = [
        MARKER,
        f"## ragspec: `{cmp.head_label}` vs `{cmp.base_label}`",
        "",
        verdict,
    ]
    if cmp.failures:
        lines += [""] + [f"- {f}" for f in cmp.failures]
    if cmp.dataset_changed:
        lines += ["", "> ⚠️ The eval dataset differs between runs. Only shared questions are "
                      "compared."]
    lines += [
        "",
        "| | Metric | Base | Head | Δ | 95% CI | ↑ / ↓ |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for md in cmp.metrics:
        ci = "–" if md.ci_low is None else f"{md.ci_low:+.3f} … {md.ci_high:+.3f}"
        name = md.metric if md.gated else f"{md.metric} _(info)_"
        floor = " ⛔" if md.below_min else ""
        lines.append(
            f"| {_ICON[md.status]} | {name}{floor} | {_fmt(md.base)} | {_fmt(md.head)} | "
            f"{_signed(md.delta)} | {ci} | {md.n_better} / {md.n_worse} |"
        )
    lat = cmp.latency
    lines.append(
        f"| ⏱ | latency p50 / p95 (ms) | {_fmt(lat['p50']['base'], 0)} / "
        f"{_fmt(lat['p95']['base'], 0)} | {_fmt(lat['p50']['head'], 0)} / "
        f"{_fmt(lat['p95']['head'], 0)} | | | |"
    )

    def case_table(title: str, rows: list[CaseChange]) -> list[str]:
        if not rows:
            return []
        out = ["", f"<details><summary>{title}</summary>", "",
               "| Question | Metric | Base | Head |", "|---|---|---:|---:|"]
        for c in rows:
            q = c.question.replace("|", "\\|").replace("\n", " ")
            q = q if len(q) <= 90 else q[:87] + "…"
            out.append(f"| `{c.case_id}` {q} | {c.metric} | {c.base:.2f} | {c.head:.2f} |")
        return out + ["", "</details>"]

    lines += case_table(f"Biggest question-level drops ({len(cmp.worst_cases)})", cmp.worst_cases)
    lines += case_table(f"Biggest improvements ({len(cmp.best_cases)})", cmp.best_cases)

    footer = [f"{cmp.n_head} questions"]
    if cmp.head_cost_usd is not None:
        footer.append(f"judge cost ${cmp.head_cost_usd:.2f}")
    if report_url:
        footer.append(f"[full report]({report_url})")
    lines += ["", "<sub>" + " · ".join(footer) + " · 🔴 significant drop · 🟢 significant gain"
              " · ↑/↓ = questions that improved/regressed</sub>"]
    return "\n".join(lines) + "\n"
