"""Run a pipeline over a dataset, score every case, and aggregate into a Run."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .adapters import Pipeline
from .config import Config
from .judge import Judge, canonicalize_citations, estimate_cost
from .metrics import generation_metrics, ordered_metric_names, percentile, retrieval_metrics
from .models import Case, CaseResult, MetricSummary, Run, RunMeta


def load_dataset(path: Path) -> list[Case]:
    cases = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    cases.append(Case.model_validate_json(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{line_no}: {e}") from e
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: case ids must be unique")
    return cases


def select_cases(
    cases: list[Case], limit: int | None = None, sample: int | None = None, seed: int = 0,
    tags: list[str] | None = None,
) -> list[Case]:
    if tags:
        cases = [c for c in cases if set(tags) & set(c.tags)]
    if sample is not None and sample < len(cases):
        # Seeded so a sampled baseline and a sampled PR run see the same questions.
        cases = sorted(random.Random(seed).sample(cases, sample), key=lambda c: c.id)
    if limit is not None:
        cases = cases[:limit]
    return cases


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def dataset_sha(cases: list[Case]) -> str:
    blob = json.dumps([c.model_dump() for c in cases], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


async def _run_case(
    case: Case,
    pipeline: Pipeline,
    judge: Judge | None,
    cfg: Config,
    pipeline_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
) -> CaseResult:
    result = CaseResult(
        case_id=case.id, question=case.question, tags=case.tags, relevant=case.relevant
    )
    async with pipeline_sem:
        start = time.perf_counter()
        try:
            output = await pipeline(case)
        except Exception as e:  # the pipeline under test can fail in any way
            result.error = f"pipeline error: {type(e).__name__}: {e}"
            return result
        result.latency_ms = (time.perf_counter() - start) * 1000

    result.answer = output.answer
    result.meta = output.meta
    result.metrics.update(
        retrieval_metrics([c.id for c in output.contexts], case.relevant, cfg.k)
    )

    if judge is not None and output.answer:
        answer, cited = canonicalize_citations(
            output.answer, output.contexts, cfg.citations.pattern, cfg.citations.style
        )
        async with judge_sem:
            result.judge = await judge.judge(case.question, output.contexts, answer, cited)
        if result.judge.error:
            result.error = result.judge.error
        result.metrics.update(generation_metrics(result.judge))

    result.contexts = [
        c.model_copy(update={"text": c.text[:cfg.store_context_chars]}) for c in output.contexts
    ]
    return result


async def run_single(
    cfg: Config, case: Case, pipeline: Pipeline, judge: Judge | None
) -> CaseResult:
    """Score one question exactly as a full run would."""
    return await _run_case(case, pipeline, judge, cfg, asyncio.Semaphore(1), asyncio.Semaphore(1))


def summarize(cases: list[CaseResult], ks: list[int]) -> dict[str, MetricSummary]:
    summary = {}
    for name in ordered_metric_names(ks):
        values = [c.metrics[name] for c in cases if c.metrics.get(name) is not None]
        if values:
            summary[name] = MetricSummary(mean=sum(values) / len(values), n=len(values))
    judged = [c for c in cases if c.judge and not c.judge.error]
    if judged:
        abstained = sum(1 for c in judged if c.judge.abstained)
        summary["abstention_rate"] = MetricSummary(mean=abstained / len(judged), n=len(judged))
    return summary


async def run_eval(
    cfg: Config,
    cases: list[Case],
    pipeline: Pipeline,
    judge: Judge | None,
    label: str | None = None,
    on_progress: Callable[[CaseResult], None] | None = None,
) -> Run:
    pipeline_sem = asyncio.Semaphore(cfg.concurrency)
    judge_sem = asyncio.Semaphore(cfg.judge.concurrency)

    async def one(case: Case) -> CaseResult:
        res = await _run_case(case, pipeline, judge, cfg, pipeline_sem, judge_sem)
        if on_progress:
            on_progress(res)
        return res

    results = await asyncio.gather(*(one(c) for c in cases))

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    judge_in = sum(r.judge.input_tokens for r in results if r.judge)
    judge_out = sum(r.judge.output_tokens for r in results if r.judge)
    cost = {"judge_input_tokens": judge_in, "judge_output_tokens": judge_out,
            "judge_cached": sum(1 for r in results if r.judge and r.judge.cached)}
    if judge is not None:
        usd = estimate_cost(judge.model, judge_in, judge_out)
        if usd is not None:
            cost["judge_usd"] = round(usd, 4)

    sha = _git("rev-parse", "--short", "HEAD")
    now = datetime.now(UTC)
    meta = RunMeta(
        label=label or sha or now.strftime("%Y-%m-%d %H:%M"),
        created_at=now.isoformat(timespec="seconds"),
        ragspec_version=__version__,
        pipeline=cfg.pipeline.describe(),
        dataset=cfg.dataset,
        dataset_sha=dataset_sha(cases),
        n_cases=len(cases),
        k=cfg.k,
        judge_model=judge.model if judge else None,
        git_sha=sha,
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
    )
    return Run(
        meta=meta,
        summary=summarize(results, cfg.k),
        latency_ms={
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": sum(latencies) / len(latencies) if latencies else None,
        },
        cost=cost,
        errors=sum(1 for r in results if r.error),
        cases=results,
    )


def save_run(run: Run, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json())
    return path


def load_run(path: Path) -> Run:
    return Run.model_validate_json(Path(path).read_text())


def default_run_path(cfg: Config, run: Run) -> Path:
    stamp = run.meta.created_at.replace(":", "").replace("-", "")[:15]
    slug = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in run.meta.label)[:40]
    return cfg.path(cfg.runs_dir) / f"{stamp}-{slug}.json"
