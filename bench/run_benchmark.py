"""Run the retrieval benchmark with ragspec and render the results for the README.

    uv run python bench/run_benchmark.py                       # retrieval only, free
    uv run python bench/run_benchmark.py --generate --sample 50   # + Claude answers and judge

Writes run files to bench/runs/, a markdown table to bench/RESULTS.md and charts to
docs/*.svg.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from ragspec.adapters import python_pipeline  # noqa: E402
from ragspec.config import Config, JudgeConfig, PipelineConfig  # noqa: E402
from ragspec.judge import CachedJudge, ClaudeJudge  # noqa: E402
from ragspec.models import Run  # noqa: E402
from ragspec.runner import load_dataset, load_run, run_eval, save_run, select_cases  # noqa: E402

DATASETS = ["scifact", "nfcorpus", "fiqa"]
VARIANTS = {
    # label: pipeline options
    "BM25": {"retriever": "bm25"},
    "Dense": {"retriever": "dense"},
    "Hybrid (RRF)": {"retriever": "hybrid"},
    "Hybrid + MiniLM rerank": {"retriever": "rerank", "reranker": "minilm"},
    "Hybrid + bge rerank": {"retriever": "rerank", "reranker": "bge"},
}
SLUG = {"BM25": "bm25", "Dense": "dense", "Hybrid (RRF)": "hybrid",
        "Hybrid + MiniLM rerank": "rerank-minilm", "Hybrid + bge rerank": "rerank-bge"}
DATASET_LABEL = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA-2018"}


def run_one(dataset: str, variant: str, generate: bool, sample: int | None) -> Run:
    mode = "gen" if generate else "ret"
    path = HERE / "runs" / f"{dataset}-{SLUG[variant]}-{mode}.json"
    if path.exists():
        return load_run(path)
    options = {"dataset": dataset, "k": 5 if generate else 10, "generate": generate,
               **VARIANTS[variant]}
    cfg = Config(
        dataset=f"evals/{dataset}.jsonl",
        pipeline=PipelineConfig(target="pipelines:make", options=options),
        k=[1, 5] if generate else [1, 5, 10],
        # Retrieval-only runs are sequential so latency is clean per-query latency.
        concurrency=8 if generate else 1,
        judge=JudgeConfig(enabled=generate, concurrency=8),
        # Retrieval runs only need ids and scores; keeping text makes the files ~70 MB.
        store_context_chars=1500 if generate else 0,
        base_dir=HERE,
    )
    cases = select_cases(load_dataset(cfg.path(cfg.dataset)), sample=sample, seed=0)
    judge = None
    if generate:
        judge = CachedJudge(ClaudeJudge(cfg.judge.model, cfg.judge.effort),
                            ROOT / ".ragspec" / "cache")
    print(f"→ {dataset} / {variant} ({len(cases)} queries)", flush=True)
    run = asyncio.run(run_eval(cfg, cases, python_pipeline(cfg.pipeline, HERE), judge,
                               label=f"{dataset}-{SLUG[variant]}"))
    save_run(run, path)
    return run


def metric(run: Run, name: str) -> float | None:
    s = run.summary.get(name)
    return s.mean if s else None


def fmt(v: float | None, digits: int = 3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def retrieval_table(results: dict[str, dict[str, Run]]) -> str:
    lines = ["| Retriever | " + " | ".join(
        f"{DATASET_LABEL[d]} nDCG@10" for d in results) + " | " + " | ".join(
        f"{DATASET_LABEL[d]} R@10" for d in results) + " | p50 / p95 latency¹ |",
        "|---|" + "---:|" * (2 * len(results) + 1)]
    for variant in VARIANTS:
        ndcg, rec, lat = [], [], []
        for runs in results.values():
            best = max(metric(r, "ndcg@10") or 0 for r in runs.values())
            v = metric(runs[variant], "ndcg@10")
            cell = fmt(v)
            ndcg.append(f"**{cell}**" if v is not None and abs(v - best) < 1e-9 else cell)
            rec.append(fmt(metric(runs[variant], "recall@10")))
            lat.append(runs[variant].latency_ms)
        p50 = sorted(lt["p50"] for lt in lat)[len(lat) // 2]
        p95 = sorted(lt["p95"] for lt in lat)[len(lat) // 2]
        lines.append(f"| {variant} | " + " | ".join(ndcg) + " | " + " | ".join(rec) +
                     f" | {fmt_ms(p50)} / {fmt_ms(p95)} ms |")
    return "\n".join(lines)


def generation_table(results: dict[str, dict[str, Run]]) -> str:
    lines = ["| Dataset | Retriever | R@5 | Faithfulness | Citation precision | "
             "Citation recall | Abstained | n |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for d, runs in results.items():
        for variant, r in runs.items():
            lines.append(
                f"| {DATASET_LABEL[d]} | {variant} | {fmt(metric(r, 'recall@5'))} | "
                f"{fmt(metric(r, 'faithfulness'))} | {fmt(metric(r, 'citation_precision'))} | "
                f"{fmt(metric(r, 'citation_recall'))} | "
                f"{fmt(metric(r, 'abstention_rate'), 2)} | {r.meta.n_cases} |"
            )
    return "\n".join(lines)


# Categorical slots 1-5 (validated: adjacent CVD ΔE ≥ 9.1 light / dark ≥ 8).
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]


def grouped_bar_svg(results: dict[str, dict[str, Run]], metric_name: str, title: str) -> str:
    """Grouped columns: one group per dataset, one column per retriever."""
    variants = list(VARIANTS)
    datasets = list(results)
    W, H, L, R, T, B = 760, 340, 44, 16, 64, 40
    plot_w, plot_h = W - L - R, H - T - B
    group_w = plot_w / len(datasets)
    bar_w, gap = 22, 4
    cluster = len(variants) * bar_w + (len(variants) - 1) * gap
    values = [metric(results[d][v], metric_name) or 0 for d in datasets for v in variants]
    top = max(0.1, round(max(values) + 0.1, 1))

    def y(v: float) -> float:
        return T + plot_h * (1 - v / top)

    css_vars = "\n".join(
        f"  .s{i} {{ fill: {c}; }}" for i, c in enumerate(LIGHT))
    css_dark = "\n".join(
        f"    .s{i} {{ fill: {c}; }}" for i, c in enumerate(DARK))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="ui-sans-serif, system-ui, -apple-system, sans-serif">',
        "<style>",
        "  .bg { fill: #fcfcfb; } .grid { stroke: #ebeae6; } .axis { stroke: #d6d5d0; }",
        "  .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .t3 { fill: #7a7974; }",
        css_vars,
        "  @media (prefers-color-scheme: dark) {",
        "    .bg { fill: #1a1a19; } .grid { stroke: #2a2a28; } .axis { stroke: #3a3a37; }",
        "    .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; } .t3 { fill: #8f8e86; }",
        css_dark,
        "  }",
        "</style>",
        f'<rect class="bg" width="{W}" height="{H}" rx="10"/>',
        f'<text class="t1" x="{L}" y="24" font-size="15" font-weight="600">{title}</text>',
    ]
    # Legend, one row above the plot.
    lx = L
    for i, v in enumerate(variants):
        out.append(f'<rect class="s{i}" x="{lx}" y="38" width="10" height="10" rx="2"/>')
        out.append(f'<text class="t2" x="{lx + 15}" y="47" font-size="12">{v}</text>')
        lx += 15 + len(v) * 6.6 + 18
    for t in range(0, int(round(top * 10)) + 1, 2):
        v = t / 10
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="t3" x="{L - 8}" y="{y(v) + 4:.1f}" font-size="11" '
                   f'text-anchor="end">{v:.1f}</text>')
    for g, d in enumerate(datasets):
        x0 = L + g * group_w + (group_w - cluster) / 2
        vals = [metric(results[d][v], metric_name) or 0 for v in variants]
        best = max(vals)
        for i, v in enumerate(vals):
            x = x0 + i * (bar_w + gap)
            yt, r = y(v), 4
            path = (f"M{x:.1f},{y(0):.1f} V{yt + r:.1f} Q{x:.1f},{yt:.1f} {x + r:.1f},{yt:.1f} "
                    f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{yt:.1f} {x + bar_w:.1f},{yt + r:.1f} "
                    f"V{y(0):.1f} Z")
            out.append(f'<path class="s{i}" d="{path}"><title>{variants[i]} on '
                       f'{DATASET_LABEL[d]}: {v:.3f}</title></path>')
            if v == best:
                out.append(f'<text class="t1" x="{x + bar_w / 2:.1f}" y="{yt - 6:.1f}" '
                           f'font-size="11" font-weight="600" text-anchor="middle">{v:.3f}</text>')
        out.append(f'<text class="t2" x="{L + g * group_w + group_w / 2:.1f}" y="{H - B + 20}" '
                   f'font-size="12" text-anchor="middle">{DATASET_LABEL[d]}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--generate", action="store_true", help="also answer with Claude + judge")
    parser.add_argument("--sample", type=int, default=50, help="questions per generation run")
    args = parser.parse_args()

    retrieval = {d: {v: run_one(d, v, False, None) for v in VARIANTS} for d in args.datasets}
    sections = ["# Benchmark results", "",
                "Generated by `bench/run_benchmark.py`. Each cell is a full `ragspec run`.", "",
                "## Retrieval (all BEIR test queries)", "", retrieval_table(retrieval), "",
                "¹ Median across datasets of per-query p50 / p95, sequential, MacBook Air M2 "
                "(8 GB), models on MPS, Postgres 17 + pgvector HNSW in Docker."]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "retrieval-ndcg.svg").write_text(
        grouped_bar_svg(retrieval, "ndcg@10", "nDCG@10 on BEIR (higher is better)"))

    if args.generate:
        gen_variants = [v for v in args.variants if v in VARIANTS]
        generation = {d: {v: run_one(d, v, True, args.sample) for v in gen_variants}
                      for d in args.datasets}
        sections += ["", f"## End-to-end RAG ({args.sample} sampled questions per dataset, "
                         "top-5 contexts, Claude Opus 5.5 answers, Claude Opus 5.5 judge)", "",
                     generation_table(generation)]
        cost = sum(r.cost.get("judge_usd", 0) for runs in generation.values()
                   for r in runs.values())
        sections += ["", f"Judge cost for this table: ${cost:.2f} (cached re-runs are free)."]

    (HERE / "RESULTS.md").write_text("\n".join(sections) + "\n")
    print((HERE / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
