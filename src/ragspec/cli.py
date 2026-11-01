"""ragspec command line."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .adapters import load_pipeline
from .compare import Comparison, check_thresholds, compare_runs, to_markdown
from .config import GateConfig, load_config
from .judge import CachedJudge, make_judge
from .models import Run
from .runner import default_run_path, load_dataset, load_run, run_eval, save_run, select_cases

app = typer.Typer(help="pytest for RAG: measure retrieval, faithfulness and citations.",
                  no_args_is_help=True, add_completion=False)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to ragspec.yaml")]


def fmt_ms(v: float) -> str:
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def print_run(run: Run) -> None:
    table = Table(title=f"ragspec · {run.meta.label} · {run.meta.n_cases} questions",
                  title_justify="left")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("n", justify="right", style="dim")
    for name, s in run.summary.items():
        table.add_row(name, "–" if s.mean is None else f"{s.mean:.3f}", str(s.n))
    lat = run.latency_ms
    if lat.get("p50") is not None:
        table.add_row("latency p50 / p95", f"{fmt_ms(lat['p50'])} / {fmt_ms(lat['p95'])} ms", "")
    console.print(table)
    cost = run.cost
    if run.meta.judge_model:
        usd = f" · ~${cost['judge_usd']:.2f}" if "judge_usd" in cost else ""
        console.print(
            f"[dim]judge {run.meta.judge_model}: {cost.get('judge_cached', 0):.0f} cached, "
            f"{cost.get('judge_input_tokens', 0):,.0f} in / "
            f"{cost.get('judge_output_tokens', 0):,.0f} out tokens{usd}[/dim]"
        )
    if run.errors:
        console.print(f"[yellow]{run.errors} case(s) had errors. Details are in the run file "
                      f"and report.[/yellow]")
        for case in [c for c in run.cases if c.error][:3]:
            console.print(f"  [yellow]{case.case_id}: {case.error}[/yellow]")


def print_comparison(cmp: Comparison) -> None:
    style = {"regressed": "red", "improved": "green", "unchanged": "", "new": "cyan",
             "missing": "dim"}
    table = Table(title=f"{cmp.head_label} vs {cmp.base_label}", title_justify="left")
    for col, justify in [("Metric", "left"), ("Base", "right"), ("Head", "right"),
                         ("Δ", "right"), ("95% CI", "left"), ("Status", "left")]:
        table.add_column(col, justify=justify)
    for md in cmp.metrics:
        ci = "" if md.ci_low is None else f"{md.ci_low:+.3f} … {md.ci_high:+.3f}"
        table.add_row(
            md.metric + ("" if md.gated else " (info)"),
            "–" if md.base is None else f"{md.base:.3f}",
            "–" if md.head is None else f"{md.head:.3f}",
            "–" if md.delta is None else f"{md.delta:+.3f}",
            ci,
            f"[{style[md.status]}]{md.status}[/{style[md.status]}]" if style[md.status]
            else md.status,
        )
    console.print(table)
    if cmp.passed:
        console.print("[green]✓ no significant regressions[/green]")
    else:
        for f in cmp.failures:
            console.print(f"[red]✗ {f}[/red]")


@app.command()
def run(
    config: ConfigOpt = Path("ragspec.yaml"),
    label: Annotated[str | None, typer.Option(help="Name for this run (default: git sha)")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions")] = None,
    sample: Annotated[int | None, typer.Option(help="Seeded random sample of N questions")] = None,
    seed: Annotated[int, typer.Option(help="Seed for --sample")] = 0,
    tag: Annotated[list[str] | None, typer.Option(help="Only questions with this tag")] = None,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Retrieval metrics only")] = False,
    out: Annotated[Path | None, typer.Option(help="Where to write the run JSON")] = None,
    baseline: Annotated[Path | None, typer.Option(help="Compare against this run JSON")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Write a PR-comment markdown file")] = None,
    report: Annotated[Path | None, typer.Option(help="Write an HTML report")] = None,
) -> None:
    """Run the pipeline over the dataset, score it, and optionally gate against a baseline."""
    cfg = load_config(config)
    cases = select_cases(load_dataset(cfg.path(cfg.dataset)), limit, sample, seed, tag)
    if not cases:
        console.print("[red]No questions selected.[/red]")
        raise typer.Exit(2)
    pipeline = load_pipeline(cfg.pipeline, cfg.base_dir)
    judge = None
    if cfg.judge.enabled and not no_judge:
        try:
            judge = CachedJudge(make_judge(cfg.judge), cfg.path(cfg.judge.cache_dir))
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2) from None

    with Progress(TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
                  TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task(f"Evaluating {cfg.pipeline.describe()}", total=len(cases))
        result = asyncio.run(
            run_eval(cfg, cases, pipeline, judge, label,
                     on_progress=lambda _: progress.advance(task))
        )

    path = save_run(result, (out or default_run_path(cfg, result)).resolve())
    print_run(result)
    console.print(f"[dim]saved {path}[/dim]")

    failures = check_thresholds(result, cfg.gate)
    cmp = None
    if baseline:
        cmp = compare_runs(load_run(baseline), result, cfg.gate)
        print_comparison(cmp)
        failures = cmp.failures
        if markdown:
            markdown.write_text(to_markdown(cmp))
    elif failures:
        for f in failures:
            console.print(f"[red]✗ {f}[/red]")
    if report:
        from .report import write_report

        runs = [load_run(baseline), result] if baseline else [result]
        write_report(runs, report)
        console.print(f"[dim]report {report}[/dim]")
    if failures:
        raise typer.Exit(1)


@app.command()
def compare(
    base: Annotated[Path, typer.Argument(help="Baseline run JSON")],
    head: Annotated[Path, typer.Argument(help="New run JSON")],
    config: Annotated[Path | None, typer.Option("--config", "-c",
                                                help="ragspec.yaml with gate settings")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Write a PR-comment markdown file")] = None,
    report_url: Annotated[str | None, typer.Option(help="Link to the HTML report")] = None,
) -> None:
    """Compare two runs. Exits 1 if a gated metric regressed significantly."""
    gate = load_config(config).gate if config else GateConfig()
    cmp = compare_runs(load_run(base), load_run(head), gate)
    print_comparison(cmp)
    if markdown:
        markdown.write_text(to_markdown(cmp, report_url))
        console.print(f"[dim]wrote {markdown}[/dim]")
    if not cmp.passed:
        raise typer.Exit(1)


@app.command()
def report(
    runs: Annotated[list[Path], typer.Argument(help="Run JSON files or directories of them")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("ragspec-report.html"),
    title: Annotated[str, typer.Option(help="Report title")] = "ragspec report",
) -> None:
    """Build a self-contained HTML dashboard from one or more runs (oldest first)."""
    from .report import write_report

    files: list[Path] = []
    for p in runs:
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    loaded = sorted((load_run(f) for f in files), key=lambda r: r.meta.created_at)
    if not loaded:
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(2)
    write_report(loaded, out, title)
    console.print(f"wrote {out} ({len(loaded)} runs)")


@app.command()
def ui(
    path: Annotated[Path, typer.Argument(help="Folder with configs and runs")] = Path("."),
    port: Annotated[int, typer.Option(help="Local port")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open a tab")] = True,
) -> None:
    """Open a local playground: ask a question, compare two configs, browse past runs."""
    from .ui import make_server

    if not path.is_dir():
        console.print(f"[red]{path} is not a folder.[/red]")
        raise typer.Exit(2)
    try:
        server = make_server(path, port)
    except OSError as e:
        console.print(f"[red]Can't use port {port}: {e}. Try --port.[/red]")
        raise typer.Exit(2) from None
    url = f"http://localhost:{port}"
    console.print(f"ragspec playground at [bold]{url}[/bold]. Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


SAMPLE_CONFIG = """\
# ragspec configuration. Paths are relative to this file.
dataset: evals/questions.jsonl

pipeline:
  type: python                 # or: http
  # module:function(question) returning {"answer": str, "contexts": [{"id", "text"}]}
  target: my_rag:answer
  # type: http
  # url: http://localhost:8000/ask
  # body: {query: "{question}"}
  # answer_path: answer
  # contexts_path: sources
  # headers: {Authorization: "Bearer ${MY_API_TOKEN}"}

k: [1, 5, 10]
concurrency: 8

judge:
  model: claude-opus-5-5
  effort: medium
  concurrency: 4
  # Free hosted model instead (Gemini here; export LLM_API_KEY=... first):
  # provider: openai
  # base_url: https://generativelanguage.googleapis.com/v1beta/openai/
  # model: gemini-3.5-flash-lite
  # concurrency: 1

gate:
  min:
    faithfulness: 0.8
  default_max_drop: 0.01       # tolerate significant drops smaller than this
"""

SAMPLE_DATASET = (
    '{"id": "q1", "question": "What is the refund window?", "relevant": ["policy-refunds"]}\n'
)


@app.command()
def init(directory: Annotated[Path, typer.Argument()] = Path(".")) -> None:
    """Write a starter ragspec.yaml and dataset."""
    cfg_path = directory / "ragspec.yaml"
    data_path = directory / "evals" / "questions.jsonl"
    for p, content in [(cfg_path, SAMPLE_CONFIG), (data_path, SAMPLE_DATASET)]:
        if p.exists():
            console.print(f"[yellow]exists, skipped {p}[/yellow]")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        console.print(f"wrote {p}")


if __name__ == "__main__":
    app()
