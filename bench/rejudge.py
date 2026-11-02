"""Re-grade the answers saved in a run with the current judge, without regenerating them.

    uv run python bench/rejudge.py run.json out.json --provider openai --base-url <url> \
        --model <name>

Use it to measure an evaluator change on its own: the answers stay byte-identical, so any score
change comes from the judge. Run files keep at most `store_context_chars` of each context, so
the judge may see less text than it did originally.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from ragspec.citations import DEFAULT_PATTERN
from ragspec.config import JudgeConfig, load_env
from ragspec.judge import CachedJudge, canonicalize_citations, make_judge
from ragspec.metrics import GENERATION_METRICS, generation_metrics
from ragspec.runner import load_run, save_run, summarize


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "ollama", "openai"])
    parser.add_argument("--base-url")
    parser.add_argument("--product")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--cache-dir", type=Path, help="Default: a fresh cache each time")
    args = parser.parse_args()
    load_env()
    judge = CachedJudge(
        make_judge(JudgeConfig(provider=args.provider, model=args.model, base_url=args.base_url,
                               product=args.product)),
        args.cache_dir or Path(tempfile.mkdtemp()),
    )

    run = load_run(args.run)
    for case in run.cases:
        for name in GENERATION_METRICS:
            case.metrics.pop(name, None)
        case.judge = None
        if case.error and case.error.startswith("pipeline error"):
            continue
        case.error = None
        if not case.answer:
            continue
        answer, cited = canonicalize_citations(case.answer, case.contexts, DEFAULT_PATTERN, "auto")
        case.judge = await judge.judge(case.question, case.contexts, answer, cited)
        case.error = case.judge.error
        case.metrics.update(generation_metrics(case.judge))
    run.summary = summarize(run.cases, run.meta.k)
    run.errors = sum(1 for c in run.cases if c.error)
    run.meta.judge_model = args.model
    save_run(run, args.out)
    print(f"{args.out}: {run.errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
