"""Turn a pipeline config into an async callable: Case -> PipelineOutput."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from .config import PipelineConfig
from .models import Case, Context, PipelineOutput

Pipeline = Callable[[Case], Awaitable[PipelineOutput]]


def coerce_output(raw: Any) -> PipelineOutput:
    """Accept a PipelineOutput, a dict, or any object with `answer` / `contexts` attributes.

    Contexts may be dicts ({"id", "text"}), Context objects, or bare strings (ids become the
    1-based position).
    """
    if isinstance(raw, PipelineOutput):
        return raw
    if not isinstance(raw, dict):
        raw = {
            "answer": getattr(raw, "answer", None),
            "contexts": getattr(raw, "contexts", []),
            "meta": getattr(raw, "meta", {}),
        }
    contexts = []
    for i, ctx in enumerate(raw.get("contexts") or []):
        if isinstance(ctx, str):
            contexts.append(Context(id=str(i + 1), text=ctx))
        elif isinstance(ctx, Context):
            contexts.append(ctx)
        else:
            contexts.append(Context.model_validate(ctx))
    return PipelineOutput(answer=raw.get("answer"), contexts=contexts, meta=raw.get("meta") or {})


def _import_target(target: str, search_dirs: list[Path]) -> Any:
    module_name, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"pipeline.target must look like 'module:callable', got {target!r}")
    for d in reversed(search_dirs):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def python_pipeline(cfg: PipelineConfig, base_dir: Path) -> Pipeline:
    dirs = [(base_dir / p).resolve() for p in cfg.python_path] + [base_dir]
    fn = _import_target(cfg.target or "", dirs)
    if cfg.options:
        fn = fn(**cfg.options)
    params = inspect.signature(fn).parameters
    wants_case = "case" in params

    async def call(case: Case) -> PipelineOutput:
        kwargs = {"case": case} if wants_case else {}
        if inspect.iscoroutinefunction(fn):
            raw = await fn(case.question, **kwargs)
        else:
            raw = await asyncio.to_thread(fn, case.question, **kwargs)
        return coerce_output(raw)

    return call


def _fill(template: Any, case: Case) -> Any:
    if isinstance(template, str):
        return template.replace("{question}", case.question).replace("{id}", case.id)
    if isinstance(template, dict):
        return {k: _fill(v, case) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill(v, case) for v in template]
    return template


def _get_path(data: Any, path: str) -> Any:
    for part in path.split(".") if path else []:
        if isinstance(data, list):
            data = data[int(part)]
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            return None
    return data


def http_pipeline(cfg: PipelineConfig) -> Pipeline:
    if not cfg.url:
        raise ValueError("pipeline.url is required for http pipelines")
    client = httpx.AsyncClient(timeout=cfg.timeout_s, headers=cfg.headers)

    async def call(case: Case) -> PipelineOutput:
        body = _fill(cfg.body, case)
        if cfg.method == "GET":
            resp = await client.get(cfg.url, params=body)
        else:
            resp = await client.post(cfg.url, json=body)
        resp.raise_for_status()
        data = resp.json()
        contexts = [
            {
                "id": _get_path(c, cfg.context_id_field),
                "text": _get_path(c, cfg.context_text_field) or "",
            }
            for c in (_get_path(data, cfg.contexts_path) or [])
        ]
        return coerce_output({"answer": _get_path(data, cfg.answer_path), "contexts": contexts})

    return call


def load_pipeline(cfg: PipelineConfig, base_dir: Path) -> Pipeline:
    if cfg.type == "http":
        return http_pipeline(cfg)
    return python_pipeline(cfg, base_dir)
