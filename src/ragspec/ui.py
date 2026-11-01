"""Local playground: ask one question of one or two configs and see every scoring step.

Questions go through the same adapter, judge and metrics as `ragspec run`, so a score here means
what it means in a run. The server listens on localhost only. It checks the Host header and
requires a JSON body, so a website open in another tab can't spend the user's API quota.
"""

from __future__ import annotations

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .adapters import load_pipeline
from .compare import compare_runs
from .config import Config, GateConfig, load_config
from .judge import CachedJudge, make_judge
from .models import Case, Run
from .report import render_report
from .runner import load_dataset, load_run, run_single


def _summary_line(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""


def discover_configs(root: Path) -> dict[str, Config]:
    configs = {}
    # Free hosted configs first (they need only an API key), and a baseline before its PR.
    paths = sorted(root.glob("*.y*ml"),
                   key=lambda p: ("hosted" not in p.name, ".pr." in p.name, p.name))
    for path in paths:
        try:
            configs[path.name] = load_config(path)
        except (ValueError, TypeError, OSError):
            continue  # some other YAML file, not a ragspec config
    return configs


def discover_runs(root: Path) -> dict[str, Run]:
    runs = {}
    for path in [*root.glob("*.json"), *root.glob(".ragspec/runs/*.json")]:
        try:
            runs[path.relative_to(root).as_posix()] = load_run(path)
        except (ValueError, OSError):
            continue  # some other JSON file, not a run
    return dict(sorted(runs.items(), key=lambda kv: kv[1].meta.created_at, reverse=True))


def _missing_key(cfg: Config) -> str | None:
    """The API key a config's judge needs and doesn't have, if any."""
    if not cfg.judge.enabled or cfg.judge.provider == "ollama":
        return None
    name = "ANTHROPIC_API_KEY" if cfg.judge.provider == "anthropic" else cfg.judge.api_key_env
    return None if os.environ.get(name) else name


def setup(root: Path) -> dict:
    configs = discover_configs(root)
    questions: list[dict] = []
    if configs:
        first = next(iter(configs.values()))
        try:
            cases = load_dataset(first.path(first.dataset))
        except (ValueError, OSError):
            cases = []
        questions = [{"id": c.id, "question": c.question, "tags": c.tags,
                      "answerable": any(g > 0 for g in c.relevant.values())} for c in cases]
    return {
        "subject": next((c.subject for c in configs.values() if c.subject), None),
        "configs": [
            {"name": name, "label": cfg.label or name, "missing_key": _missing_key(cfg),
             "summary": _summary_line(root / name), "description": cfg.description,
             "pipeline": cfg.pipeline.describe(),
             "judge": cfg.judge.model if cfg.judge.enabled else None}
            for name, cfg in configs.items()
        ],
        "questions": questions,
        "runs": [
            {"name": name, "label": run.meta.label, "created_at": run.meta.created_at,
             "n_cases": run.meta.n_cases, "pipeline": run.meta.pipeline,
             "judge": run.meta.judge_model}
            for name, run in discover_runs(root).items()
        ],
    }


def saved(root: Path, base_name: str, head_name: str) -> dict:
    """Two saved runs and their comparison, so results can be browsed without any model calls."""
    runs = discover_runs(root)
    if base_name not in runs or head_name not in runs:
        raise ValueError("unknown run")
    configs = discover_configs(root)
    gate = next((c.gate for c in configs.values()), GateConfig())
    by_pipeline = {c.pipeline.describe(): c for c in configs.values()}

    def side(name: str) -> dict:
        run = runs[name]
        cfg = by_pipeline.get(run.meta.pipeline)
        return {"name": name, "label": (cfg.label if cfg else None) or run.meta.label,
                "description": cfg.description if cfg else None,
                "run": run.model_dump(mode="json")}

    base, head = runs[base_name], runs[head_name]
    return {"base": side(base_name), "head": side(head_name),
            "comparison": compare_runs(base, head, gate).model_dump(mode="json")}


def _case_for(cfg: Config, question: str, case_id: str | None) -> Case:
    if case_id:
        for case in load_dataset(cfg.path(cfg.dataset)):
            if case.id == case_id:
                return case
    # A typed question has no relevance labels, so retrieval metrics stay empty.
    return Case(id="typed-question", question=question)


async def _ask_one(cfg: Config, question: str, case_id: str | None) -> dict:
    try:
        case = _case_for(cfg, question, case_id)
        # Built per request: their HTTP clients belong to this request's event loop.
        pipeline = load_pipeline(cfg.pipeline, cfg.base_dir)
        judge = None
        if cfg.judge.enabled:
            judge = CachedJudge(make_judge(cfg.judge), cfg.path(cfg.judge.cache_dir))
        result = await run_single(cfg, case, pipeline, judge)
    except Exception as e:  # shown in that config's column; the other one still runs
        return {"error": f"{type(e).__name__}: {e}"}
    return {"result": result.model_dump(mode="json")}


def ask(root: Path, question: str, case_id: str | None, names: list[str]) -> dict:
    configs = discover_configs(root)
    unknown = [n for n in names if n not in configs]
    if unknown:
        raise ValueError(f"unknown config: {unknown[0]}")

    async def run_all() -> list[dict]:
        return await asyncio.gather(*(_ask_one(configs[n], question, case_id) for n in names))

    return dict(zip(names, asyncio.run(run_all()), strict=True))


def make_server(root: Path, port: int = 8765) -> ThreadingHTTPServer:
    root = root.resolve()
    page = files("ragspec").joinpath("templates/ui.html").read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "").rsplit(":", 1)[0]
            if host in ("localhost", "127.0.0.1"):
                return True
            self._json(403, {"error": "forbidden host"})
            return False

        def do_GET(self) -> None:
            if not self._host_ok():
                return
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/setup":
                self._json(200, setup(root))
            elif url.path == "/api/saved":
                query = parse_qs(url.query)
                try:
                    self._json(200, saved(root, query.get("base", [""])[0],
                                          query.get("head", [""])[0]))
                except ValueError as e:
                    self._json(404, {"error": str(e)})
            elif url.path == "/report":
                names = parse_qs(url.query).get("run", [])
                runs = discover_runs(root)
                if not names or any(n not in runs for n in names):
                    self._json(404, {"error": "unknown run"})
                    return
                # The dashboard expects oldest first; the last run is the "current" one.
                chosen = sorted((runs[n] for n in names), key=lambda r: r.meta.created_at)
                self._send(200, render_report(chosen).encode(), "text/html; charset=utf-8")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._host_ok():
                return
            if urlparse(self.path).path != "/api/ask":
                self._json(404, {"error": "not found"})
                return
            # Browsers can't send this content type cross-site without a preflight we never allow.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self._json(415, {"error": "expected JSON"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                question = str(body.get("question") or "").strip()
                names = [str(n) for n in body.get("configs") or []]
                if not question or not names:
                    raise ValueError("a question and at least one config are required")
                self._json(200, ask(root, question, body.get("case_id"), names))
            except ValueError as e:
                self._json(400, {"error": str(e)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
