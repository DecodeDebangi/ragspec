"""ragspec.yaml schema and loading."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

from .citations import DEFAULT_PATTERN

DEFAULT_JUDGE_MODEL = "claude-opus-5-5"


class PipelineConfig(BaseModel):
    type: Literal["python", "http"] = "python"
    # python: "module.path:callable". If `options` is set, the callable is a factory that
    # is called once with **options and must return the pipeline callable.
    target: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    # Extra import directories, relative to the config file (its own directory is always added).
    python_path: list[str] = Field(default_factory=list)
    # http: POST/GET a JSON endpoint. String values in `body` may contain {question} and {id}.
    url: str | None = None
    method: Literal["GET", "POST"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=lambda: {"question": "{question}"})
    answer_path: str = "answer"
    contexts_path: str = "contexts"
    context_id_field: str = "id"
    context_text_field: str = "text"
    timeout_s: float = 120.0

    def describe(self) -> str:
        if self.type == "http":
            return f"http {self.method} {self.url}"
        opts = ", ".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        return f"{self.target}({opts})" if opts else str(self.target)


class JudgeConfig(BaseModel):
    enabled: bool = True
    # anthropic: the Claude API (needs ANTHROPIC_API_KEY). ollama: a local model, free.
    # openai: any hosted OpenAI-compatible API, e.g. Gemini or Groq (both have free tiers).
    provider: Literal["anthropic", "ollama", "openai"] = "anthropic"
    model: str = DEFAULT_JUDGE_MODEL
    # Claude only; local models ignore it.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    ollama_url: str = "http://localhost:11434"
    # openai provider only. The key is read from this environment variable, never from the file.
    base_url: str | None = None
    api_key_env: str = "LLM_API_KEY"
    concurrency: int = 4
    cache_dir: str = ".ragspec/cache"
    max_context_chars: int = 6000


class CitationConfig(BaseModel):
    pattern: str = DEFAULT_PATTERN
    style: Literal["auto", "id", "position"] = "auto"


class GateConfig(BaseModel):
    # Metrics that can fail the gate. Empty = every metric in the run.
    metrics: list[str] = Field(default_factory=list)
    # Absolute floors, e.g. {"faithfulness": 0.85}.
    min: dict[str, float] = Field(default_factory=dict)
    # A significant drop smaller than this is tolerated. Per metric, falling back to default.
    max_drop: dict[str, float] = Field(default_factory=dict)
    default_max_drop: float = 0.01
    confidence: float = 0.95


class Config(BaseModel):
    dataset: str
    # Shown in the playground: a friendly name (default: the file name), what this version does
    # differently, and what system is being tested.
    label: str | None = None
    description: str | None = None
    subject: str | None = None
    pipeline: PipelineConfig
    k: list[int] = Field(default_factory=lambda: [1, 5, 10])
    concurrency: int = 8
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    citations: CitationConfig = Field(default_factory=CitationConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    runs_dir: str = ".ragspec/runs"
    # Context text kept per retrieved chunk in run files and reports (0 = ids and scores only).
    store_context_chars: int = 1500
    # Directory the config was loaded from; relative paths resolve against it.
    base_dir: Path = Field(default=Path("."), exclude=True)

    def path(self, p: str) -> Path:
        candidate = Path(p)
        return candidate if candidate.is_absolute() else self.base_dir / candidate


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_env() -> None:
    """Load the nearest .env, searching up from the working directory. Real env vars win."""
    load_dotenv(find_dotenv(usecwd=True))


def load_config(path: str | Path) -> Config:
    load_env()
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = Config.model_validate(_expand_env(raw))
    cfg.base_dir = path.parent.resolve()
    return cfg
