"""Data shapes shared by the runner, the judge, comparisons and reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Case(BaseModel):
    """One evaluation question. `relevant` maps doc id -> graded relevance (>0 is relevant)."""

    id: str
    question: str
    relevant: dict[str, float] = Field(default_factory=dict)
    reference_answer: str | None = None
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> str:
        return str(v)

    @field_validator("relevant", mode="before")
    @classmethod
    def _list_to_grades(cls, v: Any) -> Any:
        # Accept ["doc1", "doc2"] as shorthand for {"doc1": 1, "doc2": 1}.
        if isinstance(v, list):
            return {str(doc_id): 1.0 for doc_id in v}
        return v


class Context(BaseModel):
    id: str
    text: str = ""
    score: float | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> str:
        return str(v)


class PipelineOutput(BaseModel):
    """What a pipeline under test returns. `answer` is optional for retrieval-only pipelines."""

    answer: str | None = None
    contexts: list[Context] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


Verdict = Literal["supported", "unsupported", "contradicted"]


class Claim(BaseModel):
    text: str
    verdict: Verdict
    cited_ids: list[str] = Field(default_factory=list)
    supporting_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class JudgeResult(BaseModel):
    model: str
    claims: list[Claim] = Field(default_factory=list)
    abstained: bool = False
    cited_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class CaseResult(BaseModel):
    case_id: str
    question: str
    tags: list[str] = Field(default_factory=list)
    relevant: dict[str, float] = Field(default_factory=dict)
    contexts: list[Context] = Field(default_factory=list)
    answer: str | None = None
    latency_ms: float | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    judge: JudgeResult | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    mean: float | None
    n: int


class RunMeta(BaseModel):
    label: str
    created_at: str
    ragspec_version: str
    pipeline: str
    dataset: str
    dataset_sha: str
    n_cases: int
    k: list[int]
    judge_model: str | None = None
    git_sha: str | None = None
    git_branch: str | None = None


class Run(BaseModel):
    meta: RunMeta
    summary: dict[str, MetricSummary]
    latency_ms: dict[str, float | None] = Field(default_factory=dict)
    cost: dict[str, float] = Field(default_factory=dict)
    errors: int = 0
    cases: list[CaseResult]
