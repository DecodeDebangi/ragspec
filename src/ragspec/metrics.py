"""Retrieval metrics (from relevance labels) and generation metrics (from judge verdicts).

Retrieval metrics follow the TREC / BEIR conventions (linear gain nDCG, ideal DCG over all
judged documents) so numbers are comparable with published BEIR results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .models import JudgeResult

# Order in which metrics are shown in tables and reports.
GENERATION_METRICS = ["faithfulness", "citation_precision", "citation_recall"]


def retrieval_metric_names(ks: Sequence[int]) -> list[str]:
    names = [f"recall@{k}" for k in ks]
    names += [f"ndcg@{k}" for k in ks]
    names += ["mrr"]
    return names


def ordered_metric_names(ks: Sequence[int]) -> list[str]:
    return retrieval_metric_names(ks) + GENERATION_METRICS


def _dedupe(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: dict[str, float], k: int) -> float:
    rel = {d for d, g in relevant.items() if g > 0}
    return len(rel.intersection(retrieved[:k])) / len(rel)


def ndcg_at_k(retrieved: Sequence[str], relevant: dict[str, float], k: int) -> float:
    dcg = sum(
        relevant.get(doc_id, 0.0) / math.log2(rank + 2) for rank, doc_id in enumerate(retrieved[:k])
    )
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: dict[str, float]) -> float:
    for rank, doc_id in enumerate(retrieved):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / (rank + 1)
    return 0.0


def retrieval_metrics(
    retrieved: Sequence[str], relevant: dict[str, float], ks: Sequence[int]
) -> dict[str, float | None]:
    """All retrieval metrics for one question. None when the question has no relevance labels."""
    names = retrieval_metric_names(ks)
    if not any(g > 0 for g in relevant.values()):
        return {name: None for name in names}
    retrieved = _dedupe(retrieved)
    out: dict[str, float | None] = {}
    for name in names:
        if name.startswith("recall@"):
            out[name] = recall_at_k(retrieved, relevant, int(name[7:]))
        elif name.startswith("ndcg@"):
            out[name] = ndcg_at_k(retrieved, relevant, int(name[5:]))
    out["mrr"] = reciprocal_rank(retrieved, relevant)
    return out


def generation_metrics(judge: JudgeResult) -> dict[str, float | None]:
    """Score one answer from the judge's claim-level verdicts.

    faithfulness        supported claims / all claims
    citation_precision  cited (claim, context) pairs where that context supports the claim
                        / all cited pairs. Citations to ids that were never retrieved count
                        as wrong. None if the answer cites nothing.
    citation_recall     supported claims carrying at least one correct citation
                        / supported claims. An answer with no citations scores 0.
    """
    empty: dict[str, float | None] = {name: None for name in GENERATION_METRICS}
    # Abstentions are tracked as a rate of their own rather than scored as (un)faithful.
    if judge.error or judge.abstained or not judge.claims:
        return empty

    claims = judge.claims
    supported = [c for c in claims if c.verdict == "supported"]
    out = dict(empty)
    out["faithfulness"] = len(supported) / len(claims)

    cited_pairs = sum(len(set(c.cited_ids)) for c in claims)
    if cited_pairs:
        correct = sum(len(set(c.cited_ids) & set(c.supporting_ids)) for c in claims)
        out["citation_precision"] = correct / cited_pairs
    if supported:
        well_cited = sum(1 for c in supported if set(c.cited_ids) & set(c.supporting_ids))
        out["citation_recall"] = well_cited / len(supported)
    return out


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
