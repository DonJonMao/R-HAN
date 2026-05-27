from __future__ import annotations

from typing import Optional

from .types import EvalSummary, SearchResult, StructureSummary


def resolve_result_summary(result: SearchResult) -> Optional[EvalSummary]:
    if result.final_summary is not None:
        return result.final_summary
    return result.best_node.tier2


def resolve_result_signature(result: SearchResult) -> str:
    if result.final_signature:
        return result.final_signature
    return result.best_node.compiled.signature()


def resolve_result_output(result: SearchResult) -> str:
    summary = resolve_result_summary(result)
    if summary is None or not summary.evaluations:
        return ""
    return summary.evaluations[0].raw_output


def is_union_result(result: SearchResult) -> bool:
    return result.final_summary is not None


def resolve_structure_summary(result: SearchResult) -> Optional[StructureSummary]:
    return result.structure_summary


def resolve_structure_signature(result: SearchResult) -> str:
    if result.structure_summary is not None:
        return result.structure_summary.signature
    return resolve_result_signature(result)


def has_structure_output(result: SearchResult) -> bool:
    return result.structure_summary is not None


def resolve_primary_reward(result: SearchResult) -> Optional[float]:
    summary = resolve_result_summary(result)
    if summary is not None:
        return summary.mean_reward
    structure = resolve_structure_summary(result)
    if structure is not None:
        return structure.metrics.total_reward
    return None
