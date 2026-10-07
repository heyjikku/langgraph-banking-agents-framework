from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated, Any, Dict, TypedDict

from core.agent_base import AgentResult
from core.data_quality import QualityReport


def merge_agent_results(
    current: Dict[str, AgentResult] | None,
    update: Dict[str, AgentResult] | None,
) -> Dict[str, AgentResult]:
    """Merge independently produced agent results without dropping sibling updates."""
    return {**(current or {}), **(update or {})}


class PipelineState(TypedDict, total=False):
    run_id: str
    customer_id: str
    task: str
    customer_data: Dict[str, Any]
    requested_agents: list[str] | None
    inflight_transaction: Dict[str, Any] | None
    as_of_override: str | None
    agent_semaphore: asyncio.Semaphore
    cleaned_data: Dict[str, Any]
    quality_report: QualityReport
    as_of_date: date
    metadata: Dict[str, Any]
    plan: Dict[str, Any]
    results: Annotated[Dict[str, AgentResult], merge_agent_results]
    synthesis: Dict[str, Any]