from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from core.agent_base import AgentContext, AgentResult, resolve_as_of
from core.data_quality import gate
from agents.orchestrator.state import PipelineState


def build_pipeline_graph(owner, agent_registry: Dict[str, type], phase1: List[str]):
    async def prepare(state: PipelineState) -> Dict[str, Any]:
        cleaned, quality_report = gate.clean(state["customer_data"])
        as_of_date = resolve_as_of(cleaned, owner.bank_config, state.get("as_of_override"))
        metadata: Dict[str, Any] = {"as_of": as_of_date}
        inflight = state.get("inflight_transaction")
        if inflight:
            metadata["inflight_transaction"] = inflight
        plan = await owner._plan(state["task"], state.get("requested_agents"), cleaned)
        return {
            "cleaned_data": cleaned,
            "quality_report": quality_report,
            "as_of_date": as_of_date,
            "metadata": metadata,
            "plan": plan,
        }

    def agent_node(agent_key: str):
        async def run(state: PipelineState) -> Dict[str, Any]:
            selected = state["plan"]["parallel" if agent_key in phase1 else "sequential"]
            if agent_key not in selected:
                return {}

            ctx = AgentContext(
                run_id=state["run_id"],
                customer_id=state["customer_id"],
                customer_data=state["cleaned_data"],
                bank_config=owner.bank_config,
                results_so_far={
                    key: result.output
                    for key, result in state["results"].items()
                    if result.success
                },
                metadata=state["metadata"],
            )
            timeout = float(owner.global_config.get("agent_timeout_seconds", 30))
            try:
                async with state["agent_semaphore"]:
                    result = await asyncio.wait_for(agent_registry[agent_key]().run(ctx), timeout)
            except asyncio.TimeoutError:
                error = f"timeout>{timeout}s" if agent_key in phase1 else "timeout"
                result = AgentResult(agent_key, agent_key, ctx.customer_id, False, {}, error=error)
            except Exception as exc:
                owner.logger.exception("[%s] failed outside BaseAgent.run", agent_key)
                result = AgentResult(agent_key, agent_key, ctx.customer_id, False, {}, error=str(exc))
            return {"results": {agent_key: result}}

        return run

    async def synthesise(state: PipelineState) -> Dict[str, Any]:
        output = await owner._synthesise(state["cleaned_data"], state["results"], state["run_id"])
        return {"synthesis": output}

    builder = StateGraph(PipelineState)
    builder.add_node("prepare", prepare)
    for agent_key in agent_registry:
        builder.add_node(agent_key, agent_node(agent_key))
    builder.add_node("after_independent", lambda _state: {})
    builder.add_node("after_credit_risk", lambda _state: {})
    builder.add_node("synthesise", synthesise)

    builder.add_edge(START, "prepare")
    for agent_key in phase1:
        builder.add_edge("prepare", agent_key)
    builder.add_edge(list(phase1), "after_independent")
    builder.add_conditional_edges(
        "after_independent",
        lambda state: "credit_risk" if "credit_risk" in state["plan"]["sequential"] else "after_credit_risk",
        {"credit_risk": "credit_risk", "after_credit_risk": "after_credit_risk"},
    )
    builder.add_edge("credit_risk", "after_credit_risk")
    builder.add_conditional_edges(
        "after_credit_risk",
        lambda state: "nba" if "nba" in state["plan"]["sequential"] else "synthesise",
        {"nba": "nba", "synthesise": "synthesise"},
    )
    builder.add_edge("nba", "synthesise")
    builder.add_edge("synthesise", END)
    return builder.compile()