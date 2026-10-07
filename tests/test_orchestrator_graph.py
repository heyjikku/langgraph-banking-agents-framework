import asyncio
import copy
from datetime import date
import uuid

import pytest

from agents.orchestrator.agent import AGENT_REGISTRY, PHASE1, OrchestratorAgent
from agents.orchestrator.state import merge_agent_results
from core.agent_base import AgentContext, AgentResult, SKIP_LLM, resolve_as_of
from core.data_quality import gate
from core.data_source import source
from core.memory import memory


BASE = {
    "profile": {
        "customer_id": "GRAPH-TEST",
        "first_name": "Test",
        "last_name": "Customer",
        "annual_income": 80000,
        "credit_score": 720,
        "nps_score": 7,
        "segment": "Mass Retail",
        "is_active": True,
        "onboarding_date": "2020-01-01",
        "date_of_birth": "1985-01-01",
    },
    "accounts": [{"account_type": "Savings", "balance": 10000, "status": "Active"}],
    "transactions": [],
    "interactions": [],
    "sentiment": [],
    "bureau": {},
    "persona": {},
    "fraud": [],
    "nudges": [],
}


def _agent_type(key, handler=None):
    class StubAgent:
        async def run(self, ctx):
            if handler:
                return await handler(key, ctx)
            return AgentResult(key, key, ctx.customer_id, True, {"agent": key})

    return StubAgent


def _orchestrator(monkeypatch, handler=None):
    for key in AGENT_REGISTRY:
        monkeypatch.setitem(AGENT_REGISTRY, key, _agent_type(key, handler))
    orchestrator = OrchestratorAgent()

    async def synthesise(_cleaned, _results, run_id):
        return {"run_id": run_id}

    monkeypatch.setattr(orchestrator, "_synthesise", synthesise)
    return orchestrator


async def _previous_dispatch(orchestrator, customer_id, payload):
    token = SKIP_LLM.set(True)
    try:
        cleaned, quality_report = gate.clean(payload)
        as_of_date = resolve_as_of(cleaned, orchestrator.bank_config, "2025-12-31")
        ctx = AgentContext(
            run_id="comparison",
            customer_id=customer_id,
            customer_data=cleaned,
            metadata={"as_of": as_of_date},
        )
        plan = await orchestrator._plan("full_360", None, cleaned)
        semaphore = asyncio.Semaphore(int(orchestrator.global_config.get("max_parallel_agents", 4)))
        timeout = float(orchestrator.global_config.get("agent_timeout_seconds", 30))

        async def run_parallel(key):
            async with semaphore:
                try:
                    result = await asyncio.wait_for(AGENT_REGISTRY[key]().run(ctx), timeout)
                except asyncio.TimeoutError:
                    result = AgentResult(key, key, customer_id, False, {}, error=f"timeout>{timeout}s")
                return key, result

        results = dict(await asyncio.gather(*(run_parallel(key) for key in plan["parallel"])))
        for key in plan["sequential"]:
            ctx.results_so_far = {
                name: result.output for name, result in results.items() if result.success
            }
            try:
                results[key] = await asyncio.wait_for(AGENT_REGISTRY[key]().run(ctx), timeout)
            except asyncio.TimeoutError:
                results[key] = AgentResult(key, key, customer_id, False, {}, error="timeout")
        synthesis = await orchestrator._synthesise(cleaned, results, "comparison")
        return plan, quality_report.to_dict(), results, synthesis
    finally:
        SKIP_LLM.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "requested", "expected"),
    [
        ("full_360", None, ["churn", "fraud", "clv", "sentiment", "credit_risk", "nba"]),
        ("churn_only", None, ["churn"]),
        ("fraud_only", None, ["fraud"]),
        ("nba_only", None, ["churn", "clv", "fraud", "credit_risk", "nba"]),
        ("risk_only", None, ["fraud", "churn", "credit_risk"]),
        ("custom", ["sentiment", "churn"], ["sentiment", "churn"]),
        ("auto", None, ["churn", "clv", "fraud", "credit_risk", "nba"]),
    ],
)
async def test_graph_routes_each_task_mode(monkeypatch, task, requested, expected):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    orchestrator = _orchestrator(monkeypatch)

    result = await orchestrator.run_pipeline(
        "GRAPH-TEST", BASE, task=task, requested_agents=requested,
        use_cache=False, narratives=False, as_of="2025-12-31",
    )

    assert result["agents_executed"] == expected
    assert result["agents_succeeded"] == expected
    assert result["agents_failed"] == {}


def test_result_reducer_merges_parallel_updates_without_mutating_inputs():
    churn = AgentResult("churn", "churn", "C1", True, {"score": 0.2})
    fraud = AgentResult("fraud", "fraud", "C1", True, {"score": 0.8})
    current = {"churn": churn}
    update = {"fraud": fraud}

    merged = merge_agent_results(current, update)

    assert merged == {"churn": churn, "fraud": fraud}
    assert current == {"churn": churn} and update == {"fraud": fraud}


@pytest.mark.asyncio
async def test_quality_gate_runs_once_and_agents_receive_clean_data_and_metadata(monkeypatch):
    seen = {}
    original_clean = gate.clean
    clean_calls = 0

    def counted_clean(raw):
        nonlocal clean_calls
        clean_calls += 1
        return original_clean(raw)

    async def capture(key, ctx):
        seen[key] = ctx
        return AgentResult(key, key, ctx.customer_id, True, {"agent": key})

    monkeypatch.setattr(gate, "clean", counted_clean)
    orchestrator = _orchestrator(monkeypatch, capture)
    raw = {**BASE, "accounts": [{"account_type": "Credit Card", "balance": 100, "status": "Active"}]}
    inflight = {"amount": 250, "merchant": "Example"}

    await orchestrator.run_pipeline(
        "GRAPH-TEST", raw, task="churn_only", use_cache=False,
        narratives=False, as_of="2025-12-31", inflight_transaction=inflight,
    )

    assert clean_calls == 1
    ctx = seen["churn"]
    assert ctx.customer_data["accounts"][0]["balance"] == -100
    assert ctx.customer_data["_quality"]["customer_id"] == "GRAPH-TEST"
    assert ctx.as_of == date(2025, 12, 31)
    assert ctx.metadata["inflight_transaction"] == inflight


@pytest.mark.asyncio
async def test_disabled_agent_is_filtered_and_dependencies_are_inserted(monkeypatch):
    orchestrator = _orchestrator(monkeypatch)
    agents = dict(orchestrator.bank_config.get("agents", {}))
    agents["sentiment"] = {"enabled": False}
    orchestrator.bank_config = {**orchestrator.bank_config, "agents": agents}

    result = await orchestrator.run_pipeline(
        "GRAPH-TEST", BASE, task="full_360", use_cache=False, narratives=False,
    )
    plan = await orchestrator._plan("custom", ["nba"], BASE)

    assert "sentiment" not in result["agents_executed"]
    assert result["plan"]["skipped_disabled"] == ["sentiment"]
    assert {"churn", "clv", "fraud"} <= set(plan["parallel"])
    assert plan["sequential"] == ["credit_risk", "nba"]


@pytest.mark.asyncio
async def test_credit_risk_and_nba_wait_for_successful_prerequisites(monkeypatch):
    completed = set()
    observed_priors = {}

    async def observe(key, ctx):
        if key in {"churn", "clv", "fraud"}:
            await asyncio.sleep(0.001)
            completed.add(key)
        elif key == "credit_risk":
            assert {"churn", "clv", "fraud"} <= completed
            assert ctx.prior("fraud")["agent"] == "fraud"
            observed_priors[key] = set(ctx.results_so_far)
        elif key == "nba":
            assert {"churn", "clv", "fraud", "credit_risk"} <= set(ctx.results_so_far)
            observed_priors[key] = set(ctx.results_so_far)
        return AgentResult(key, key, ctx.customer_id, True, {"agent": key})

    orchestrator = _orchestrator(monkeypatch, observe)
    result = await orchestrator.run_pipeline(
        "GRAPH-TEST", BASE, task="custom", requested_agents=["nba"],
        use_cache=False, narratives=False,
    )

    assert result["agents_succeeded"] == ["churn", "clv", "fraud", "credit_risk", "nba"]
    assert "fraud" in observed_priors["credit_risk"]
    assert {"churn", "clv", "fraud", "credit_risk"} <= observed_priors["nba"]


@pytest.mark.asyncio
async def test_parallel_limit_timeout_and_failure_preserve_successful_results(monkeypatch):
    active = 0
    max_active = 0

    async def controlled(key, ctx):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            if key == "churn":
                await asyncio.sleep(0.2)
            elif key == "fraud":
                raise RuntimeError("fraud stub failed")
            else:
                await asyncio.sleep(0.05)
            return AgentResult(key, key, ctx.customer_id, True, {"agent": key})
        finally:
            active -= 1

    orchestrator = _orchestrator(monkeypatch, controlled)
    orchestrator.global_config = {
        **orchestrator.global_config,
        "agent_timeout_seconds": 0.1,
        "max_parallel_agents": 2,
    }
    result = await orchestrator.run_pipeline(
        "GRAPH-TEST", BASE, task="custom", requested_agents=["churn", "fraud", "clv", "sentiment"],
        use_cache=False, narratives=False,
    )

    assert result["agents_failed"]["churn"] == "timeout>0.1s"
    assert result["agents_failed"]["fraud"] == "fraud stub failed"
    assert result["agents_succeeded"] == ["clv", "sentiment"]
    assert result["results"] == {"clv": {"agent": "clv"}, "sentiment": {"agent": "sentiment"}}
    assert max_active == 2


@pytest.mark.asyncio
async def test_credit_risk_timeout_is_reported_and_nba_still_runs(monkeypatch):
    async def slow_credit(key, ctx):
        if key == "credit_risk":
            await asyncio.sleep(0.1)
        return AgentResult(key, key, ctx.customer_id, True, {"agent": key})

    orchestrator = _orchestrator(monkeypatch, slow_credit)
    orchestrator.global_config = {**orchestrator.global_config, "agent_timeout_seconds": 0.01}
    result = await orchestrator.run_pipeline(
        "GRAPH-TEST", BASE, task="custom", requested_agents=["nba"],
        use_cache=False, narratives=False,
    )

    assert result["agents_failed"]["credit_risk"] == "timeout"
    assert result["agents_succeeded"] == ["churn", "clv", "fraud", "nba"]


@pytest.mark.asyncio
async def test_auto_uses_claude_plan_when_available(monkeypatch):
    orchestrator = _orchestrator(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    async def planned(*_args, **_kwargs):
        return '{"parallel": ["fraud"], "sequential": ["credit_risk"], "why": "signals"}'

    monkeypatch.setattr(orchestrator, "llm_reason", planned)
    plan = await orchestrator._plan("auto", None, BASE)

    assert plan["source"] == "claude"
    assert plan["parallel"] == ["fraud"]
    assert plan["sequential"] == ["credit_risk"]


@pytest.mark.asyncio
async def test_cache_hit_and_agent_result_persistence_are_preserved(monkeypatch):
    customer_id = f"GRAPH-CACHE-{uuid.uuid4().hex}"
    await memory.clear_customer(customer_id)
    orchestrator = _orchestrator(monkeypatch)

    first = await orchestrator.run_pipeline(
        customer_id, BASE, task="churn_only", use_cache=True, narratives=False,
    )
    first_cache_hit = first["cache_hit"]
    first_run_id = first["run_id"]
    second = await orchestrator.run_pipeline(
        customer_id, BASE, task="churn_only", use_cache=True, narratives=False,
    )
    saved = await memory.get(f"ctx:{customer_id}")

    assert first_cache_hit is False
    assert second["cache_hit"] is True
    assert second["run_id"] == first_run_id
    assert saved["churn"]["result"] == first["results"]["churn"]


@pytest.mark.asyncio
async def test_narratives_disabled_and_skip_context_reset_on_synthesis_failure(monkeypatch):
    orchestrator = OrchestratorAgent()
    result = await orchestrator.run_pipeline(
        "GRAPH-TEST", BASE, task="churn_only", use_cache=False, narratives=False,
    )
    assert result["synthesis"]["executive_narrative"] == orchestrator.template_narrative("")
    assert SKIP_LLM.get() is False

    async def fail_synthesis(_cleaned, _results, _run_id):
        assert SKIP_LLM.get() is True
        raise RuntimeError("synthesis failed")

    monkeypatch.setattr(orchestrator, "_synthesise", fail_synthesis)
    with pytest.raises(RuntimeError, match="synthesis failed"):
        await orchestrator.run_pipeline(
            "GRAPH-TEST", BASE, task="churn_only", use_cache=False, narratives=False,
        )
    assert SKIP_LLM.get() is False


@pytest.mark.asyncio
async def test_pipeline_response_fields_remain_compatible():
    orchestrator = OrchestratorAgent()
    result = await orchestrator.run_pipeline(
        "GRAPH-TEST", BASE, task="full_360", use_cache=False,
        narratives=False, as_of="2025-12-31",
    )

    assert set(result) == {
        "run_id", "started_at", "as_of_date", "customer_id", "task", "bank_profile",
        "bank_name", "plan", "total_latency_ms", "agents_executed", "agents_succeeded",
        "agents_failed", "agent_latencies_ms", "agent_confidence", "results", "synthesis",
        "data_quality", "cache_hit", "data_fingerprint",
    }
    assert set(result["synthesis"]) == {
        "health_score", "health_grade", "health_components", "health_weights_applied",
        "risk_alerts", "executive_narrative", "top_recommended_action", "churn_summary",
        "fraud_summary", "clv_summary", "credit_summary", "sentiment_summary", "run_id",
    }
    assert set(result["agents_succeeded"]) == set(AGENT_REGISTRY)


@pytest.mark.asyncio
@pytest.mark.parametrize("customer_id", ["CUST-00001", "CUST-00007"])
async def test_real_fixture_outputs_match_previous_phase_dispatch(monkeypatch, customer_id):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr(memory, "set", no_op)
    monkeypatch.setattr(memory, "save_agent_result", no_op)
    source.load()
    payload = copy.deepcopy(source.bundle(customer_id))
    orchestrator = OrchestratorAgent()
    old_plan, old_quality, old_results, old_synthesis = await _previous_dispatch(
        orchestrator, customer_id, copy.deepcopy(payload),
    )
    current = await orchestrator.run_pipeline(
        customer_id, copy.deepcopy(payload), task="full_360", use_cache=False,
        narratives=False, as_of="2025-12-31",
    )

    assert old_plan == current["plan"]
    assert old_quality == current["data_quality"]
    assert {
        key: result.output for key, result in old_results.items() if result.success
    } == current["results"]
    assert {key for key, result in old_results.items() if result.success} == set(current["agents_succeeded"])
    assert {key: value for key, value in old_synthesis.items() if key != "run_id"} == {
        key: value for key, value in current["synthesis"].items() if key != "run_id"
    }