"""
Orchestrator Agent (v2.1, post-audit)

Fixes:
  • Every payload passes through DataQualityGate BEFORE any agent runs; the quality
    report is returned with the result.
  • _plan() genuinely asks Claude (JSON-constrained) when an API key exists and the task
    is 'auto'; deterministic plans otherwise. Either way the plan is validated against the
    registry and bank feature flags.
  • Cache key includes a fingerprint of the customer payload — changed data never serves stale.
  • max_parallel_agents is enforced with an asyncio.Semaphore.
  • Health score is re-normalised over agents that actually ran — no default padding when
    a bank disables sentiment or CLV. Weights are exposed in the output.
  • Phase 2 agents receive Phase 1 outputs via ctx.results_so_far (NBA / Credit consume them).
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from core.agent_base import BaseAgent, AgentContext, AgentResult, AgentTool, ACTIVE_PROFILE, SKIP_LLM, resolve_as_of
from core.data_quality import gate
from core.deterministic import data_fingerprint
from core.memory import memory

from agents.churn.agent import ChurnRiskAgent
from agents.fraud.agent import FraudDetectionAgent
from agents.nba.agent import NextBestActionAgent
from agents.credit_risk.agent import CreditRiskAgent
from agents.clv.agent import CLVAgent
from agents.sentiment.agent import SentimentAgent

logger = logging.getLogger("orchestrator")

AGENT_REGISTRY: Dict[str, type] = {
    "churn": ChurnRiskAgent, "fraud": FraudDetectionAgent, "nba": NextBestActionAgent,
    "credit_risk": CreditRiskAgent, "clv": CLVAgent, "sentiment": SentimentAgent,
}
PHASE1 = ["churn", "fraud", "clv", "sentiment"]          # independent
PHASE2 = ["credit_risk", "nba"]                          # consume phase-1 outputs; nba also reads credit_risk
TASK_PLANS = {
    "full_360":   {"parallel": PHASE1, "sequential": PHASE2},
    "churn_only": {"parallel": ["churn"], "sequential": []},
    "fraud_only": {"parallel": ["fraud"], "sequential": []},
    "nba_only":   {"parallel": ["churn", "clv", "fraud"], "sequential": ["credit_risk", "nba"]},
    "risk_only":  {"parallel": ["fraud", "churn"], "sequential": ["credit_risk"]},
}
HEALTH_WEIGHTS = {"churn": 0.30, "clv": 0.25, "credit_risk": 0.25, "sentiment": 0.20}


class OrchestratorAgent(BaseAgent):
    agent_name = "orchestrator"
    agent_description = "Quality-gate → plan → parallel dispatch → phase-2 → synthesis"
    version = "2.1.0"

    def build_tools(self) -> List[AgentTool]:
        return []

    async def execute(self, ctx: AgentContext) -> Dict[str, Any]:
        return {}

    # ── Public entry ───────────────────────────────────────────────────────

    async def run_pipeline(self, customer_id: str, customer_data: Dict, task: str = "full_360",
                           requested_agents: Optional[List[str]] = None, use_cache: bool = True,
                           inflight_transaction: Optional[Dict] = None, narratives: bool = True,
                           as_of: Optional[str] = None) -> Dict[str, Any]:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        token = SKIP_LLM.set(not narratives)
        fp = data_fingerprint(customer_data)
        cache_key = f"pipeline:{customer_id}:{task}:{fp}:{','.join(requested_agents or [])}:{'n' if narratives else 'b'}:{as_of or ''}"

        if use_cache and not inflight_transaction:
            cached = await memory.get(cache_key)
            if cached:
                cached["cache_hit"] = True
                return cached

        # 1. Quality gate — agents never see raw data
        cleaned, qreport = gate.clean(customer_data)
        as_of_date = resolve_as_of(cleaned, self.bank_config, as_of)
        meta: Dict[str, Any] = {"as_of": as_of_date}
        if inflight_transaction: meta["inflight_transaction"] = inflight_transaction
        ctx = AgentContext(run_id=run_id, customer_id=customer_id, customer_data=cleaned, metadata=meta)

        # 2. Plan
        plan = await self._plan(task, requested_agents, cleaned)

        # 3. Phase 1 (parallel, bounded)
        results: Dict[str, AgentResult] = await self._dispatch_parallel(plan["parallel"], ctx)

        # 4. Phase 2 (sequential, sees phase-1 outputs)
        for key in plan["sequential"]:
            ctx.results_so_far = {k: v.output for k, v in results.items() if v.success}
            r = await self._dispatch_single(key, ctx)
            if r: results[key] = r

        # 5. Synthesis
        try:
            synthesis = await self._synthesise(cleaned, results, run_id)
        finally:
            SKIP_LLM.reset(token)

        out = {
            "run_id": run_id, "started_at": started_at, "as_of_date": as_of_date.isoformat(), "customer_id": customer_id, "task": task, "bank_profile": ACTIVE_PROFILE,
            "bank_name": self.bank_config.get("name"), "plan": plan,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "agents_executed": list(results), "agents_succeeded": [k for k, v in results.items() if v.success],
            "agents_failed": {k: v.error for k, v in results.items() if not v.success},
            "agent_latencies_ms": {k: v.latency_ms for k, v in results.items()},
            "agent_confidence": {k: v.confidence for k, v in results.items() if v.success},
            "results": {k: v.output for k, v in results.items() if v.success},
            "synthesis": synthesis,
            "data_quality": qreport.to_dict(),
            "cache_hit": False, "data_fingerprint": fp,
        }
        if not inflight_transaction:
            await memory.set(cache_key, out, ttl=int(self.global_config.get("cache_ttl_seconds", 300)))
            for k, v in results.items():
                if v.success: await memory.save_agent_result(customer_id, k, v.output)
        return out

    # ── Planning ───────────────────────────────────────────────────────────

    def _is_enabled(self, key: str) -> bool:
        return key in AGENT_REGISTRY and self.bank_config.get("agents", {}).get(key, {}).get("enabled", True)

    async def _plan(self, task: str, requested: Optional[List[str]], cleaned: Dict) -> Dict:
        source = "preset"
        if task == "custom" and requested:
            plan = {"parallel": [a for a in requested if a in PHASE1], "sequential": [a for a in requested if a in PHASE2]}
        elif task == "auto":
            plan, source = await self._llm_plan(cleaned)
        else:
            base = TASK_PLANS.get(task, TASK_PLANS["full_360"])
            plan = {"parallel": list(base["parallel"]), "sequential": list(base["sequential"])}
        # Validate against registry + feature flags
        plan["parallel"] = [a for a in plan["parallel"] if self._is_enabled(a)]
        plan["sequential"] = [a for a in plan["sequential"] if self._is_enabled(a)]
        # Phase-2 agents need their inputs; auto-add missing dependencies
        if "nba" in plan["sequential"]:
            for dep in ("churn", "clv", "fraud"):
                if self._is_enabled(dep) and dep not in plan["parallel"]: plan["parallel"].append(dep)
            if self._is_enabled("credit_risk") and "credit_risk" not in plan["sequential"]: plan["sequential"].insert(0, "credit_risk")
            # credit_risk must precede nba
            plan["sequential"] = [a for a in plan["sequential"] if a != "nba"] + ["nba"]
        if "credit_risk" in plan["sequential"] and self._is_enabled("fraud") and "fraud" not in plan["parallel"]:
            plan["parallel"].append("fraud")
        skipped = [a for a in AGENT_REGISTRY if a not in plan["parallel"] + plan["sequential"]]
        return {**plan, "source": source, "skipped_disabled": [a for a in skipped if not self._is_enabled(a)]}

    async def _llm_plan(self, cleaned: Dict) -> tuple[Dict, str]:
        """Ask Claude which agents matter for THIS customer; fall back to a signal-driven heuristic."""
        p = cleaned["profile"]; q = cleaned.get("_quality", {})
        summary = (f"segment={p.get('segment')} active={p.get('is_active')} nps={p.get('nps_score')} score={p.get('credit_score')} "
                   f"fraud_alerts={len(cleaned.get('fraud', []))} open={sum(1 for f in cleaned.get('fraud', []) if str(f.get('status','')).lower()=='open')} "
                   f"sentiment_records={len(cleaned.get('sentiment', []))} interactions={len(cleaned.get('interactions', []))} "
                   f"liabilities={cleaned.get('computed', {}).get('total_liabilities', 0):.0f} data_trust={q.get('trust_score')}")
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                txt = await self.llm_reason(
                    f"Customer signals: {summary}\nAvailable agents: {list(AGENT_REGISTRY)}. "
                    "Return ONLY JSON: {\"parallel\": [...], \"sequential\": [...], \"why\": \"...\"} choosing the agents worth running. "
                    "'credit_risk' then 'nba' must be sequential (in that order); others parallel.",
                    system="You are a banking analytics planner. Output strict JSON, nothing else.", max_tokens=200)
                js = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
                return {"parallel": [a for a in js.get("parallel", []) if a in PHASE1],
                        "sequential": [a for a in js.get("sequential", []) if a in PHASE2], "why": js.get("why", "")}, "claude"
            except Exception as e:
                logger.warning(f"LLM plan failed ({e}); heuristic plan")
        # Heuristic: always churn; fraud if alerts or flagged txns; sentiment if any records; clv+nba if active; credit if liabilities
        par, seq = ["churn"], []
        if cleaned.get("fraud") or any(t.get("is_flagged") for t in cleaned.get("transactions", [])): par.append("fraud")
        if cleaned.get("sentiment") or cleaned.get("interactions"): par.append("sentiment")
        if p.get("is_active", True): par.append("clv"); seq.append("nba")
        if cleaned.get("computed", {}).get("total_liabilities", 0) > 0 or p.get("credit_score"): seq.append("credit_risk")
        return {"parallel": par, "sequential": seq, "why": "signal-driven heuristic"}, "heuristic"

    # ── Dispatch ───────────────────────────────────────────────────────────

    async def _dispatch_parallel(self, keys: List[str], ctx: AgentContext) -> Dict[str, AgentResult]:
        if not keys: return {}
        sem = asyncio.Semaphore(int(self.global_config.get("max_parallel_agents", 4)))
        timeout = float(self.global_config.get("agent_timeout_seconds", 30))

        async def one(k: str):
            async with sem:
                try:
                    return k, await asyncio.wait_for(AGENT_REGISTRY[k]().run(ctx), timeout)
                except asyncio.TimeoutError:
                    return k, AgentResult(k, k, ctx.customer_id, False, {}, error=f"timeout>{timeout}s")
        return dict(await asyncio.gather(*(one(k) for k in keys if k in AGENT_REGISTRY)))

    async def _dispatch_single(self, key: str, ctx: AgentContext) -> Optional[AgentResult]:
        if key not in AGENT_REGISTRY: return None
        try:
            return await asyncio.wait_for(AGENT_REGISTRY[key]().run(ctx), float(self.global_config.get("agent_timeout_seconds", 30)))
        except asyncio.TimeoutError:
            return AgentResult(key, key, ctx.customer_id, False, {}, error="timeout")

    # ── Synthesis ──────────────────────────────────────────────────────────

    async def _synthesise(self, cleaned: Dict, results: Dict[str, AgentResult], run_id: str) -> Dict[str, Any]:
        p = cleaned["profile"]
        r = {k: v.output for k, v in results.items() if v.success}
        churn, fraud, nba, credit, clv, sent = (r.get(k, {}) for k in ("churn", "fraud", "nba", "credit_risk", "clv", "sentiment"))

        # Health: each component in 0..1, only for agents that ran; renormalise weights
        comps = {}
        if churn: comps["churn"] = 1 - float(churn.get("churn_probability", 0.5))
        if clv: comps["clv"] = float(clv.get("clv_score", 0)) / 100
        if credit:
            pd_ = float(credit.get("probability_of_default", 0.1)); comps["credit_risk"] = max(0.0, 1 - pd_ / 0.25)
        if sent: comps["sentiment"] = float(sent.get("sentiment_score", 0.5))
        wsum = sum(HEALTH_WEIGHTS[k] for k in comps) or 1.0
        health = round(sum(comps[k] * HEALTH_WEIGHTS[k] for k in comps) / wsum * 100, 1) if comps else None
        grade = None if health is None else "Excellent" if health > 82 else "Good" if health > 68 else "Fair" if health > 52 else "At Risk"

        alerts = []
        if churn.get("risk_level") == "High": alerts.append({"type": "churn", "severity": "high", "text": f"Churn risk {churn.get('risk_score', 0):.0f}%"})
        if fraud.get("recommendation") in ("Block", "Flag for Review"): alerts.append({"type": "fraud", "severity": "high" if fraud["recommendation"] == "Block" else "medium", "text": f"Fraud: {fraud['recommendation']} ({fraud.get('open_fraud_alerts', 0)} open)"})
        if credit.get("risk_appetite") == "Decline": alerts.append({"type": "credit", "severity": "high", "text": "Credit: Decline — " + "; ".join(credit.get("decline_reasons", []))})
        if sent.get("requires_escalation"): alerts.append({"type": "sentiment", "severity": "medium", "text": "CX escalation required"})
        elif sent.get("open_case_backlog"): alerts.append({"type": "sentiment", "severity": "low", "text": f"{sent.get('open_interactions', 0)} unresolved cases"})
        if sent.get("sources_agree") is False: alerts.append({"type": "data", "severity": "low", "text": sent.get("contradiction_note", "Sentiment sources disagree")})
        dq = cleaned.get("_quality", {})
        if dq.get("trust_score", 1) < 0.7: alerts.append({"type": "data", "severity": "medium", "text": f"Data trust {dq['trust_score']:.2f} — {dq.get('issue_count', 0)} issues repaired/excluded"})
        if not alerts: alerts.append({"type": "none", "severity": "info", "text": "No critical flags"})

        sym = self.bank_config.get("currency_symbol", "$")
        narrative = await self.llm_reason(
            f"{p.get('first_name')} {p.get('last_name')}, {p.get('segment')}, {p.get('city')}. Health {health} ({grade}). "
            f"Churn {churn.get('churn_probability', 0):.0%}; CLV 3Y {sym}{clv.get('clv_3_year', 0):,.0f} ({clv.get('tier')}); "
            f"credit {credit.get('internal_grade')} / {credit.get('risk_appetite')}; fraud {fraud.get('recommendation')}; "
            f"top NBA {(nba.get('top_action') or {}).get('product_name')}. Data trust {dq.get('trust_score')}. "
            "Three sentences for the RM: value, key risk, single most important action.",
            system=f"Chief Customer Intelligence AI for {self.bank_config.get('name')}. Concise, data-driven, address the RM directly.")

        return {
            "health_score": health, "health_grade": grade, "health_components": {k: round(v, 3) for k, v in comps.items()},
            "health_weights_applied": {k: round(HEALTH_WEIGHTS[k] / wsum, 3) for k in comps},
            "risk_alerts": alerts, "executive_narrative": narrative,
            "top_recommended_action": nba.get("top_action"),
            "churn_summary": {"probability": churn.get("churn_probability"), "level": churn.get("risk_level"), "action": (churn.get("recommended_action") or {}).get("action")},
            "fraud_summary": {"risk_score": fraud.get("composite_risk_score"), "open_alerts": fraud.get("open_fraud_alerts"), "recommendation": fraud.get("recommendation")},
            "clv_summary": {"tier": clv.get("tier"), "clv_3y": clv.get("clv_3_year"), "wallet_share": clv.get("wallet_share_pct"), "model": (clv.get("model_selection") or {}).get("primary_model")},
            "credit_summary": {"grade": credit.get("internal_grade"), "max_loan": credit.get("max_eligible_loan"), "appetite": credit.get("risk_appetite"), "decline_reasons": credit.get("decline_reasons")},
            "sentiment_summary": {"overall": sent.get("overall_sentiment"), "csat": sent.get("avg_csat"), "escalate": sent.get("requires_escalation"), "sources_agree": sent.get("sources_agree")},
            "run_id": run_id,
        }

    def template_narrative(self, prompt: str) -> str:
        return "Executive narrative unavailable offline — health_score, risk_alerts and top_recommended_action carry the decision."
