"""
API Gateway (v2.1, post-audit)

Fixes:
  • Authentication: every /api/* route requires X-API-Key (set API_KEYS=key1,key2 in env).
    Health/docs remain open. Unset API_KEYS → refuses to start in non-DEV mode.
  • Real per-key token-bucket rate limiting (RATE_LIMIT_RPM env, default 120).
  • CORS: explicit ALLOWED_ORIGINS list; never '*' with credentials.
  • /chat now passes conversation history to Claude and persists it (Redis-backed).
  • /fraud/realtime passes the in-flight transaction (amount, hour, international, velocity,
    MCC) into the Fraud agent instead of a placeholder.
  • Agents are instantiated once (no per-request construction).
"""
from __future__ import annotations
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
import asyncio
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agents.orchestrator.agent import OrchestratorAgent, AGENT_REGISTRY
from core.agent_base import AgentContext, ACTIVE_PROFILE
from core.data_quality import gate
from core.data_source import source
from core.memory import memory, conversation_memory

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("gateway")

DEV_MODE = bool(os.getenv("DEV_MODE"))
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
if not API_KEYS and not DEV_MODE:
    raise RuntimeError("API_KEYS is empty. Set API_KEYS=key1,key2 or DEV_MODE=1 for local development.")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load the data tables at startup so the first request doesn't pay for it."""
    source.load()
    logger.info(f"startup: {len(source.list_customers())} customers, profile {ACTIVE_PROFILE}, as_of {source.as_of_date()}")
    yield


app = FastAPI(title="Banking Agentic AI Framework", version="2.2.0", lifespan=_lifespan,
              description="Quality-gated multi-agent Customer 360. Auth: X-API-Key header.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-API-Key"])

orchestrator = OrchestratorAgent()
AGENTS = {k: cls() for k, cls in AGENT_REGISTRY.items()}   # singletons; agents are stateless per run

# ── Auth + rate limit ──────────────────────────────────────────────────────────
import hmac
_buckets: Dict[str, Dict[str, float]] = {}


def _key_valid(candidate: str) -> bool:
    return any(hmac.compare_digest(candidate, k) for k in API_KEYS)


async def _rate_limited(key: str) -> bool:
    """Shared fixed-window counter in Redis when available (correct across workers and restarts);
    otherwise a per-process token bucket."""
    if memory._redis:
        try:
            window = f"rl:{key}:{int(time.time() // 60)}"
            n = await memory._redis.incr(window)
            if n == 1:
                await memory._redis.expire(window, 60)
            return n > RATE_LIMIT_RPM
        except Exception:
            pass
    b = _buckets.setdefault(key, {"tokens": float(RATE_LIMIT_RPM), "ts": time.time()})
    now = time.time()
    b["tokens"] = min(float(RATE_LIMIT_RPM), b["tokens"] + (now - b["ts"]) * RATE_LIMIT_RPM / 60)
    b["ts"] = now
    if b["tokens"] < 1:
        return True
    b["tokens"] -= 1
    return False


async def require_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    if DEV_MODE and not API_KEYS:
        return "dev"
    if not x_api_key or not _key_valid(x_api_key):
        raise HTTPException(401, "Invalid or missing X-API-Key")
    if await _rate_limited(x_api_key):
        raise HTTPException(429, f"Rate limit {RATE_LIMIT_RPM}/min exceeded")
    return x_api_key


@app.middleware("http")
async def request_meta(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]; t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = rid
    resp.headers["X-Latency-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return resp

# ── Models ─────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    customer_id: str
    customer_data: Dict[str, Any]
    task: str = Field("full_360", pattern="^(full_360|churn_only|fraud_only|nba_only|risk_only|custom|auto)$")
    requested_agents: Optional[List[str]] = None
    use_cache: bool = True

class SingleAgentRequest(BaseModel):
    customer_id: str
    customer_data: Dict[str, Any]

class ChatRequest(BaseModel):
    customer_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    customer_data: Optional[Dict[str, Any]] = None

class InflightTransaction(BaseModel):
    transaction_id: str
    amount: float
    merchant: str = ""
    merchant_category: str = ""
    channel: str = ""
    hour_of_day: int = Field(12, ge=0, le=23)
    is_international: bool = False
    transaction_count_1h: int = Field(1, ge=0)

class RealtimeFraudRequest(BaseModel):
    customer_id: str
    transaction: InflightTransaction
    customer_data: Dict[str, Any] = Field(default_factory=dict)

# ── Dashboard (served same-origin so no CORS / key plumbing in the browser) ───
# Prefers the React build (client/react-app/build); falls back to the single-file dashboard.
_CLIENT = os.path.join(os.path.dirname(__file__), "..", "client")
_REACT_BUILD = os.path.join(_CLIENT, "react-app", "build")
_DASH = os.path.join(_CLIENT, "c360_dashboard.html")
_USE_REACT = os.path.exists(os.path.join(_REACT_BUILD, "index.html"))

@app.get("/", include_in_schema=False)
async def dashboard():
    if _USE_REACT:
        return FileResponse(os.path.join(_REACT_BUILD, "index.html"), media_type="text/html")
    if not os.path.exists(_DASH):
        raise HTTPException(404, "No dashboard found: build client/react-app or keep client/c360_dashboard.html")
    return FileResponse(_DASH, media_type="text/html")

# ── Customer data (service-owned) → live agent execution ──────────────────────

def _essentials(cleaned: Dict) -> Dict:
    p, c = cleaned["profile"], cleaned["computed"]
    return {"customer_id": p.get("customer_id"), "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "segment": p.get("segment"), "segment_reliability": p.get("segment_reliability"), "occupation": p.get("occupation"),
            "city": p.get("city"), "state": p.get("state"), "since": str(p.get("onboarding_date", ""))[:4], "is_active": p.get("is_active"),
            "relationship_manager": p.get("relationship_manager"), "preferred_channel": p.get("preferred_channel"),
            "annual_income": p.get("annual_income"), "credit_score": p.get("credit_score"), "nps": p.get("nps_score"),
            "assets": c.get("total_assets"), "liabilities": c.get("total_liabilities"), "net": c.get("net_position"),
            "spend_by_category": c.get("spend_by_category"), "salary_credits": c.get("salary_credits"),
            "accounts": [{"id": a.get("account_id"), "type": a.get("account_type"), "balance": a.get("balance"), "status": a.get("status")} for a in cleaned["accounts"]],
            "interactions": [{"ts": str(i.get("timestamp", ""))[:10], "channel": i.get("channel"), "intent": i.get("intent"), "res": i.get("resolution_status"), "sat": i.get("satisfaction_score")} for i in cleaned["interactions"]][:8],
            "transactions": sorted([{"date": t.get("date"), "merchant": t.get("merchant"), "cat": t.get("category"), "amt": t.get("amount")} for t in cleaned["transactions"]], key=lambda x: x["date"] or "", reverse=True)[:10],
            "life_events": [{"type": e.get("event_type"), "date": str(e.get("detected_date", e.get("event_date", "")))[:10], "conf": e.get("confidence_score"), "source": e.get("detection_source")} for e in cleaned.get("life_events", [])][:6]}

@app.get("/api/v2/customers", tags=["Customers"], dependencies=[Depends(require_key)])
async def customers(limit: int = Query(50, ge=1, le=500)):
    return {"customers": source.list_customers(limit), "total": len(source.list_customers())}

@app.post("/api/v2/customers/{customer_id}/360", tags=["Customers"], dependencies=[Depends(require_key)])
async def customer_360(customer_id: str, task: str = Query("full_360", pattern="^(full_360|churn_only|fraud_only|nba_only|risk_only|auto)$"), refresh: bool = Query(False)):
    """Load the customer's data from the service's data source and execute the agent pipeline."""
    raw = source.bundle(customer_id)
    if raw is None:
        raise HTTPException(404, f"customer {customer_id} not found")
    run = await orchestrator.run_pipeline(customer_id, raw, task=task, use_cache=not refresh, as_of=source.as_of_date())
    cleaned, _ = gate.clean(raw)
    return {"customer": _essentials(cleaned), "run": run}

_PORTFOLIO_SEM = asyncio.Semaphore(int(os.getenv("PORTFOLIO_CONCURRENCY", "16")))


async def _summary(cid: str, refresh: bool) -> Dict:
    raw = source.bundle(cid)
    async with _PORTFOLIO_SEM:
        run = await orchestrator.run_pipeline(cid, raw, task="full_360", use_cache=not refresh, narratives=False, as_of=source.as_of_date())
    r, s = run["results"], run["synthesis"]
    cleaned, _ = gate.clean(raw); p = cleaned["profile"]
    churn, clv, nba, fraud, credit, sent = (r.get(k, {}) for k in ("churn", "clv", "nba", "fraud", "credit_risk", "sentiment"))
    clv3 = float(clv.get("clv_3_year", 0) or 0) if clv else None; pc = float(churn.get("churn_probability", 0) or 0)
    return {"customer_id": cid, "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(), "segment": p.get("segment"),
            "health": s.get("health_score"), "grade": s.get("health_grade"),
            "churn_probability": pc, "churn_level": churn.get("risk_level"),
            "fraud": fraud.get("recommendation"), "fraud_open": fraud.get("open_fraud_alerts", 0),
            "credit": credit.get("risk_appetite"), "credit_grade": credit.get("internal_grade"),
            "clv_3y": clv3, "clv_tier": clv.get("tier"), "value_at_risk": round(clv3 * pc, 0) if clv3 is not None else None,
            "nba": (nba.get("top_action") or {}).get("product_name"), "nba_revenue": sum(float(a.get("expected_revenue", 0) or 0) for a in nba.get("all_actions", [])),
            "nba_suppressed": bool(nba.get("suppressed")), "escalate": bool(sent.get("requires_escalation")),
            "trust": run["data_quality"]["trust_score"], "run_id": run["run_id"], "latency_ms": run["total_latency_ms"], "cache_hit": run.get("cache_hit", False)}

@app.post("/api/v2/portfolio", tags=["Customers"], dependencies=[Depends(require_key)])
async def portfolio(limit: int = Query(50, ge=1, le=500), refresh: bool = Query(False)):
    """Execute the pipeline (no LLM narratives) across the portfolio and return decision-level summaries."""
    ids = [c["customer_id"] for c in source.list_customers(limit)]
    t0 = time.perf_counter()
    rows = await asyncio.gather(*(_summary(cid, refresh) for cid in ids))
    n = len(rows) or 1
    agg = {"customers": len(rows), "executed_ms": round((time.perf_counter() - t0) * 1000, 1),
           "high_churn": sum(1 for x in rows if x["churn_level"] == "High"),
           "fraud_blocks": sum(1 for x in rows if x["fraud"] == "Block"), "fraud_reviews": sum(1 for x in rows if x["fraud"] == "Flag for Review"),
           "credit_declines": sum(1 for x in rows if x["credit"] == "Decline"),
           "escalations": sum(1 for x in rows if x["escalate"]),
           "clv_enabled": any(x["clv_3y"] is not None for x in rows),
           "clv_3y_total": round(sum(x["clv_3y"] or 0 for x in rows), 0), "value_at_risk": round(sum(x["value_at_risk"] or 0 for x in rows), 0),
           "nba_revenue_opportunity": round(sum(x["nba_revenue"] for x in rows), 0),
           "avg_health": round(sum(x["health"] or 0 for x in rows) / n, 1), "avg_trust": round(sum(x["trust"] for x in rows) / n, 3),
           "segments": {}}
    for x in rows:
        g = agg["segments"].setdefault(x["segment"] or "Unknown", {"n": 0, "high_churn": 0, "clv_3y": 0.0, "value_at_risk": 0.0})
        g["n"] += 1; g["high_churn"] += x["churn_level"] == "High"; g["clv_3y"] += x["clv_3y"] or 0; g["value_at_risk"] += x["value_at_risk"] or 0
    # Rank by value at risk when CLV ran; otherwise by churn probability
    rows = sorted(rows, key=lambda x: (-(x["value_at_risk"] or 0), -x["churn_probability"]))
    return {"summary": agg, "rows": rows, "as_of_date": source.as_of_date(), "bank_profile": ACTIVE_PROFILE, "bank_name": orchestrator.bank_config.get("name")}

# ── System ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "version": "2.2.0", "bank_profile": ACTIVE_PROFILE,
            "bank_name": orchestrator.bank_config.get("name"), "agents": list(AGENTS),
            "auth": "api-key" if API_KEYS else "dev-open", "llm": "claude" if os.getenv("ANTHROPIC_API_KEY") else "template-fallback",
            "memory": "redis" if memory._redis else "in-process", "customers_loaded": len(source.list_customers()), "as_of_date": source.as_of_date()}

@app.get("/agents", tags=["System"], dependencies=[Depends(require_key)])
async def list_agents():
    return {k: {"name": a.agent_name, "version": a.version, "enabled": a.is_enabled, "realtime": a.is_realtime,
                "priority": a.priority, "tools": list(a.tools), "required_inputs": a.required_inputs} for k, a in AGENTS.items()}

@app.get("/api/v2/bank-config", tags=["System"], dependencies=[Depends(require_key)])
async def bank_config():
    c = orchestrator.bank_config
    return {"profile": ACTIVE_PROFILE, "name": c.get("name"), "currency": c.get("currency"),
            "agents": c.get("agents"), "thresholds": c.get("thresholds"), "products": c.get("products"),
            "segments": c.get("segments"), "nba_segment_map": c.get("nba_segment_map")}

@app.post("/api/v2/data-quality", tags=["System"], dependencies=[Depends(require_key)])
async def data_quality(req: SingleAgentRequest):
    """Run only the quality gate — see what would be repaired/excluded before scoring."""
    cleaned, rep = gate.clean(req.customer_data)
    return {"report": rep.to_dict(), "computed": cleaned["computed"]}

# ── Orchestrator ───────────────────────────────────────────────────────────────

@app.post("/api/v2/pipeline", tags=["Orchestrator"], dependencies=[Depends(require_key)])
async def run_pipeline(req: PipelineRequest):
    return await orchestrator.run_pipeline(req.customer_id, req.customer_data, req.task, req.requested_agents, req.use_cache)

# ── Individual agents ──────────────────────────────────────────────────────────

async def _run_single(key: str, req: SingleAgentRequest, metadata: Optional[Dict] = None):
    cleaned, rep = gate.clean(req.customer_data)
    ctx = AgentContext(run_id=uuid.uuid4().hex[:12], customer_id=req.customer_id, customer_data=cleaned, metadata=metadata or {})
    r = await AGENTS[key].run(ctx)
    if not r.success:
        raise HTTPException(422 if "disabled" in (r.error or "") else 500, r.error)
    return {"agent": key, "customer_id": req.customer_id, "latency_ms": r.latency_ms, "confidence": r.confidence,
            "output": r.output, "data_quality": rep.to_dict()}

@app.post("/api/v2/agents/churn", tags=["Agents"], dependencies=[Depends(require_key)])
async def churn(req: SingleAgentRequest): return await _run_single("churn", req)

@app.post("/api/v2/agents/fraud", tags=["Agents"], dependencies=[Depends(require_key)])
async def fraud(req: SingleAgentRequest): return await _run_single("fraud", req)

@app.post("/api/v2/agents/nba", tags=["Agents"], dependencies=[Depends(require_key)])
async def nba(req: SingleAgentRequest): return await _run_single("nba", req)

@app.post("/api/v2/agents/credit-risk", tags=["Agents"], dependencies=[Depends(require_key)])
async def credit_risk(req: SingleAgentRequest): return await _run_single("credit_risk", req)

@app.post("/api/v2/agents/clv", tags=["Agents"], dependencies=[Depends(require_key)])
async def clv(req: SingleAgentRequest): return await _run_single("clv", req)

@app.post("/api/v2/agents/sentiment", tags=["Agents"], dependencies=[Depends(require_key)])
async def sentiment(req: SingleAgentRequest): return await _run_single("sentiment", req)

@app.post("/api/v2/agents/fraud/realtime", tags=["Realtime"], dependencies=[Depends(require_key)])
async def fraud_realtime(req: RealtimeFraudRequest):
    """Authorise an in-flight transaction against the customer's own history. If the caller sends no
    customer_data, the service loads the customer's bundle from its data source."""
    data = req.customer_data or source.bundle(req.customer_id) or {}
    if not data:
        raise HTTPException(404, f"customer {req.customer_id} not found and no customer_data supplied")
    cleaned, _ = gate.clean(data)
    from core.agent_base import resolve_as_of
    out = await _run_single("fraud", SingleAgentRequest(customer_id=req.customer_id, customer_data=data),
                            metadata={"inflight_transaction": req.transaction.model_dump(),
                                      "as_of": resolve_as_of(cleaned, orchestrator.bank_config, source.as_of_date())})
    o = out["output"]
    return {"transaction_id": req.transaction.transaction_id, "decision": o["recommendation"], "risk_score": o["composite_risk_score"],
            "risk_level": o["risk_level"], "inflight_signals": o["inflight_assessment"]["signals"], "rules_fired": o["rules_fired"],
            "baseline_quality": o.get("baseline_quality", "unknown"),
            "latency_ms": out["latency_ms"]}

# ── BankAI chat ────────────────────────────────────────────────────────────────

@app.post("/api/v2/chat", tags=["BankAI"], dependencies=[Depends(require_key)])
async def chat(req: ChatRequest):
    history = await conversation_memory.get(req.customer_id, last_n=8)
    ctx_lines = []
    if req.customer_data:
        cleaned, _ = gate.clean(req.customer_data)
        p = cleaned["profile"]; c = cleaned["computed"]
        sym = orchestrator.bank_config.get("currency_symbol", "$")
        ctx_lines.append(f"Customer {p.get('first_name')} {p.get('last_name')} | {p.get('segment')} | income {sym}{float(p.get('annual_income', 0) or 0):,.0f} | "
                         f"score {p.get('credit_score')} | NPS {p.get('nps_score')} | assets {sym}{c.get('total_assets', 0):,.0f} | liabilities {sym}{c.get('total_liabilities', 0):,.0f}")
        prior = await memory.get(f"ctx:{req.customer_id}") or {}
        for k, v in prior.items():
            r = v.get("result", {})
            if k == "churn": ctx_lines.append(f"Churn {r.get('churn_probability', 0):.0%} ({r.get('risk_level')})")
            if k == "fraud": ctx_lines.append(f"Fraud {r.get('recommendation')} score {r.get('composite_risk_score')}")
            if k == "credit_risk": ctx_lines.append(f"Credit {r.get('internal_grade')} {r.get('risk_appetite')}")
            if k == "clv": ctx_lines.append(f"CLV 3Y {sym}{r.get('clv_3_year', 0):,.0f} ({r.get('tier')})")
            if k == "nba" and r.get("top_action"): ctx_lines.append(f"Top NBA {r['top_action'].get('product_name')}")
    system = (f"You are BankAI for {orchestrator.bank_config.get('name')}. Help relationship managers act on customer intelligence. "
              "Be concise and data-driven. If the context lacks the needed data, say so rather than inventing figures.\n"
              + ("Context:\n" + "\n".join(ctx_lines) if ctx_lines else "No customer context supplied."))
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = (transcript + "\nuser: " if transcript else "") + req.message
    reply = await orchestrator.llm_reason(prompt, system=system, max_tokens=500)
    await conversation_memory.add(req.customer_id, "user", req.message)
    await conversation_memory.add(req.customer_id, "assistant", reply)
    return {"customer_id": req.customer_id, "reply": reply, "history_turns": len(history),
            "llm": "claude" if os.getenv("ANTHROPIC_API_KEY") else "template-fallback"}

@app.delete("/api/v2/cache/{customer_id}", tags=["System"], dependencies=[Depends(require_key)])
async def clear_cache(customer_id: str):
    n = await memory.clear_customer(customer_id)
    await conversation_memory.clear(customer_id)
    return {"cleared": True, "customer_id": customer_id, "entries": n}


if _USE_REACT:
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=os.path.join(_REACT_BUILD, "static")), name="static")
    logger.info("Serving React build from client/react-app/build")


if __name__ == "__main__":
    uvicorn.run("api.gateway:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=int(os.getenv("WORKERS", "2")))
