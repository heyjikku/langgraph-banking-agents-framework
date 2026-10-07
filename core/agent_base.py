"""
Banking Agentic AI Framework — BaseAgent (v2.1, post-audit)

Fixes vs v2.0:
  • BANK_PROFILE env var now overrides YAML ACTIVE_PROFILE (deploy-time switching works)
  • Segment lookup uses an alias table ("Ultra High Net Worth" → ultra_hnw)
  • No global `random` — confidence is derived from data completeness
  • Cross-agent reads are explicit via ctx.prior(agent_key)
"""
from __future__ import annotations
import contextvars
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ── Config loading (env var wins) ─────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/bank_config.yaml")


def load_bank_config(profile: Optional[str] = None) -> tuple[Dict, Dict, str]:
    try:
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
    except Exception as e:  # pragma: no cover
        logger.error(f"Config load failed: {e}")
        return {}, {}, "unknown"
    active = profile or os.getenv("BANK_PROFILE") or raw.get("ACTIVE_PROFILE", "mid_scale_bank")
    if active not in raw.get("profiles", {}):
        logger.warning(f"Profile '{active}' not found; falling back to ACTIVE_PROFILE")
        active = raw.get("ACTIVE_PROFILE", next(iter(raw["profiles"])))
    return raw["profiles"][active], raw.get("global", {}), active


BANK_CONFIG, GLOBAL_CONFIG, ACTIVE_PROFILE = load_bank_config()

# Set to True by the orchestrator for bulk/portfolio runs — agents then use template narratives only.
SKIP_LLM: contextvars.ContextVar[bool] = contextvars.ContextVar("skip_llm", default=False)

# Segment name → config key. Add aliases here, never in agent code.
SEGMENT_ALIASES = {
    "mass retail": "mass_retail", "retail": "mass_retail", "basic": "mass_retail",
    "mass affluent": "mass_affluent", "affluent": "mass_affluent", "premium": "mass_affluent",
    "small business": "small_business", "sme": "small_business", "msme": "small_business",
    "high net worth": "high_net_worth", "hnw": "high_net_worth", "hni": "high_net_worth",
    "ultra high net worth": "ultra_hnw", "uhnw": "ultra_hnw", "ultra-hni": "ultra_hnw", "ultra hnw": "ultra_hnw",
}


def segment_key(segment: str) -> str:
    s = (segment or "").strip().lower()
    return SEGMENT_ALIASES.get(s, s.replace(" ", "_").replace("-", "_"))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgentTool:
    name: str
    description: str
    fn: Callable
    is_async: bool = False


@dataclass
class ThoughtStep:
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None


@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    customer_id: str
    success: bool
    output: Dict[str, Any]
    reasoning_chain: List[ThoughtStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    confidence: float = 0.0
    error: Optional[str] = None


@dataclass
class AgentContext:
    """Shared context for one pipeline run. `customer_data` is ALWAYS post-DataQualityGate."""
    run_id: str
    customer_id: str
    customer_data: Dict[str, Any]
    bank_config: Dict = field(default_factory=lambda: BANK_CONFIG)
    results_so_far: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prior(self, agent_key: str) -> Dict[str, Any]:
        """Output of an agent that already ran in this pipeline (empty dict if not)."""
        return self.results_so_far.get(agent_key, {}) or {}

    @property
    def as_of(self) -> date:
        """The date every 'recency', 'age' and 'tenure' is measured from. Never the wall clock
        unless nothing else is known — see resolve_as_of()."""
        v = self.metadata.get("as_of")
        return v if isinstance(v, date) else resolve_as_of(self.customer_data, self.bank_config)


def resolve_as_of(customer_data: Dict[str, Any], bank_config: Dict, explicit: Optional[str] = None) -> date:
    """Priority: explicit arg → profile as_of_date → latest activity date in the payload → today."""
    for cand in (explicit, bank_config.get("as_of_date"), (customer_data.get("computed") or {}).get("latest_activity_date")):
        if cand:
            try:
                return datetime.strptime(str(cand)[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
    return date.today()


_ANTHROPIC = None


def _anthropic_client(api_key: str):
    """One AsyncAnthropic per process — connection pooling instead of a new client per call."""
    global _ANTHROPIC
    if _ANTHROPIC is None:
        import anthropic
        _ANTHROPIC = anthropic.AsyncAnthropic(api_key=api_key)
    return _ANTHROPIC


# ── Base Agent ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    agent_name: str = "base_agent"
    agent_description: str = "Base banking agent"
    version: str = "2.1.0"
    # Dotted paths this agent needs; drives the data-completeness confidence.
    required_inputs: List[str] = []

    def __init__(self):
        self.agent_id = f"{self.agent_name}-{uuid.uuid4().hex[:8]}"
        self.tools: Dict[str, AgentTool] = {}
        self.bank_config = BANK_CONFIG
        self.global_config = GLOBAL_CONFIG
        self.logger = logging.getLogger(f"agent.{self.agent_name}")
        self._tools_used: List[str] = []
        for tool in self.build_tools():
            self.tools[tool.name] = tool
        cfg_key = self.agent_name.replace("_agent", "")
        agent_cfg = self.bank_config.get("agents", {}).get(cfg_key, {})
        self.is_enabled = agent_cfg.get("enabled", True)
        self.is_realtime = agent_cfg.get("realtime", False)
        self.priority = agent_cfg.get("priority", 3)

    @abstractmethod
    def build_tools(self) -> List[AgentTool]: ...

    @abstractmethod
    async def execute(self, ctx: AgentContext) -> Dict[str, Any]: ...

    # ── Run wrapper ──────────────────────────────────────────────────────────

    async def run(self, ctx: AgentContext) -> AgentResult:
        if not self.is_enabled:
            return AgentResult(self.agent_id, self.agent_name, ctx.customer_id, False, {},
                               error=f"{self.agent_name} disabled in bank profile '{ACTIVE_PROFILE}'")
        t0 = time.perf_counter()
        self._tools_used = []
        chain = [ThoughtStep(f"Analysing {ctx.customer_id}: {self.agent_description}", "execute")]
        try:
            output = await self.execute(ctx)
            conf = self.data_completeness(ctx)
            output.setdefault("confidence", conf)
            output["data_completeness"] = conf
            output["data_trust_score"] = ctx.customer_data.get("_quality", {}).get("trust_score", 1.0)
            chain.append(ThoughtStep("Synthesised output", observation=f"{len(output)} fields, tools={self._tools_used}"))
            ms = round((time.perf_counter() - t0) * 1000, 2)
            return AgentResult(self.agent_id, self.agent_name, ctx.customer_id, True, output,
                               chain, list(self._tools_used), ms, output["confidence"])
        except Exception as e:
            self.logger.error(f"[{self.agent_name}] {e}", exc_info=True)
            return AgentResult(self.agent_id, self.agent_name, ctx.customer_id, False, {},
                               error=str(e), latency_ms=(time.perf_counter() - t0) * 1000)

    # ── Confidence = how much required input actually exists ────────────────

    def data_completeness(self, ctx: AgentContext) -> float:
        if not self.required_inputs:
            return 0.9
        d = ctx.customer_data
        present = 0
        for path in self.required_inputs:
            cur: Any = d
            ok = True
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur and cur[part] not in (None, "", [], {}):
                    cur = cur[part]
                else:
                    ok = False
                    break
            present += ok
        base = present / len(self.required_inputs)
        trust = d.get("_quality", {}).get("trust_score", 1.0)
        return round(0.5 + 0.5 * base * (0.6 + 0.4 * trust), 3)

    # ── Tool dispatch ────────────────────────────────────────────────────────

    async def use_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered in {self.agent_name}")
        self._tools_used.append(name)
        tool = self.tools[name]
        return await tool.fn(**kwargs) if tool.is_async else tool.fn(**kwargs)

    # ── Config helpers ───────────────────────────────────────────────────────

    def get_threshold(self, key: str, default: float) -> float:
        return float(self.bank_config.get("thresholds", {}).get(key, default))

    def get_enabled_products(self) -> List[str]:
        return list(self.bank_config.get("products", []))

    def get_segment_config(self, segment: str) -> Dict:
        return self.bank_config.get("segments", {}).get(segment_key(segment), {"clv_multiplier": 1.0, "nba_depth": 3})

    def param(self, section: str, key: str, default: Any = None) -> Any:
        """Currency-dependent model parameter from the active profile, e.g. param("clv", "product_fee")."""
        return (self.bank_config.get(section) or {}).get(key, default)

    # ── LLM (Claude) with deterministic fallback ─────────────────────────────

    async def llm_reason(self, prompt: str, system: str = "", max_tokens: int = 400) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and not SKIP_LLM.get():
            try:
                client = _anthropic_client(api_key)
                resp = await client.messages.create(
                    model=self.global_config.get("llm_model", "claude-sonnet-4-6"),
                    max_tokens=max_tokens,
                    system=system or f"You are the {self.agent_name} for {self.bank_config.get('name', 'a bank')}. Be concise and actionable.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                self.logger.warning(f"LLM call failed ({e}); using template narrative")
        return self.template_narrative(prompt)

    def template_narrative(self, prompt: str) -> str:
        """Override in subclasses for a data-driven fallback. Never a canned sentence."""
        return "Narrative unavailable (LLM offline). See structured output."

    def __repr__(self):
        return f"<{self.agent_name} v{self.version} enabled={self.is_enabled} profile={ACTIVE_PROFILE}>"
