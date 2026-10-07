"""
CLV Agent (v2.1, post-audit)

Fixes:
  • BG/NBD on one transaction is meaningless. Below `min_txns_for_bgnbd` the agent
    switches to a relationship-value model built on deposits, liabilities (NIM), product
    count and tenure. (Previously an HNW client with $607K on deposit scored $1,200.)
  • Known `profile.lifetime_value` is blended in as a prior when present.
  • Wallet share is deposits ÷ estimated investable assets by segment — no more 80% cap for everyone.
  • Deterministic.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List
from core.agent_base import BaseAgent, AgentContext, AgentTool, segment_key
from core.deterministic import clamp


class CLVAgent(BaseAgent):
    agent_name = "clv_agent"
    agent_description = "Hybrid CLV: BG/NBD when history allows, relationship-value model otherwise"
    version = "2.1.0"
    required_inputs = ["profile.annual_income", "accounts", "profile.onboarding_date"]

    # Defaults only — the active profile's `clv:` section overrides these (currency-dependent).
    DEFAULTS = {"deposit_yield": 0.018, "liability_yield": 0.032, "product_fee": 95.0,
                "investable_multiple": {"mass_retail": 0.8, "mass_affluent": 2.0, "small_business": 2.5, "high_net_worth": 6.0, "ultra_hnw": 15.0}}

    def _p(self, key):
        return self.param("clv", key, self.DEFAULTS[key])

    def build_tools(self) -> List[AgentTool]:
        return [
            AgentTool("relationship_value", "Annual revenue from balances, products and tenure", self._tool_relationship),
            AgentTool("bgnbd", "Transactional frequency/monetary model (only with enough history)", self._tool_bgnbd),
            AgentTool("blend_and_project", "Blend models + known LTV; project 1/3/5 years", self._tool_project),
            AgentTool("wallet_share", "Deposits ÷ segment-estimated investable assets", self._tool_wallet),
        ]

    async def execute(self, ctx: AgentContext) -> Dict[str, Any]:
        d = ctx.customer_data
        p, comp = d["profile"], d.get("computed", {})
        txns = d.get("transactions", [])
        tenure = max(0.5, ctx.as_of.year - int(str(p.get("onboarding_date", "2020"))[:4]))
        seg = p.get("segment", "")
        seg_cfg = self.get_segment_config(seg)

        rel = await self.use_tool("relationship_value", computed=comp, n_products=len(d.get("accounts", [])), tenure=tenure)
        min_txn = int(self.get_threshold("min_txns_for_bgnbd", 6))
        bg = await self.use_tool("bgnbd", transactions=txns, tenure=tenure) if len(txns) >= min_txn else {"used": False, "annual_revenue": 0.0, "reason": f"{len(txns)} txns < {min_txn}"}
        proj = await self.use_tool("blend_and_project", rel=rel, bg=bg, known_ltv=p.get("lifetime_value"),
                                   seg_mult=float(seg_cfg.get("clv_multiplier", 1.0)), tenure=tenure)
        wallet = await self.use_tool("wallet_share", computed=comp, income=float(p.get("annual_income", 0) or 0), seg=seg)

        floor = self.get_threshold("clv_premium_floor", 200000)
        clv3 = proj["clv_3y"]
        tier = "Platinum" if clv3 >= floor * 2 else "Gold" if clv3 >= floor else "Silver" if clv3 >= floor * 0.4 else "Standard"
        strategy = {"Platinum": "Retain & deepen wallet share", "Gold": "Grow with premium bundle",
                    "Silver": "Develop via cross-sell", "Standard": "Activate via digital engagement"}[tier]
        clv_score = round(clamp(clv3 / (floor * 2) * 60 + float(p.get("nps_score", 7) or 7) / 10 * 20 + min(len(d.get("accounts", [])), 5) / 5 * 20, 0, 100), 1)

        sym = self.bank_config.get("currency_symbol", "$")
        insight = await self.llm_reason(
            f"3Y CLV {sym}{clv3:,.0f} ({tier}), wallet share {wallet['wallet_share_pct']:.0f}%, model {proj['primary_model']}. "
            "One-sentence growth opportunity for the RM.", system="You are a CLV analyst. One sentence.")

        return {
            "clv_score": clv_score, "clv_1_year": proj["clv_1y"], "clv_3_year": clv3, "clv_5_year": proj["clv_5y"],
            "annual_revenue_estimate": proj["annual_revenue"],
            "revenue_potential": round(clv3 * 0.08, 0),
            "wallet_share_pct": wallet["wallet_share_pct"], "competitor_gap": wallet["competitor_gap"],
            "tier": tier, "tier_strategy": strategy,
            "model_selection": {"primary_model": proj["primary_model"], "bgnbd": bg, "relationship": rel, "known_ltv_used": proj["known_ltv_used"]},
            "segment_multiplier": float(seg_cfg.get("clv_multiplier", 1.0)),
            "llm_insight": insight,
            "model": "Hybrid CLV v2.1 (relationship-value + BG/NBD)",
        }

    # ── Tools ──────────────────────────────────────────────────────────────

    def _tool_relationship(self, computed: Dict, n_products: int, tenure: float) -> Dict:
        assets = float(computed.get("total_assets", 0) or 0)
        liab = float(computed.get("total_liabilities", 0) or 0)
        dy, ly, fee = float(self._p("deposit_yield")), float(self._p("liability_yield")), float(self._p("product_fee"))
        rev = assets * dy + liab * ly + n_products * fee
        return {"annual_revenue": round(rev, 2), "deposit_income": round(assets * dy, 2),
                "lending_income": round(liab * ly, 2), "fee_income": round(n_products * fee, 2),
                "tenure_years": tenure, "parameters": {"deposit_yield": dy, "liability_yield": ly, "product_fee": fee}}

    def _tool_bgnbd(self, transactions: list, tenure: float) -> Dict:
        debits = [abs(float(t.get("amount", 0))) for t in transactions if float(t.get("amount", 0)) < 0]
        freq_per_year = len(transactions) / tenure
        avg_order = sum(debits) / len(debits) if debits else 0.0
        # p(alive): more recent & frequent → closer to 1 (logistic on frequency)
        p_alive = 1 / (1 + 2.718 ** (-(freq_per_year / 12 - 1)))
        annual_rev = freq_per_year * avg_order * 0.012 * p_alive   # interchange/fee take ≈1.2%
        return {"used": True, "frequency_per_year": round(freq_per_year, 2), "avg_debit": round(avg_order, 2),
                "p_alive": round(p_alive, 3), "annual_revenue": round(annual_rev, 2)}

    def _tool_project(self, rel: Dict, bg: Dict, known_ltv, seg_mult: float, tenure: float) -> Dict:
        rel_rev, bg_rev = rel["annual_revenue"], bg.get("annual_revenue", 0.0)
        if bg.get("used"):
            annual = 0.6 * rel_rev + 0.4 * bg_rev; primary = "blended"
        else:
            annual = rel_rev; primary = "relationship_value"
        annual *= seg_mult
        # Known LTV (if present) is a lifetime figure; convert to an annual prior over observed tenure and blend 30%
        used_ltv = False
        if known_ltv:
            try:
                prior_annual = float(known_ltv) / max(tenure, 1) * 0.10  # assume LTV field is gross value; 10% margin
                annual = 0.7 * annual + 0.3 * prior_annual; used_ltv = True
            except (TypeError, ValueError):
                pass
        g = 1.05 + min(tenure / 50, 0.05)   # 5–10% growth
        clv1 = annual
        clv3 = annual * (1 + g + g ** 2)
        clv5 = annual * sum(g ** i for i in range(5))
        return {"annual_revenue": round(annual, 0), "clv_1y": round(clv1, 0), "clv_3y": round(clv3, 0), "clv_5y": round(clv5, 0),
                "primary_model": primary, "known_ltv_used": used_ltv, "growth": round(g - 1, 3)}

    def _tool_wallet(self, computed: Dict, income: float, seg: str) -> Dict:
        deposits = float(computed.get("total_assets", 0) or 0)
        mult = float((self._p("investable_multiple") or {}).get(segment_key(seg), 1.0))
        # Investable estimate: the larger of income-based or 1.25× what we already hold (they clearly have at least that)
        investable = max(income * mult, deposits * 1.25, 1.0)
        share = clamp(deposits / investable * 100, 0, 95)
        return {"wallet_share_pct": round(share, 1), "estimated_investable_assets": round(investable, 0),
                "competitor_gap": round(max(investable - deposits, 0), 0)}

    def template_narrative(self, prompt: str) -> str:
        return "CLV derived from balances, product depth and tenure; see model_selection for which model drove it."
