"""
Next Best Action Agent (v2.1, post-audit)

Fixes:
  • Peer priors come from bank_config.nba_segment_map → an NBFC never sees US products.
  • expected_revenue = config product_revenue_annual × income tier (was random.uniform).
  • urgency derives from the Churn agent output; channel from persona digital affinity.
  • Reads ctx.prior("churn") and ctx.prior("clv") — Phase 2 dependency is now real.
  • Held-product filter maps account types → product ids properly ("Credit Card" now blocks
    both credit_card_standard and credit_card_rewards).
  • Deterministic ordering with stable tie-break.
"""
from __future__ import annotations
from typing import Any, Dict, List, Set
from core.agent_base import BaseAgent, AgentContext, AgentTool, segment_key
from core.deterministic import clamp, unit


class NextBestActionAgent(BaseAgent):
    agent_name = "nba_agent"
    agent_description = "Ranked, eligible, revenue-weighted next-best-actions"
    version = "2.1.0"
    required_inputs = ["profile.segment", "profile.annual_income", "profile.credit_score", "accounts"]

    CATALOGUE = {
        "savings_account": ("High-Yield Savings", "Deposits"), "checking_account": ("Premium Checking", "Deposits"),
        "money_market": ("Money Market Account", "Deposits"), "cd": ("Certificate of Deposit", "Deposits"),
        "mortgage": ("Home Mortgage", "Loans"), "home_loan": ("Home Loan", "Loans"), "auto_loan": ("Auto Loan", "Loans"),
        "personal_loan": ("Personal Loan", "Loans"), "business_loan": ("Business Loan", "Loans"),
        "msme_loan": ("MSME Loan", "Loans"), "gold_loan": ("Gold Loan", "Loans"),
        "credit_card_standard": ("Standard Credit Card", "Cards"), "credit_card_rewards": ("Rewards Credit Card", "Cards"),
        "term_insurance": ("Term Life Insurance", "Insurance"), "insurance": ("Life Insurance", "Insurance"),
        "investment_advisory": ("Investment Advisory", "Investments"), "mutual_fund_sip": ("Mutual Fund SIP", "Investments"),
    }
    # account_type (lower) → product ids the customer already effectively holds
    HELD_MAP = {
        "savings": {"savings_account"}, "checking": {"checking_account"}, "business checking": {"checking_account"},
        "money market": {"money_market"}, "cd": {"cd"}, "mortgage": {"mortgage", "home_loan"},
        "auto loan": {"auto_loan"}, "personal loan": {"personal_loan"}, "business loan": {"business_loan", "msme_loan"},
        "credit card": {"credit_card_standard", "credit_card_rewards"}, "investment": {"investment_advisory", "mutual_fund_sip"},
        "insurance": {"term_insurance", "insurance"},
    }
    # Score floors are bureau-scale (comparable across markets); income/asset floors are currency-dependent
    # and therefore come from the active profile's `nba:` section (these are last-resort defaults).
    MIN_SCORE_FOR = {"mortgage": 620, "home_loan": 620, "business_loan": 640, "msme_loan": 640, "credit_card_rewards": 660, "auto_loan": 580, "personal_loan": 600}
    DEFAULT_MIN_INCOME_FOR = {"investment_advisory": 40000, "money_market": 30000, "mortgage": 35000}
    DEFAULT_MIN_ASSETS_FOR = {"investment_advisory": 10_000, "money_market": 5_000, "cd": 2_500}

    def build_tools(self) -> List[AgentTool]:
        return [
            AgentTool("held_products", "Products the customer already holds", self._tool_held),
            AgentTool("peer_candidates", "Bank-profile peer priors for the segment", self._tool_peers),
            AgentTool("propensity", "Feature-driven propensity per candidate", self._tool_propensity),
            AgentTool("eligibility", "Score / income / holding filters", self._tool_eligibility),
            AgentTool("rank", "Revenue-weighted ranking with churn-aware urgency", self._tool_rank),
        ]

    async def execute(self, ctx: AgentContext) -> Dict[str, Any]:
        d = ctx.customer_data
        p, persona = d["profile"], d.get("persona", {}) or {}
        churn, clv, fraud, credit = ctx.prior("churn"), ctx.prior("clv"), ctx.prior("fraud"), ctx.prior("credit_risk")

        # No cross-sell while the customer is under a fraud block — the only "next best action" is to clear it.
        if str(fraud.get("recommendation", "")) == "Block":
            hold = {"product_id": "service_clear_fraud_hold", "product_name": "Clear fraud hold before any offer", "category": "Service",
                    "propensity_score": 1.0, "expected_revenue": 0.0, "urgency": "High", "delivery_channel": "RM Call",
                    "rationale": f"{fraud.get('open_fraud_alerts', 0)} open fraud alert(s); cross-sell suppressed", "rank_score": 1.0}
            return {"top_action": hold, "all_actions": [hold], "candidates_considered": 0, "eligible_count": 0,
                    "held_products": sorted(await self.use_tool("held_products", accounts=d.get("accounts", []))),
                    "suppressed": True, "suppression_reason": "fraud_block",
                    "cross_agent_inputs": {"churn_risk_level": churn.get("risk_level"), "clv_tier": clv.get("tier"), "fraud": "Block", "credit": credit.get("risk_appetite")},
                    "existing_nudges_count": len(d.get("nudges", [])), "model": "Peer priors + feature propensity v2.1 (deterministic)"}

        held = await self.use_tool("held_products", accounts=d.get("accounts", []))
        cands = await self.use_tool("peer_candidates", segment=p.get("segment", ""))
        scored = await self.use_tool("propensity", cands=cands, profile=p, persona=persona, computed=d.get("computed", {}), churn=churn, clv=clv)
        eligible = await self.use_tool("eligibility", scored=scored, profile=p, held=held, computed=d.get("computed", {}), credit=credit)
        depth = int(self.get_segment_config(p.get("segment", "")).get("nba_depth", 3))
        final = await self.use_tool("rank", eligible=eligible, profile=p, persona=persona, churn=churn, depth=depth)

        if final:
            sym = self.bank_config.get("currency_symbol", "$")
            pitch = await self.llm_reason(
                f"One-sentence pitch to {p.get('first_name', 'the customer')} ({p.get('segment')}) for {final[0]['product_name']} "
                f"via {final[0]['delivery_channel']}. Income {sym}{float(p.get('annual_income', 0) or 0):,.0f}.",
                system="Personal banking advisor. One conversational sentence, no jargon.")
            final[0]["llm_pitch"] = None if pitch == self.template_narrative("") else pitch

        return {
            "top_action": final[0] if final else None,
            "all_actions": final,
            "candidates_considered": len(cands), "eligible_count": len(eligible),
            "held_products": sorted(held),
            "suppressed": False,
            "cross_agent_inputs": {"churn_risk_level": churn.get("risk_level"), "clv_tier": clv.get("tier"), "fraud": fraud.get("recommendation"), "credit": credit.get("risk_appetite")},
            "existing_nudges_count": len(d.get("nudges", [])),
            "model": "Peer priors + feature propensity v2.1 (deterministic)",
        }

    # ── Tools ──────────────────────────────────────────────────────────────

    def _tool_held(self, accounts: list) -> Set[str]:
        held: Set[str] = set()
        for a in accounts:
            t = str(a.get("account_type", "")).lower()
            for k, ids in self.HELD_MAP.items():
                if k in t:
                    held |= ids
        return held

    def _tool_peers(self, segment: str) -> List[str]:
        smap = self.bank_config.get("nba_segment_map", {})
        enabled = set(self.get_enabled_products())
        base = smap.get(segment_key(segment)) or smap.get("mass_retail") or list(enabled)
        return [x for x in base if x in enabled]

    def _tool_propensity(self, cands: List[str], profile: Dict, persona: Dict, computed: Dict, churn: Dict, clv: Dict) -> List[Dict]:
        income = float(profile.get("annual_income", 0) or 0)
        cs = int(profile.get("credit_score", 680) or 680)
        dig = float(persona.get("digital_affinity_score", 50) or 50)
        risk_app = float(persona.get("risk_appetite_score", 50) or 50)
        assets = float(computed.get("total_assets", 0) or 0)
        liab = float(computed.get("total_liabilities", 0) or 0)
        spend = computed.get("spend_by_category", {}) or {}
        revenue_tbl = self.bank_config.get("product_revenue_annual", {})
        churn_p = float(churn.get("churn_probability", 0.2) or 0.2)
        cid = profile.get("customer_id", "x")

        out = []
        for pid in cands:
            name, cat = self.CATALOGUE.get(pid, (pid, "Other"))
            s = 0.45
            # Investment products: assets + risk appetite
            if cat == "Investments":
                s += 0.20 * clamp(assets / 200_000, 0, 1) + 0.15 * (risk_app - 50) / 100
            # Deposits: cash-rich, low risk appetite
            if cat == "Deposits":
                s += 0.15 * clamp(assets / 100_000, 0, 1) + 0.10 * (50 - risk_app) / 100
            # Loans: good score, low existing leverage
            if cat == "Loans":
                s += 0.15 * clamp((cs - 600) / 250, 0, 1) - 0.20 * clamp(liab / max(income, 1) / 2, 0, 1)
            # Cards: spending signal
            if cat == "Cards":
                s += 0.20 * clamp(sum(spend.values()) / 15_000, 0, 1)
            # Insurance: income + loans (protection need)
            if cat == "Insurance":
                s += 0.10 * clamp(income / 100_000, 0, 1) + (0.10 if liab > 0 else 0)
            # Retention bias: at-risk customers get sticky, low-friction products boosted
            if churn_p > 0.5 and cat in ("Deposits", "Cards"):
                s += 0.08
            # Tiny deterministic tie-break so ordering is stable but not degenerate
            s += 0.02 * unit(cid, pid)
            income_tier = 0.6 if income < 40_000 else 1.0 if income < 120_000 else 1.5
            out.append({
                "product_id": pid, "product_name": name, "category": cat,
                "propensity_score": round(clamp(s, 0.05, 0.98), 3),
                "expected_revenue": round(float(revenue_tbl.get(pid, 200)) * income_tier, 0),
                "rationale": self._rationale(cat, assets, liab, cs, income, churn_p),
            })
        return out

    def _rationale(self, cat, assets, liab, cs, income, churn_p) -> str:
        if cat == "Investments": return f"deposits {assets:,.0f} available to deploy"
        if cat == "Deposits": return "cash-rich, conservative profile" if assets > 50_000 else "build savings habit"
        if cat == "Loans": return f"score {cs}, leverage {liab / max(income, 1):.1f}× income"
        if cat == "Cards": return "spending pattern supports card usage"
        if cat == "Insurance": return "protection gap vs income and liabilities"
        return "segment peer prior"

    def _tool_eligibility(self, scored: List[Dict], profile: Dict, held: Set[str], computed: Dict, credit: Dict) -> List[Dict]:
        cs = int(profile.get("credit_score", 680) or 680)
        income = float(profile.get("annual_income", 0) or 0)
        assets = float(computed.get("total_assets", 0) or 0)
        credit_declined = str(credit.get("risk_appetite", "")) == "Decline"
        min_income = self.param("nba", "min_income_for", self.DEFAULT_MIN_INCOME_FOR) or {}
        min_assets = self.param("nba", "min_assets_for", self.DEFAULT_MIN_ASSETS_FOR) or {}
        out = []
        for it in scored:
            pid = it["product_id"]
            if pid in held: continue
            if cs < self.MIN_SCORE_FOR.get(pid, 0): continue
            if income < float(min_income.get(pid, 0)): continue
            if assets < float(min_assets.get(pid, 0)): continue
            if credit_declined and it["category"] in ("Loans", "Cards"): continue   # never offer credit to a declined customer
            out.append(it)
        return out

    def _tool_rank(self, eligible: List[Dict], profile: Dict, persona: Dict, churn: Dict, depth: int) -> List[Dict]:
        dig = float(persona.get("digital_affinity_score", 50) or 50)
        pref = str(profile.get("preferred_channel", "") or "")
        lvl = str(churn.get("risk_level", "Low") or "Low")
        for it in eligible:
            it["urgency"] = "High" if lvl == "High" else "Medium" if lvl == "Medium" or it["category"] == "Loans" else "Low"
            it["delivery_channel"] = ("RM Call" if lvl == "High" else pref if pref else "Mobile App" if dig >= 60 else "Email" if dig >= 35 else "Branch")
            it["rank_score"] = round(it["propensity_score"] * 0.7 + clamp(it["expected_revenue"] / 3000, 0, 1) * 0.3, 4)
        return sorted(eligible, key=lambda x: (-x["rank_score"], x["product_id"]))[:depth]

    def template_narrative(self, prompt: str) -> str:
        return "Recommendation ranked by propensity and expected revenue; channel and urgency follow churn and persona."
