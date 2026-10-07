"""
Credit Risk Agent (v2.1, post-audit)

Fixes:
  • Hard decline gates from bank config: PD > credit_hard_decline_pd OR
    delinquencies ≥ credit_hard_decline_delinquencies → max_loan = 0.
    (Previously a 14% PD customer with 3 delinquencies got $172K.)
  • Max loan is capped by an income multiple that depends on the internal GRADE.
  • Reads fraud output from ctx.prior("fraud"): open fraud raises PD and can force Decline.
  • Bureau contradictions (score 800 + 3 delinquencies) reduce reliance on the score.
  • Deterministic — no random confidence.
"""
from __future__ import annotations
from typing import Any, Dict, List
from core.agent_base import BaseAgent, AgentContext, AgentTool
from core.deterministic import clamp


class CreditRiskAgent(BaseAgent):
    agent_name = "credit_risk_agent"
    agent_description = "IRB-style PD/LGD/EAD scoring with hard policy gates"
    version = "2.1.0"
    required_inputs = ["profile.credit_score", "profile.annual_income", "bureau.debt_to_income", "bureau.utilization_pct", "bureau.payment_history_pct"]

    GRADES = [(0.015, "AAA"), (0.025, "AA"), (0.040, "A"), (0.065, "BBB"), (0.100, "BB"), (0.160, "B"), (1.0, "CCC")]
    BANDS = {"AAA": "Exceptional", "AA": "Very Good", "A": "Good", "BBB": "Acceptable", "BB": "Watch", "B": "Substandard", "CCC": "Doubtful"}

    def build_tools(self) -> List[AgentTool]:
        return [
            AgentTool("features", "Scorecard inputs from profile + cleaned bureau", self._tool_features),
            AgentTool("pd", "Probability of default", self._tool_pd),
            AgentTool("lgd_ead", "Loss-given-default and exposure", self._tool_lgd_ead),
            AgentTool("policy_gates", "Hard decline / cap rules from bank config", self._tool_gates),
        ]

    async def execute(self, ctx: AgentContext) -> Dict[str, Any]:
        d = ctx.customer_data
        p, bur = d["profile"], d.get("bureau", {}) or {}
        fraud_prior = ctx.prior("fraud")

        f = await self.use_tool("features", profile=p, bureau=bur, accounts=d.get("accounts", []), computed=d.get("computed", {}))
        pd_r = await self.use_tool("pd", f=f, fraud_prior=fraud_prior)
        le = await self.use_tool("lgd_ead", f=f)
        grade = next(g for t, g in self.GRADES if pd_r["pd"] <= t)
        gates = await self.use_tool("policy_gates", f=f, pd=pd_r["pd"], grade=grade, fraud_prior=fraud_prior)

        scorecard = {
            "Payment history": round(f["payment_history_pct"] / 100 * 35, 1),
            "Credit utilization": round((1 - f["utilization"]) * 30, 1),
            "Credit age": round(min(f["oldest_account_years"] / 20, 1) * 15, 1),
            "Credit mix": round(min(f["num_accounts"] / 5, 1) * 10, 1),
            "New credit": round(max(0.0, 10 - f["hard_inquiries_12m"] * 2), 1),
        }
        sym = self.bank_config.get("currency_symbol", "$")
        narrative = await self.llm_reason(
            f"Grade {grade}, PD {pd_r['pd']:.1%}, DTI {f['dti']:.0%}, delinquencies {f['delinquent_accounts']}, "
            f"decision {gates['risk_appetite']}, max loan {sym}{gates['max_loan_amount']:,.0f}. One-sentence credit recommendation.",
            system="You are a credit risk analyst. Regulatory-aware, one sentence.")

        return {
            "probability_of_default": pd_r["pd"],
            "pd_components": pd_r["components"],
            "loss_given_default": le["lgd"],
            "exposure_at_default": le["ead"],
            "expected_loss": round(pd_r["pd"] * le["lgd"] * le["ead"], 0),
            "internal_grade": grade,
            "bureau_band": self.BANDS[grade],
            "credit_score": f["credit_score"],
            "score_reliability": bur.get("score_reliability", "ok"),
            "bureau_source": bur.get("bureau_source", "n/a"),
            "debt_to_income": round(f["dti"], 3),
            "utilization_pct": round(f["utilization"] * 100, 1),
            "payment_history_pct": f["payment_history_pct"],
            "delinquent_accounts": f["delinquent_accounts"],
            "max_eligible_loan": gates["max_loan_amount"],
            "income_multiple_cap": gates["income_multiple"],
            "recommended_rate": gates["rate"],
            "risk_appetite": gates["risk_appetite"],
            "decline_reasons": gates["decline_reasons"],
            "fraud_adjustment_applied": pd_r["components"].get("fraud_prior", 0) != 0,
            "scorecard_breakdown": scorecard,
            "llm_summary": narrative,
            "model": "IRB scorecard v2.1 + policy gates",
        }

    # ── Tools ──────────────────────────────────────────────────────────────

    def _tool_features(self, profile: Dict, bureau: Dict, accounts: list, computed: Dict) -> Dict:
        income = max(float(profile.get("annual_income", 0) or 0), 1.0)
        # DTI: prefer bureau ratio (quality gate exposes debt_to_income_ratio); else derive from liabilities
        if bureau.get("debt_to_income_ratio") is not None:
            dti = float(bureau["debt_to_income_ratio"])
        else:
            liabilities = float(computed.get("total_liabilities", 0) or 0)
            dti = liabilities / income
        util = clamp(float(bureau.get("utilization_pct", 40) or 40) / 100, 0, 1.5)
        return {
            "credit_score": int(profile.get("credit_score", bureau.get("credit_score", 680)) or 680),
            "score_reliability": bureau.get("score_reliability", "ok"),
            "income": income, "dti": round(clamp(dti, 0, 2.0), 4), "utilization": util,
            "payment_history_pct": float(bureau.get("payment_history_pct", 90) or 90),
            "hard_inquiries_12m": int(bureau.get("hard_inquiries_12m", 1) or 0),
            "delinquent_accounts": int(bureau.get("delinquent_accounts", 0) or 0),
            "oldest_account_years": float(bureau.get("oldest_account_years", 5) or 5),
            "total_credit_limit": float(bureau.get("total_credit_limit", 30000) or 30000),
            "num_accounts": int(bureau.get("total_accounts", len(accounts)) or len(accounts)),
        }

    def _tool_pd(self, f: Dict, fraud_prior: Dict) -> Dict:
        comp = {}
        cs = f["credit_score"]
        score_mult = 0.25 if cs >= 800 else 0.40 if cs >= 750 else 0.70 if cs >= 700 else 1.30 if cs >= 650 else 2.20 if cs >= 600 else 4.50
        if f["score_reliability"] == "low":       # bureau contradiction → don't trust a high score
            score_mult = max(score_mult, 1.3)
        comp["score"] = 0.05 * score_mult
        dti = f["dti"]
        comp["dti"] = comp["score"] * (1.0 if dti > 0.60 else 0.5 if dti > 0.45 else 0.0) - (comp["score"] * 0.3 if dti < 0.20 else 0)
        comp["delinquencies"] = 0.03 * f["delinquent_accounts"]
        comp["payment_history"] = max(0.0, (95 - f["payment_history_pct"]) / 100) * 0.4
        comp["utilization"] = max(0.0, f["utilization"] - self.get_threshold("credit_max_utilization", 0.8)) * 0.15
        comp["inquiries"] = 0.005 * max(0, f["hard_inquiries_12m"] - 2)
        # Cross-agent: open fraud alerts raise PD
        fr_open = int(fraud_prior.get("open_fraud_alerts", 0) or 0)
        fr_score = float(fraud_prior.get("composite_risk_score", 0) or 0)
        comp["fraud_prior"] = 0.04 * fr_open + (0.05 if fr_score >= 60 else 0.0)
        pd = round(clamp(sum(comp.values()), 0.005, 0.95), 4)
        return {"pd": pd, "components": {k: round(v, 4) for k, v in comp.items()}}

    def _tool_lgd_ead(self, f: Dict) -> Dict:
        lgd = clamp(0.40 + f["dti"] * 0.15 + (0.05 if f["delinquent_accounts"] else 0), 0.25, 0.85)
        ead = max(f["total_credit_limit"] * min(f["utilization"], 1.0), 5000.0)
        return {"lgd": round(lgd, 3), "ead": round(ead, 0)}

    def _tool_gates(self, f: Dict, pd: float, grade: str, fraud_prior: Dict) -> Dict:
        reasons = []
        if pd > self.get_threshold("credit_hard_decline_pd", 0.20):
            reasons.append(f"PD {pd:.1%} exceeds policy max {self.get_threshold('credit_hard_decline_pd', 0.20):.0%}")
        if f["delinquent_accounts"] >= int(self.get_threshold("credit_hard_decline_delinquencies", 3)):
            reasons.append(f"{f['delinquent_accounts']} delinquent accounts ≥ policy limit")
        if f["dti"] > self.get_threshold("credit_max_dti", 0.50):
            reasons.append(f"DTI {f['dti']:.0%} exceeds max {self.get_threshold('credit_max_dti', 0.5):.0%}")
        if str(fraud_prior.get("recommendation", "")) == "Block":
            reasons.append("Fraud agent recommends Block")
        caps = self.bank_config.get("loan_income_multiple_by_grade", {})
        multiple = float(caps.get(grade, 1.0))
        if reasons:
            appetite, max_loan = "Decline", 0.0
        elif pd > 0.10:
            appetite, max_loan = "Monitor", round(f["income"] * multiple * 0.6, -3)
        else:
            appetite, max_loan = "Within Limit", round(f["income"] * multiple, -3)
        rate = round(6.5 + pd * 45 + (0.5 if f["utilization"] > 0.7 else 0), 2)
        return {"risk_appetite": appetite, "max_loan_amount": max_loan, "income_multiple": multiple,
                "rate": rate, "decline_reasons": reasons}

    def template_narrative(self, prompt: str) -> str:
        return "Decision reflects PD, policy gates and grade-based exposure caps; see decline_reasons."
