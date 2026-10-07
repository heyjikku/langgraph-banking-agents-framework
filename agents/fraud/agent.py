"""
Fraud Detection Agent (v2.1, post-audit)

Fixes:
  • Alert weight depends on STATUS: Open=1.0, Under Review=0.6, Resolved/False Positive=0.05.
    Three resolved false-positives previously scored 66 → "Flag for Review". Now ~8 → Allow.
  • FRAUD_TYPE_WEIGHTS is actually applied (was dead code).
  • No random noise in the anomaly score — a decision must not flip on retry.
  • Accepts an `inflight_transaction` (amount, hour, international, new merchant) so the
    realtime endpoint evaluates the transaction being authorised, not a placeholder.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from core.agent_base import BaseAgent, AgentContext, AgentTool
from core.deterministic import clamp


class FraudDetectionAgent(BaseAgent):
    agent_name = "fraud_agent"
    agent_description = "Anomaly scoring + rule engine on history and the in-flight transaction"
    version = "2.1.0"
    required_inputs = ["transactions", "fraud", "accounts"]

    FRAUD_TYPE_WEIGHTS = {
        "account takeover": 0.95, "identity theft": 0.90, "sim swap fraud": 0.90,
        "money laundering suspect": 0.85, "synthetic identity": 0.80,
        "high value transaction": 0.55, "card not present": 0.55, "unusual location": 0.45,
    }
    STATUS_WEIGHTS = {"open": 1.0, "investigating": 0.8, "under review": 0.6, "escalated": 0.9,
                      "resolved": 0.05, "false positive": 0.05, "closed": 0.05, "confirmed fraud": 1.0}
    SEVERITY_WEIGHTS = {"critical": 1.0, "high": 0.75, "medium": 0.45, "low": 0.2}

    def build_tools(self) -> List[AgentTool]:
        return [
            AgentTool("history_signals", "Behavioural baseline from cleaned transaction history", self._tool_history),
            AgentTool("alert_pressure", "Status- and type-weighted pressure from existing alerts", self._tool_alerts),
            AgentTool("inflight_anomaly", "Deviation of the in-flight transaction from the baseline", self._tool_inflight),
            AgentTool("rule_engine", "Bank-configured hard rules", self._tool_rules),
            AgentTool("decide", "Combine scores against bank thresholds", self._tool_decide),
        ]

    async def execute(self, ctx: AgentContext) -> Dict[str, Any]:
        d = ctx.customer_data
        inflight: Optional[Dict] = ctx.metadata.get("inflight_transaction")
        hist = await self.use_tool("history_signals", transactions=d.get("transactions", []))
        alerts = await self.use_tool("alert_pressure", fraud=d.get("fraud", []), as_of=ctx.as_of)
        infl = await self.use_tool("inflight_anomaly", txn=inflight, baseline=hist)
        rules = await self.use_tool("rule_engine", hist=hist, alerts=alerts, infl=infl)
        decision = await self.use_tool("decide", hist=hist, alerts=alerts, infl=infl, rules=rules)

        note = await self.llm_reason(
            f"Fraud composite {decision['composite_risk_score']} → {decision['recommendation']}. "
            f"Open alerts {alerts['open_count']}, rules fired {rules['fired']}. One-sentence analyst note.",
            system="You are a fraud analyst. One precise sentence.")

        open_alerts = [a for a in d.get("fraud", []) if str(a.get("status", "")).lower() == "open"]
        return {
            "composite_risk_score": decision["composite_risk_score"],
            "risk_level": decision["risk_level"],
            "recommendation": decision["recommendation"],
            "score_components": decision["components"],
            "alert_pressure": alerts,
            "inflight_assessment": infl,
            "rules_fired": rules["fired"],
            "open_fraud_alerts": len(open_alerts),
            "total_fraud_alerts": len(d.get("fraud", [])),
            "flagged_transactions": hist["flagged_count"],
            "baseline_quality": hist["baseline_quality"], "baseline_debits": hist["n_debits"],
            "alert_details": [{
                "id": a.get("alert_id"), "type": a.get("fraud_type"), "severity": a.get("severity"),
                "status": a.get("status"), "confidence": a.get("confidence_score"), "risk_score": a.get("risk_score"),
                "weight_applied": round(self._alert_weight(a), 3),
            } for a in d.get("fraud", [])],
            "remediation_steps": self._remediation(decision["recommendation"]),
            "sla_hours": {"Block": 1, "Flag for Review": 4}.get(decision["recommendation"], 720),
            "llm_observation": note,
            "model": "Weighted anomaly + rules v2.1 (deterministic)",
        }

    # ── Tools ──────────────────────────────────────────────────────────────

    def _tool_history(self, transactions: list) -> Dict:
        debits = [abs(float(t.get("amount", 0))) for t in transactions if float(t.get("amount", 0)) < 0]
        n = len(debits)
        mean = sum(debits) / n if n else 0.0
        var = sum((x - mean) ** 2 for x in debits) / n if n else 0.0
        std = var ** 0.5
        channels = {str(t.get("channel", "")) for t in transactions}
        merchants = {str(t.get("merchant", "")).lower() for t in transactions}
        return {
            "n_debits": n, "mean_debit": round(mean, 2), "std_debit": round(std, 2),
            "max_debit": round(max(debits, default=0.0), 2),
            "flagged_count": sum(1 for t in transactions if t.get("is_flagged")),
            "known_channels": sorted(channels), "known_merchants": merchants,
            "baseline_quality": "ok" if n >= 8 else "thin",
        }

    def _alert_weight(self, a: Dict) -> float:
        s = self.STATUS_WEIGHTS.get(str(a.get("status", "")).lower(), 0.5)
        t = self.FRAUD_TYPE_WEIGHTS.get(str(a.get("fraud_type", "")).lower(), 0.5)
        sev = self.SEVERITY_WEIGHTS.get(str(a.get("severity", "")).lower(), 0.4)
        conf = clamp(float(a.get("confidence_score", 0.5) or 0.5), 0, 1)
        return s * (0.5 * t + 0.3 * sev + 0.2 * conf)

    def _tool_alerts(self, fraud: list, as_of=None) -> Dict:
        from datetime import datetime as _dt
        max_age = float(self.param("fraud", "max_alert_age_days", 365))
        def _age_factor(a):
            if as_of is None: return 1.0
            try:
                d = _dt.strptime(str(a.get("alert_date", ""))[:10], "%Y-%m-%d").date()
            except ValueError:
                return 1.0
            age = (as_of - d).days
            return 1.0 if age <= max_age else max(0.25, 1 - (age - max_age) / (2 * max_age))
        weights = [self._alert_weight(a) * _age_factor(a) for a in fraud]
        open_ = [a for a in fraud if str(a.get("status", "")).lower() in ("open", "investigating", "escalated", "confirmed fraud")]
        # Pressure = max live alert + diminishing sum of the rest, scaled to 0..100
        if weights:
            sw = sorted(weights, reverse=True)
            pressure = sw[0] + sum(w * 0.5 ** i for i, w in enumerate(sw[1:], 1))
        else:
            pressure = 0.0
        return {
            "pressure_score": round(clamp(pressure * 70, 0, 100), 1),
            "open_count": len(open_),
            "resolved_count": len(fraud) - len(open_),
            "max_live_weight": round(max([self._alert_weight(a) for a in open_], default=0.0), 3),
        }

    def _tool_inflight(self, txn: Optional[Dict], baseline: Dict) -> Dict:
        if not txn:
            return {"present": False, "anomaly_score": 0.0, "signals": []}
        amt = abs(float(txn.get("amount", 0)))
        signals, score = [], 0.0
        mean, std = baseline["mean_debit"], baseline["std_debit"]
        if baseline["n_debits"] >= 3 and std > 0:
            z = (amt - mean) / std
            if z > 3: signals.append(f"amount {z:.1f}σ above baseline"); score += 30
            elif z > 2: signals.append(f"amount {z:.1f}σ above baseline"); score += 15
        elif amt > float(self.param("fraud", "thin_baseline_high_value", 10_000)):
            signals.append("high value with thin baseline"); score += 20
        hour = int(txn.get("hour_of_day", 12))
        if hour < 5 or hour >= 23: signals.append(f"off-hours ({hour:02d}:00)"); score += 12
        if txn.get("is_international"): signals.append("international"); score += 15
        m = str(txn.get("merchant", "")).lower()
        if m and m not in baseline["known_merchants"]: signals.append("first-seen merchant"); score += 8
        ch = str(txn.get("channel", ""))
        if ch and baseline["known_channels"] and ch not in baseline["known_channels"]: signals.append(f"new channel {ch}"); score += 8
        if int(txn.get("transaction_count_1h", 1)) > 4: signals.append("velocity >4/hour"); score += 20
        if str(txn.get("merchant_category", "")).lower() in {"cryptocurrency", "gift cards", "casino", "international wire"}:
            signals.append("high-risk MCC"); score += 15
        return {"present": True, "anomaly_score": round(clamp(score, 0, 100), 1), "signals": signals}

    def _tool_rules(self, hist: Dict, alerts: Dict, infl: Dict) -> Dict:
        fired = []
        if alerts["open_count"] > 0 and alerts["max_live_weight"] >= 0.8: fired.append("R01 live critical-type alert")
        if alerts["open_count"] >= 2: fired.append("R02 multiple live alerts")
        if hist["flagged_count"] > 0: fired.append("R03 flagged transactions in history")
        if infl["present"] and infl["anomaly_score"] >= 45: fired.append("R04 in-flight anomaly ≥ 45")
        if infl["present"] and "velocity >4/hour" in infl["signals"]: fired.append("R05 velocity")
        return {"fired": fired, "count": len(fired)}

    def _tool_decide(self, hist: Dict, alerts: Dict, infl: Dict, rules: Dict) -> Dict:
        # Weighted composite. When no in-flight txn, alert pressure dominates.
        # In-flight authorisation: the transaction's own anomaly must be able to block on its own,
        # with alert pressure as an additive escalator. Portfolio review (no in-flight): alerts dominate.
        w_alert, w_infl, w_rules = (0.20, 0.65, 0.15) if infl["present"] else (0.80, 0.0, 0.20)
        rule_score = clamp(rules["count"] * 25, 0, 100)
        comp = w_alert * alerts["pressure_score"] + w_infl * infl["anomaly_score"] + w_rules * rule_score
        comp = round(clamp(comp, 0, 100), 1)
        block, flag = self.get_threshold("fraud_block_score", 75), self.get_threshold("fraud_flag_score", 45)
        # Hard overrides: an open critical-type alert, or an extreme in-flight anomaly, always blocks
        if alerts["open_count"] and alerts["max_live_weight"] >= 0.85:
            comp = max(comp, block)
        if infl["present"] and infl["anomaly_score"] >= 80:
            comp = max(comp, block)
        rec = "Block" if comp >= block else "Flag for Review" if comp >= flag else "Allow"
        lvl = "Critical" if comp >= block else "High" if comp >= 60 else "Medium" if comp >= flag else "Low"
        return {"composite_risk_score": comp, "recommendation": rec, "risk_level": lvl,
                "components": {"alert_pressure": alerts["pressure_score"], "inflight_anomaly": infl["anomaly_score"],
                               "rules": rule_score, "weights": {"alert": w_alert, "inflight": w_infl, "rules": w_rules}}}

    def _remediation(self, rec: str) -> List[str]:
        return {
            "Block": ["Decline / freeze outbound", "Notify customer on registered channels", "Fraud ops P1 queue",
                      "SAR review if AML-type alert", "Compliance review ≤ 24h"],
            "Flag for Review": ["Soft hold + step-up authentication", "Analyst queue", "Customer contact ≤ 4h"],
            "Allow": ["Standard monitoring"],
        }.get(rec, [])

    def template_narrative(self, prompt: str) -> str:
        return "Decision is driven by live alert pressure and in-flight anomaly; resolved alerts are discounted."
