"""
Data Quality Gate — runs BEFORE any agent touches customer data.

The audit of the input dataset found:
  • 26% of bureau records have open_accounts > total_accounts
  • 79% have utilization_pct inconsistent with balance/limit
  • 25% of sentiment labels contradict the feedback text
  • 100% of "closed" accounts still carry balances
  • Credit-card "balances" up to $250K stored as positive numbers
  • Salary credits from Costco / Uber / Shell Gas
  • risk_level contradicts credit_score in 58 customers

A model scored on garbage produces confident garbage. This gate:
  1. Normalises field names (the dataset uses two schemas)
  2. Repairs what can be derived (utilization from balance/limit, sentiment from text)
  3. Excludes what cannot be trusted (closed accounts from balances, retail "salary")
  4. Emits a per-customer quality report so the bank sees exactly what was trusted
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.deterministic import data_fingerprint

# ── Lexicons for text-based sentiment repair ───────────────────────────────────
NEGATIVE_TERMS = {
    "couldn't", "could not", "not competitive", "too complicated", "long wait", "needs improvement",
    "frustrated", "angry", "disappointed", "bad", "worst", "delay", "error", "failed", "wrong",
    "complaint", "refuse", "poor", "terrible", "unacceptable", "never", "useless", "awful",
    "overcharged", "hidden fee", "rude", "slow", "confusing", "broken",
}
POSITIVE_TERMS = {
    "professional", "courteous", "quick", "hassle-free", "peace of mind", "helpful", "resolved",
    "happy", "satisfied", "great", "excellent", "good", "love", "thanks", "easy", "appreciate",
    "perfect", "wonderful", "smooth", "friendly", "efficient", "impressed", "recommend", "fast",
}

# Merchants that cannot plausibly pay a salary
RETAIL_MERCHANTS = {
    "costco", "uber", "shell gas", "cvs pharmacy", "target", "apple", "mcdonald's", "walmart",
    "amazon", "netflix", "starbucks", "doordash", "spotify", "home depot", "best buy",
    "whole foods", "instacart", "delta airlines", "hilton hotels", "venmo", "paypal",
}
# Merchants that are online-only and cannot be an ATM channel
ONLINE_ONLY = {"netflix", "amazon", "spotify", "paypal", "venmo", "doordash", "instacart"}

LIABILITY_TYPES = {"credit card", "auto loan", "mortgage", "personal loan", "business loan", "loan", "line of credit"}


@dataclass
class QualityIssue:
    field: str
    issue: str
    action: str          # "repaired" | "excluded" | "flagged"
    severity: str = "medium"  # low | medium | high


@dataclass
class QualityReport:
    customer_id: str
    issues: List[QualityIssue] = field(default_factory=list)
    trust_score: float = 1.0           # 0..1 — how much of the payload survived intact
    fields_repaired: int = 0
    records_excluded: int = 0

    def add(self, field_: str, issue: str, action: str, severity: str = "medium"):
        self.issues.append(QualityIssue(field_, issue, action, severity))
        if action == "repaired":
            self.fields_repaired += 1
        elif action == "excluded":
            self.records_excluded += 1

    def to_dict(self) -> Dict:
        return {
            "customer_id": self.customer_id,
            "trust_score": round(self.trust_score, 3),
            "fields_repaired": self.fields_repaired,
            "records_excluded": self.records_excluded,
            "issue_count": len(self.issues),
            "high_severity": sum(1 for i in self.issues if i.severity == "high"),
            "issues": [vars(i) for i in self.issues[:25]],  # cap payload size
        }


# ── Field-name normalisation ───────────────────────────────────────────────────
# The dashboard export uses short keys (fn/ln/inc/cs) while the raw dataset uses
# long keys. Agents should see ONE schema.

_PROFILE_ALIASES = {
    "fn": "first_name", "ln": "last_name", "inc": "annual_income", "cs": "credit_score",
    "seg": "segment", "occ": "occupation", "since": "onboarding_date", "ch": "preferred_channel",
    "active": "is_active", "ltv": "lifetime_value", "nps": "nps_score", "id": "customer_id",
    "dob": "date_of_birth", "rm": "relationship_manager", "risk": "risk_level",
}
_ACCOUNT_ALIASES = {"type": "account_type", "bal": "balance", "rate": "interest_rate", "opened": "opened_date", "id": "account_id"}
_TXN_ALIASES = {"amt": "amount", "cat": "category", "merch": "merchant", "ch": "channel", "flag": "is_flagged"}
_INT_ALIASES = {"ch": "channel", "ts": "timestamp", "dur": "duration_seconds", "res": "resolution_status", "sat": "satisfaction_score", "sent": "sentiment"}
_SENT_ALIASES = {"src": "source", "sent": "sentiment", "score": "sentiment_score", "txt": "feedback_text", "churn": "churn_risk", "tier": "loyalty_tier"}
_FRAUD_ALIASES = {"id": "alert_id", "type": "fraud_type", "sev": "severity", "conf": "confidence_score", "score": "risk_score", "date": "alert_date"}
_BUREAU_ALIASES = {"src": "bureau_source", "cs": "credit_score", "total": "total_accounts", "open": "open_accounts", "delinq": "delinquent_accounts",
                   "limit": "total_credit_limit", "util": "utilization_pct", "inq": "hard_inquiries_12m", "pay": "payment_history_pct", "dti": "debt_to_income", "oldest": "oldest_account_years"}
_PERSONA_ALIASES = {"type": "persona_type", "dig": "digital_affinity_score", "risk": "risk_appetite_score", "eng": "engagement_level_score", "fh": "financial_health_score",
                    "time": "preferred_contact_time", "freq": "communication_frequency", "triggers": "context_triggers"}
_NUDGE_ALIASES = {"type": "nudge_type", "action": "action_recommended", "ch": "delivery_channel", "conf": "confidence_score", "pri": "priority", "val": "expected_value"}


def _norm(d: Dict, aliases: Dict[str, str]) -> Dict:
    out = dict(d)
    for short, long in aliases.items():
        if short in out and long not in out:
            out[long] = out.pop(short)
    return out


def _text_sentiment(text: str) -> Optional[str]:
    t = (text or "").lower()
    neg = sum(1 for k in NEGATIVE_TERMS if k in t)
    pos = sum(1 for k in POSITIVE_TERMS if k in t)
    if neg == 0 and pos == 0:
        return None
    if neg > pos:
        return "Very Negative" if neg >= 2 else "Negative"
    if pos > neg:
        return "Very Positive" if pos >= 2 else "Positive"
    return "Neutral"


class DataQualityGate:
    """Call `.clean(raw)` → (cleaned_data, QualityReport). Results are memoised by payload fingerprint
    so the gateway and orchestrator never clean the same payload twice."""

    def __init__(self, bank_config: Optional[Dict] = None, cache_size: int = 512):
        self._cfg = bank_config
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._cache_size = cache_size

    def _params(self) -> Dict:
        if self._cfg is None:
            from core.agent_base import BANK_CONFIG  # late import: agent_base must not depend on this module
            self._cfg = BANK_CONFIG
        return self._cfg.get("quality_gate") or {}

    def clean(self, raw: Dict[str, Any]) -> tuple[Dict[str, Any], QualityReport]:
        fp = data_fingerprint(raw)
        hit = self._cache.get(fp)
        if hit:
            self._cache.move_to_end(fp)
            return hit
        out = self._clean(raw)
        self._cache[fp] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def _clean(self, raw: Dict[str, Any]) -> tuple[Dict[str, Any], QualityReport]:
        prm = self._params()
        closed_min = float(prm.get("closed_account_min_balance", 1000))
        cc_implausible = float(prm.get("cc_implausible_above", 100_000))
        cc_cap = float(prm.get("cc_cap", 50_000))
        p = _norm(raw.get("profile", raw.get("p", {})), _PROFILE_ALIASES)
        cid = str(p.get("customer_id", "UNKNOWN"))
        rep = QualityReport(customer_id=cid)
        checks = 0

        # Reference date for age/tenure checks: latest activity in the payload, never the wall clock
        dates = [str(t.get("date", t.get("transaction_date", "")))[:10] for t in raw.get("transactions", raw.get("txns", []))]
        dates += [str(i.get("timestamp", i.get("ts", "")))[:10] for i in raw.get("interactions", raw.get("ints", []))]
        dates += [str(s.get("date", ""))[:10] for s in raw.get("sentiment", raw.get("sents", []))]
        dates = [d for d in dates if len(d) == 10 and d[:2] in ("19", "20")]
        latest_activity = max(dates) if dates else None
        ref_year = int(latest_activity[:4]) if latest_activity else datetime.now().year

        # ── Profile ───────────────────────────────────────────────────────────
        checks += 4
        cs = p.get("credit_score")
        rl = p.get("risk_level")
        if cs is not None and rl:
            if rl == "Low" and cs < 650:
                rep.add("profile.risk_level", f"risk_level=Low contradicts credit_score={cs}", "repaired", "high")
                p["risk_level"] = "High" if cs < 600 else "Medium"
            elif rl == "Very High" and cs > 780:
                rep.add("profile.risk_level", f"risk_level=Very High contradicts credit_score={cs}", "repaired", "high")
                p["risk_level"] = "Low"
        try:
            yob = int(str(p.get("date_of_birth", "1980"))[:4])
            yon = int(str(p.get("onboarding_date", "2020"))[:4])
            if yon - yob < 18:
                rep.add("profile.onboarding_date", f"onboarded at age {yon - yob}", "flagged", "medium")
            if p.get("occupation") == "Retired" and (ref_year - yob) < 55:
                rep.add("profile.occupation", f"'Retired' at age {ref_year - yob}", "flagged", "low")
        except (ValueError, TypeError):
            pass
        seg = p.get("segment", "")
        inc = float(p.get("annual_income", 0) or 0)
        if seg == "Ultra High Net Worth" and inc < 100_000:
            rep.add("profile.segment", f"Ultra-HNW label with income ${inc:,.0f} — segment unreliable, agents use balances instead", "flagged", "high")
            p["segment_reliability"] = "low"
        else:
            p["segment_reliability"] = "ok"

        # ── Accounts ──────────────────────────────────────────────────────────
        accounts = [_norm(a, _ACCOUNT_ALIASES) for a in raw.get("accounts", [])]
        checks += len(accounts) * 2
        clean_accounts = []
        for a in accounts:
            atype = str(a.get("account_type", "")).lower()
            bal = float(a.get("balance", 0) or 0)
            status = str(a.get("status", "Active"))
            if status == "Closed" and abs(bal) > closed_min:
                rep.add(f"accounts.{a.get('account_id','?')}", f"Closed account still shows ${bal:,.0f}", "excluded", "high")
                continue
            # Liabilities must be negative. Dataset stores credit-card "balance" as positive.
            if atype in LIABILITY_TYPES and bal > 0:
                rep.add(f"accounts.{a.get('account_id','?')}", f"{a.get('account_type')} balance +${bal:,.0f} → treated as owed", "repaired", "medium")
                bal = -bal
            if atype == "credit card" and abs(bal) > cc_implausible:
                rep.add(f"accounts.{a.get('account_id','?')}", f"Credit-card balance {abs(bal):,.0f} implausible — capped at {cc_cap:,.0f}", "repaired", "high")
                bal = -cc_cap
            a["balance"] = bal
            a["is_liability"] = atype in LIABILITY_TYPES
            clean_accounts.append(a)

        # ── Transactions ──────────────────────────────────────────────────────
        txns = [_norm(t, _TXN_ALIASES) for t in raw.get("transactions", raw.get("txns", []))]
        checks += len(txns) * 3
        clean_txns = []
        for t in txns:
            merch = str(t.get("merchant", "")).lower()
            cat = str(t.get("category", ""))
            ch = str(t.get("channel", ""))
            amt = float(t.get("amount", 0) or 0)
            if cat == "Salary" and merch in RETAIL_MERCHANTS:
                rep.add(f"transactions.{t.get('transaction_id','?')}", f"'Salary' credited from {t.get('merchant')} — merchant unreliable, kept as generic income", "repaired", "medium")
                t["merchant"] = "Employer (unverified)"
            if cat == "Travel" and merch in {"doordash", "mcdonald's", "starbucks", "instacart"}:
                rep.add(f"transactions.{t.get('transaction_id','?')}", f"'Travel' at {t.get('merchant')} → recategorised Food & Dining", "repaired", "low")
                t["category"] = "Food & Dining"
            if ch == "ATM" and merch in ONLINE_ONLY:
                rep.add(f"transactions.{t.get('transaction_id','?')}", f"ATM channel for {t.get('merchant')} → Online", "repaired", "low")
                t["channel"] = "Online Banking"
            if cat == "Salary" and amt < 0:
                rep.add(f"transactions.{t.get('transaction_id','?')}", "Negative salary", "repaired", "medium")
                amt = abs(amt)
            t["amount"] = amt
            t["date"] = str(t.get("date", ""))[:10]
            t.setdefault("category", "Uncategorised")
            t.setdefault("merchant", "")
            t.setdefault("channel", "")
            clean_txns.append(t)

        # ── Bureau ────────────────────────────────────────────────────────────
        bur = _norm(raw.get("bureau", {}) or {}, _BUREAU_ALIASES)
        if bur:
            checks += 5
            total, open_ = bur.get("total_accounts"), bur.get("open_accounts")
            if total is not None and open_ is not None and open_ > total:
                rep.add("bureau.open_accounts", f"open={open_} > total={total}", "repaired", "high")
                bur["total_accounts"] = open_
            delinq = bur.get("delinquent_accounts", 0) or 0
            if open_ is not None and delinq > open_:
                rep.add("bureau.delinquent_accounts", f"delinquent={delinq} > open={open_}", "repaired", "high")
                bur["delinquent_accounts"] = open_
            lim, tb = bur.get("total_credit_limit"), bur.get("total_balance")
            if lim and tb is not None:
                implied = tb / lim * 100
                if abs(implied - float(bur.get("utilization_pct", implied))) > 15:
                    rep.add("bureau.utilization_pct", f"stated {bur.get('utilization_pct')}% vs implied {implied:.1f}% → recomputed", "repaired", "medium")
                    bur["utilization_pct"] = round(implied, 1)
            bcs = bur.get("credit_score")
            if bcs and bcs > 800 and delinq >= 3:
                rep.add("bureau.credit_score", f"score {bcs} with {delinq} delinquencies — contradictory; delinquencies given precedence", "flagged", "high")
                bur["score_reliability"] = "low"
            else:
                bur["score_reliability"] = "ok"
            if "debt_to_income" in bur and bur["debt_to_income"] is not None:
                bur["debt_to_income_ratio"] = float(bur["debt_to_income"]) / 100.0  # explicit unit

        # ── Sentiment ─────────────────────────────────────────────────────────
        sents = [_norm(s, _SENT_ALIASES) for s in raw.get("sentiment", raw.get("sents", []))]
        checks += len(sents) * 2
        clean_sents = []
        for s in sents:
            label = s.get("sentiment", "Neutral")
            text_label = _text_sentiment(s.get("feedback_text", ""))
            s["label_source"] = "dataset"
            if text_label:
                neg_l = label in ("Negative", "Very Negative"); neg_t = text_label in ("Negative", "Very Negative")
                pos_l = label in ("Positive", "Very Positive"); pos_t = text_label in ("Positive", "Very Positive")
                if (neg_l and pos_t) or (pos_l and neg_t):
                    rep.add("sentiment.label", f"'{label}' contradicts text \"{str(s.get('feedback_text',''))[:40]}…\" → using text", "repaired", "high")
                    s["sentiment"] = text_label
                    s["label_source"] = "text_analysis"
                    # Score must follow the corrected label
                    s["sentiment_score"] = 0.2 if neg_t else 0.8
            cr = s.get("churn_risk")
            if cr is not None and s["sentiment"] in ("Positive", "Very Positive") and cr > 0.6:
                rep.add("sentiment.churn_risk", f"positive feedback but churn_risk={cr:.2f} — down-weighted", "repaired", "medium")
                s["churn_risk"] = round(cr * 0.5, 3)
            clean_sents.append(s)

        # ── Fraud ─────────────────────────────────────────────────────────────
        fraud = [_norm(f, _FRAUD_ALIASES) for f in raw.get("fraud", [])]
        checks += len(fraud)
        for f in fraud:
            if f.get("status") == "False Positive" and float(f.get("confidence_score", 0) or 0) > 0.9:
                rep.add(f"fraud.{f.get('alert_id','?')}", f"model confidence {f.get('confidence_score')} on a false positive — detector miscalibrated", "flagged", "medium")

        # ── Others: normalise only ────────────────────────────────────────────
        ints = [_norm(i, _INT_ALIASES) for i in raw.get("interactions", raw.get("ints", []))]
        persona = _norm(raw.get("persona", {}) or {}, _PERSONA_ALIASES)
        nudges = [_norm(n, _NUDGE_ALIASES) for n in raw.get("nudges", [])]
        micro = raw.get("microsegment", raw.get("micro", {})) or {}
        if micro:
            checks += 1
            seg_name = str(micro.get("segment_name", micro.get("name", ""))).lower()
            summ = str(micro.get("genai_persona_summary", micro.get("summary", ""))).lower()
            if ("millennial" in seg_name and "retirement" in summ) or ("young famil" in seg_name and ("retirement" in summ or "entrepreneur" in summ)) \
               or ("empty nester" in seg_name and "young family" in summ) or ("home buyer" in seg_name and ("retirement" in summ or "high-value" in summ)):
                rep.add("microsegment.genai_persona_summary", "summary contradicts segment_name", "excluded", "medium")
                micro["genai_persona_summary"] = None

        # ── Derived, trustworthy aggregates ───────────────────────────────────
        assets = sum(a["balance"] for a in clean_accounts if not a["is_liability"] and a["balance"] > 0)
        liabilities = sum(-a["balance"] for a in clean_accounts if a["is_liability"] or a["balance"] < 0)
        spend_by_cat: Dict[str, float] = {}
        income_90d = 0.0
        for t in clean_txns:
            if t["amount"] < 0:
                spend_by_cat[t["category"]] = spend_by_cat.get(t["category"], 0) + abs(t["amount"])
            elif t.get("category") == "Salary":
                income_90d += t["amount"]

        computed = {
            "total_assets": round(assets, 2),
            "total_liabilities": round(liabilities, 2),
            "net_position": round(assets - liabilities, 2),
            "total_balance": round(assets - liabilities, 2),
            "active_accounts": len(clean_accounts),
            "spend_by_category": {k: round(v, 2) for k, v in spend_by_cat.items()},
            "total_debits": round(sum(spend_by_cat.values()), 2),
            "salary_credits": round(income_90d, 2),
            "avg_churn_risk": round(sum(s.get("churn_risk", 0.3) for s in clean_sents) / max(len(clean_sents), 1), 3) if clean_sents else None,
            "latest_activity_date": latest_activity,
        }

        # Trust score: fraction of checks that passed without needing repair/exclusion
        penal = sum({"low": 0.25, "medium": 0.6, "high": 1.0}[i.severity] for i in rep.issues)
        rep.trust_score = max(0.0, 1.0 - penal / max(checks, 1))

        cleaned = {
            "profile": p, "accounts": clean_accounts, "transactions": clean_txns,
            "bureau": bur, "sentiment": clean_sents, "fraud": fraud, "interactions": ints,
            "persona": persona, "nudges": nudges, "microsegment": micro,
            "life_events": raw.get("life_events", raw.get("events", [])),
            "mobile": raw.get("mobile", {}), "journeys": raw.get("journeys", []),
            "computed": computed,
            "_quality": rep.to_dict(),
        }
        return cleaned, rep


gate = DataQualityGate()
