"""
Sentiment Agent (v2.1, post-audit)

Fixes:
  • Analyses `feedback_text` with a finance-domain lexicon instead of copying the dataset label
    (the quality gate already flips 25% of labels that contradict their text; this agent
    re-scores every text and reports label/text agreement).
  • Reconciles three sources — feedback records, interaction sentiment, CSAT — and reports
    a contradiction flag when they disagree (previously "Negative" with CSAT 5.0/5 went unremarked).
  • Deterministic.
"""
from __future__ import annotations
from typing import Any, Dict, List
from core.agent_base import BaseAgent, AgentContext, AgentTool
from core.data_quality import NEGATIVE_TERMS, POSITIVE_TERMS
from core.deterministic import clamp


class SentimentAgent(BaseAgent):
    agent_name = "sentiment_agent"
    agent_description = "Lexicon NLP on feedback + reconciliation across interaction sentiment and CSAT"
    version = "2.1.0"
    required_inputs = ["sentiment", "interactions", "profile.nps_score"]

    LABEL_TO_SCORE = {"Very Positive": 0.9, "Positive": 0.72, "Neutral": 0.5, "Negative": 0.28, "Very Negative": 0.1}
    ESCALATION_INTENTS = {"complaint", "dispute", "escalat", "fraud", "closure", "cancel"}

    def build_tools(self) -> List[AgentTool]:
        return [
            AgentTool("score_texts", "Lexicon scoring of every feedback text", self._tool_texts),
            AgentTool("interaction_signal", "Sentiment + CSAT + resolution from interactions", self._tool_interactions),
            AgentTool("reconcile", "Blend sources; flag contradictions", self._tool_reconcile),
            AgentTool("intents", "Intent breakdown and escalation triggers", self._tool_intents),
        ]

    async def execute(self, ctx: AgentContext) -> Dict[str, Any]:
        d = ctx.customer_data
        p = d["profile"]
        texts = await self.use_tool("score_texts", sentiment=d.get("sentiment", []))
        inter = await self.use_tool("interaction_signal", interactions=d.get("interactions", []))
        rec = await self.use_tool("reconcile", texts=texts, inter=inter, nps=float(p.get("nps_score", 7) or 7))
        intents = await self.use_tool("intents", interactions=d.get("interactions", []))

        esc_thresh = self.get_threshold("sentiment_escalation_score", 0.30)
        # Escalate = a human is needed now: clearly negative sentiment, or repeated dispute-type contact
        # that is still unresolved AND souring. An open-case backlog alone is reported separately.
        blended, open_ = rec["blended_score"], inter["open_count"]
        requires_escalation = (blended <= esc_thresh) \
            or (intents["escalation_intents"] >= 2 and open_ >= 2 and blended < 0.45) \
            or (open_ >= 3 and blended < 0.40)
        open_case_backlog = open_ >= 2

        digest = await self.llm_reason(
            f"Blended sentiment {rec['overall']} ({rec['blended_score']:.2f}), CSAT {inter['avg_csat'] if inter['avg_csat'] is not None else 'n/a'}/5, "
            f"sources agree: {rec['sources_agree']}. Top intent {intents['top_intent']}. One-sentence CX action.",
            system="You are a CX analyst. One empathetic, specific sentence.")

        return {
            "overall_sentiment": rec["overall"],
            "sentiment_score": rec["blended_score"],
            "source_scores": rec["sources"],
            "sources_agree": rec["sources_agree"],
            "contradiction_note": rec["note"],
            "label_text_agreement_pct": texts["agreement_pct"],
            "text_scored": texts["scored"],
            "sentiment_distribution": texts["distribution"],
            "avg_churn_risk_from_sentiment": texts["avg_churn_risk"],
            "avg_csat": inter["avg_csat"], "resolution_rate": inter["resolution_rate"],
            "open_interactions": inter["open_count"], "avg_handle_time_mins": inter["avg_handle_mins"],
            "interaction_intents": intents["counts"], "top_intent": intents["top_intent"],
            "escalation_intents": intents["escalation_intents"],
            "nps_band": "Promoter" if float(p.get("nps_score", 7) or 7) >= 9 else "Passive" if float(p.get("nps_score", 7) or 7) >= 7 else "Detractor",
            "requires_escalation": requires_escalation,
            "open_case_backlog": open_case_backlog,
            "llm_cx_recommendation": digest,
            "model": "Finance lexicon + source reconciliation v2.1",
        }

    # ── Tools ──────────────────────────────────────────────────────────────

    NEGATORS = {"not", "no", "never", "n't", "cannot", "hardly", "without", "isn't", "wasn't", "don't", "didn't", "couldn't", "won't"}

    def _score_text(self, text: str) -> float:
        """Lexicon polarity with a 3-token negation window: 'not helpful' counts as negative,
        'never had a problem' does not count as negative."""
        import re
        t = (text or "").lower().replace("n't", " n't")
        tokens = re.findall(r"[a-z']+", t)
        pos = neg = 0
        # multi-word phrases first (they carry their own polarity)
        for phrase in NEGATIVE_TERMS:
            if " " in phrase and phrase in t: neg += 1
        for phrase in POSITIVE_TERMS:
            if " " in phrase and phrase in t: pos += 1
        single_neg = {k for k in NEGATIVE_TERMS if " " not in k} - self.NEGATORS
        single_pos = {k for k in POSITIVE_TERMS if " " not in k}
        for i, tok in enumerate(tokens):
            negated = any(tokens[j] in self.NEGATORS for j in range(max(0, i - 3), i))
            if tok in single_pos:
                if negated: neg += 1
                else: pos += 1
            elif tok in single_neg:
                if negated: pos += 0.5   # "no problem" — weak positive
                else: neg += 1
        if neg == pos == 0: return 0.5
        return clamp(0.5 + 0.2 * (pos - neg), 0.05, 0.95)

    def _tool_texts(self, sentiment: list) -> Dict:
        scored, dist, agree, churn = [], {}, 0, []
        for s in sentiment:
            txt = s.get("feedback_text", "")
            ts = self._score_text(txt)
            label = s.get("sentiment", "Neutral")
            ls = self.LABEL_TO_SCORE.get(label, 0.5)
            label_from_text = "Very Positive" if ts >= 0.85 else "Positive" if ts > 0.6 else "Negative" if ts < 0.4 else "Very Negative" if ts <= 0.15 else "Neutral"
            agrees = (ts > 0.6) == (ls > 0.6) and (ts < 0.4) == (ls < 0.4)
            agree += agrees
            dist[label_from_text] = dist.get(label_from_text, 0) + 1
            if s.get("churn_risk") is not None: churn.append(float(s["churn_risk"]))
            scored.append({"text": str(txt)[:90], "text_score": round(ts, 2), "dataset_label": label,
                           "label_source": s.get("label_source", "dataset"), "agrees": agrees, "topic": s.get("topic"), "date": s.get("date")})
        n = len(sentiment)
        return {"scored": scored, "distribution": dist, "avg_text_score": round(sum(x["text_score"] for x in scored) / n, 3) if n else None,
                "agreement_pct": round(agree / n * 100, 0) if n else None,
                "avg_churn_risk": round(sum(churn) / len(churn), 3) if churn else None}

    def _tool_interactions(self, interactions: list) -> Dict:
        n = len(interactions)
        if not n:
            return {"avg_sentiment_score": None, "avg_csat": None, "resolution_rate": None, "open_count": 0, "avg_handle_mins": None}
        ss = [self.LABEL_TO_SCORE.get(i.get("sentiment", "Neutral"), 0.5) for i in interactions]
        cs = [float(i.get("satisfaction_score", 3) or 3) for i in interactions]
        resolved = sum(1 for i in interactions if str(i.get("resolution_status", "")) == "Resolved")
        open_ = sum(1 for i in interactions if str(i.get("resolution_status", "")) in ("Pending", "Escalated"))
        mins = [float(i.get("duration_seconds", 300) or 300) / 60 for i in interactions]
        return {"avg_sentiment_score": round(sum(ss) / n, 3), "avg_csat": round(sum(cs) / n, 2),
                "resolution_rate": round(resolved / n, 3), "open_count": open_, "avg_handle_mins": round(sum(mins) / n, 1)}

    def _tool_reconcile(self, texts: Dict, inter: Dict, nps: float) -> Dict:
        parts = []
        if texts["avg_text_score"] is not None: parts.append(("feedback_text", texts["avg_text_score"], 0.45))
        if inter["avg_sentiment_score"] is not None: parts.append(("interaction_sentiment", inter["avg_sentiment_score"], 0.25))
        if inter["avg_csat"] is not None: parts.append(("csat", (inter["avg_csat"] - 1) / 4, 0.20))
        parts.append(("nps", nps / 10, 0.10))
        w = sum(x[2] for x in parts)
        blended = round(sum(v * wt for _, v, wt in parts) / w, 3)
        vals = [v for _, v, _ in parts]
        spread = max(vals) - min(vals)
        agree = spread < 0.35
        note = None
        if not agree:
            hi = max(parts, key=lambda x: x[1]); lo = min(parts, key=lambda x: x[1])
            note = f"{hi[0]} reads {hi[1]:.2f} but {lo[0]} reads {lo[1]:.2f} — verify before acting"
        overall = "Positive" if blended >= 0.62 else "Negative" if blended <= 0.38 else "Neutral"
        return {"blended_score": blended, "overall": overall, "sources": {n: round(v, 3) for n, v, _ in parts},
                "sources_agree": agree, "note": note}

    def _tool_intents(self, interactions: list) -> Dict:
        counts: Dict[str, int] = {}
        for i in interactions:
            k = str(i.get("intent", "General")); counts[k] = counts.get(k, 0) + 1
        esc = sum(v for k, v in counts.items() if any(e in k.lower() for e in self.ESCALATION_INTENTS))
        top = max(counts, key=counts.get) if counts else None
        return {"counts": counts, "top_intent": top, "escalation_intents": esc}

    def template_narrative(self, prompt: str) -> str:
        return "Sentiment blended from text, interactions, CSAT and NPS; check contradiction_note before acting."
