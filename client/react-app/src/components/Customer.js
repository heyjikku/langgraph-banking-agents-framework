import React, { useCallback, useEffect, useState } from 'react';
import Tile from './Tile';
import Insight from './Insight';
import Overview from './Overview';
import Risk from './Risk';
import Value from './Value';
import DataQuality from './DataQuality';
import { api } from '../services/api';
import { money, pct, TONE, healthColor } from '../lib/format';

const TABS = ['Overview', 'Risk', 'Value', 'Data quality'];

/** A tile for an agent the bank profile has switched off (or that failed this run). */
const off = (agent, why) => ({ agent, tone: 'slate', decision: 'Not run', value: why, reason: '', confidence: 1 });

/** Derives the six verdict tiles from the agent outputs; any agent may be absent. */
function buildTiles(res, failed) {
  const { churn, fraud, credit_risk: cr, clv, nba, sentiment: sent } = res;
  const why = (k) => (failed[k] ? `failed: ${failed[k]}` : 'not enabled in this profile');
  const top = nba && nba.top_action;
  return [
    churn ? { agent: 'Churn', tone: TONE[churn.risk_level], decision: churn.risk_level, value: `${pct(churn.churn_probability)} in 90 days`,
      reason: `Driven by ${churn.top_drivers.slice(0, 2).join(' and ').toLowerCase()}`, confidence: churn.confidence } : off('Churn', why('churn')),
    fraud ? { agent: 'Fraud', tone: TONE[fraud.recommendation], decision: fraud.recommendation, value: `score ${fraud.composite_risk_score}`,
      reason: fraud.open_fraud_alerts ? `${fraud.open_fraud_alerts} open alert${fraud.open_fraud_alerts > 1 ? 's' : ''}, ${fraud.rules_fired.length} rule${fraud.rules_fired.length === 1 ? '' : 's'} fired`
        : fraud.total_fraud_alerts ? `${fraud.total_fraud_alerts} alerts, all resolved` : 'No alerts on file', confidence: fraud.confidence } : off('Fraud', why('fraud')),
    cr ? { agent: 'Credit', tone: TONE[cr.risk_appetite], decision: cr.risk_appetite, value: `${cr.internal_grade} · PD ${pct(cr.probability_of_default, 1)}`,
      reason: cr.decline_reasons.length ? cr.decline_reasons[0] : `Up to ${money(cr.max_eligible_loan)} at ${cr.recommended_rate}%`, confidence: cr.confidence } : off('Credit', why('credit_risk')),
    clv ? { agent: 'Lifetime value', tone: 'bank', decision: clv.tier, value: `${money(clv.clv_3_year)} over 3 years`,
      reason: `Wallet share ${clv.wallet_share_pct}% · ${clv.model_selection.primary_model.replace('_', ' ')} model`, confidence: clv.confidence } : off('Lifetime value', why('clv')),
    nba ? { agent: 'Next best action', tone: nba.suppressed ? 'red' : 'bank', decision: top ? top.product_name : 'None',
      value: top && !nba.suppressed ? `${pct(top.propensity_score)} propensity · ${money(top.expected_revenue)}/yr` : nba.suppressed ? 'cross-sell suppressed' : 'no eligible product',
      reason: top ? `${top.urgency} urgency via ${top.delivery_channel}` : 'holding, score, income or asset filters', confidence: nba.confidence } : off('Next best action', why('nba')),
    sent ? { agent: 'Sentiment', tone: sent.overall_sentiment === 'Negative' ? 'red' : sent.overall_sentiment === 'Positive' ? 'green' : 'slate', decision: sent.overall_sentiment,
      value: `${sent.sentiment_score.toFixed(2)} blended`, reason: sent.sources_agree ? 'Feedback, interactions, CSAT and NPS agree' : 'Sources disagree — verify before acting', confidence: sent.confidence } : off('Sentiment', why('sentiment')),
  ];
}

/** The single most important action, in priority order: fraud hold → high churn → fraud review → CX escalation → credit decline → offer. */
function mostImportantAction(res, valueAtRisk) {
  const { churn = {}, fraud = {}, credit_risk: cr = {}, nba = {}, sentiment: sent = {} } = res;
  const top = nba.top_action;
  if (nba.suppressed) return { value: 'Clear the fraud hold', basis: `${fraud.open_fraud_alerts} open alert(s) block every offer` };
  if (churn.risk_level === 'High') return { value: churn.recommended_action.action, basis: `${churn.recommended_action.owner} within ${churn.recommended_action.urgency.toLowerCase()}${valueAtRisk != null ? ` · ${money(valueAtRisk)} at stake` : ''}` };
  if (fraud.recommendation === 'Flag for Review') return { value: 'Resolve the open fraud alert', basis: `score ${fraud.composite_risk_score} · ${fraud.open_fraud_alerts} open` };
  if (sent.requires_escalation) return { value: 'Resolve the open complaint', basis: `${sent.open_interactions} unresolved · sentiment ${sent.sentiment_score.toFixed(2)}` };
  if (cr.risk_appetite === 'Decline') return { value: 'Review credit exposure', basis: cr.decline_reasons[0] };
  if (top) return { value: `Offer ${top.product_name}`, basis: `${top.urgency} urgency via ${top.delivery_channel} · ${money(top.expected_revenue)}/yr` };
  return { value: 'Standard nurture', basis: 'no active risk or eligible offer' };
}

const NotRun = ({ agents }) => <div className="empty">The {agents} agents are not enabled in this bank profile.</div>;

export default function Customer({ id, onStatus, onAuthError }) {
  const [data, setData] = useState(null);
  const [tab, setTab] = useState('Overview');
  const [error, setError] = useState('');

  const load = useCallback(async (refresh = false) => {
    setError('');
    onStatus('Executing agents…');
    try {
      const d = await api.customer360(id, refresh);
      setData(d);
      onStatus(`Run ${d.run.run_id} · ${d.run.total_latency_ms} ms · ${d.run.agents_succeeded.length} agents${d.run.cache_hit ? ' · cached' : ''}`);
    } catch (e) {
      if (e.auth) { onAuthError(); return; }
      setError(e.message);
      onStatus('Error');
    }
  }, [id, onStatus, onAuthError]);

  useEffect(() => { setData(null); setTab('Overview'); load(false); }, [id, load]);

  if (error) return <div className="loading t-red">{error}</div>;
  if (!data) return <div className="loading">Executing agents for {id}…</div>;

  const { customer: c, run: r } = data;
  const res = r.results;
  const s = r.synthesis;
  const tiles = buildTiles(res, r.agents_failed || {});
  const valueAtRisk = res.clv && res.churn ? res.clv.clv_3_year * res.churn.churn_probability : null;
  const opportunity = !res.nba ? null : res.nba.suppressed ? 0 : res.nba.all_actions.reduce((sum, a) => sum + a.expected_revenue, 0);
  const action = mostImportantAction(res, valueAtRisk);
  const has = (k) => Boolean(res[k]);

  return (
    <>
      <div className="hero">
        <div>
          <h1>{c.name}</h1>
          <div className="sub">
            {c.segment}{c.segment_reliability === 'low' ? ' (label unreliable)' : ''} · {c.occupation} · {c.city}, {c.state} · since {c.since}{c.is_active ? '' : ' · inactive'} · {c.relationship_manager}
          </div>
          <div className="facts">
            <div>Income<b>{money(c.annual_income)}</b></div>
            <div>Credit score<b>{c.credit_score}</b></div>
            <div>NPS<b>{c.nps} / 10</b></div>
            <div>Net position<b style={{ color: c.net < 0 ? 'var(--red)' : 'inherit' }}>{money(c.net)}</b></div>
            <div>Prefers<b>{c.preferred_channel}</b></div>
          </div>
        </div>
        <div className="health">
          <div className="n" style={{ color: healthColor(s.health_score) }}>{s.health_score == null ? '—' : s.health_score.toFixed(0)}</div>
          <div className="g">{s.health_grade} · relationship health</div>
          <div className="w">as of {r.as_of_date}</div>
          <button className="link" onClick={() => load(true)}>Re-run all agents</button>
        </div>
      </div>

      <div className="strip">{tiles.map((t) => <Tile key={t.agent} {...t} />)}</div>

      <div className="insights three">
        <Insight label="Lifetime value at risk" value={valueAtRisk == null ? '—' : money(valueAtRisk)} tone={valueAtRisk > 10000 ? 'var(--red)' : 'inherit'}
          basis={valueAtRisk == null ? 'needs the CLV and churn agents' : `${money(res.clv.clv_3_year)} 3-year CLV × ${pct(res.churn.churn_probability)} churn`} />
        <Insight label="Revenue opportunity" value={opportunity == null ? '—' : money(opportunity)} tone={opportunity > 0 ? 'var(--green)' : 'inherit'}
          basis={!res.nba ? 'NBA agent not enabled' : res.nba.suppressed ? 'none while fraud hold is active' : res.nba.all_actions.map((a) => a.product_name).join(', ') || 'no eligible products'} />
        <Insight label="Single most important action" value={action.value} basis={action.basis} />
      </div>

      <div className="alerts">{s.risk_alerts.map((a, i) => <span key={i} className={`chip ${a.severity}`}>{a.text}</span>)}</div>
      {s.executive_narrative && !s.executive_narrative.startsWith('Executive narrative unavailable') && <p className="narrative">{s.executive_narrative}</p>}

      <div className="tabs" role="tablist">
        {TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t}</button>)}
      </div>
      {tab === 'Overview' && (has('churn') || has('nba') ? <Overview customer={c} run={r} /> : <NotRun agents="churn and next-best-action" />)}
      {tab === 'Risk' && (has('fraud') || has('credit_risk') ? <Risk customer={c} run={r} /> : <NotRun agents="fraud and credit" />)}
      {tab === 'Value' && (has('clv') || has('sentiment') ? <Value customer={c} run={r} /> : <NotRun agents="CLV and sentiment" />)}
      {tab === 'Data quality' && <DataQuality run={r} />}
    </>
  );
}
