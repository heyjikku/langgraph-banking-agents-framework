/**
 * Banking Agentic AI — JavaScript client for the v2 API (React 17 compatible, no TypeScript).
 * Drop into banking-c360/frontend/src/services/ and import instead of the old api.js.
 *
 * Fixes audit finding #INTEGRATION: the dashboard was calling /api/customers & /api/ml/* which
 * the agentic backend does not expose. This adapter maps the dashboard's expectations onto
 * /api/v2/pipeline and normalises the response shape the panels already render.
 */
const API_BASE = process.env.REACT_APP_AGENTS_URL || 'http://localhost:8000';
const API_KEY  = process.env.REACT_APP_API_KEY   || '';

async function post(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
    body: JSON.stringify(body),
  });
  if (res.status === 401) throw new Error('Unauthorised — set REACT_APP_API_KEY');
  if (res.status === 429) throw new Error('Rate limited — retry shortly');
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

export const agentsAPI = {
  /** Full 360 for one customer. `customerData` is the raw bundle (profile, accounts, ...). */
  pipeline: (customerId, customerData, task = 'full_360') =>
    post('/api/v2/pipeline', { customer_id: customerId, customer_data: customerData, task }),

  /** Individual agents */
  churn:      (id, data) => post('/api/v2/agents/churn',       { customer_id: id, customer_data: data }),
  fraud:      (id, data) => post('/api/v2/agents/fraud',       { customer_id: id, customer_data: data }),
  nba:        (id, data) => post('/api/v2/agents/nba',         { customer_id: id, customer_data: data }),
  creditRisk: (id, data) => post('/api/v2/agents/credit-risk', { customer_id: id, customer_data: data }),
  clv:        (id, data) => post('/api/v2/agents/clv',         { customer_id: id, customer_data: data }),
  sentiment:  (id, data) => post('/api/v2/agents/sentiment',   { customer_id: id, customer_data: data }),

  /** Authorise an in-flight transaction (<50 ms target) */
  authorise: (id, transaction, data = {}) =>
    post('/api/v2/agents/fraud/realtime', { customer_id: id, transaction, customer_data: data }),

  /** Data-quality preview without scoring */
  dataQuality: (id, data) => post('/api/v2/data-quality', { customer_id: id, customer_data: data }),

  /** BankAI chat — history is kept server-side */
  chat: (id, message, data = null) => post('/api/v2/chat', { customer_id: id, message, customer_data: data }),
};

/**
 * Map a /api/v2/pipeline response onto the shape the existing dashboard panels expect
 * (ml.churn, ml.clv, ml.next_best_action, ml.credit_risk, ml.health_score ...).
 */
export function toDashboardML(pipeline) {
  const r = pipeline.results || {};
  const s = pipeline.synthesis || {};
  return {
    health_score: s.health_score,
    health_grade: s.health_grade,
    churn: r.churn && {
      churn_probability: r.churn.churn_probability,
      risk_level: r.churn.risk_level,
      risk_score: r.churn.risk_score,
      key_drivers: r.churn.top_drivers,
      shap_values: Object.fromEntries((r.churn.attributions || []).map(a => [a.feature, a.contribution])),
      recommended_action: r.churn.recommended_action && r.churn.recommended_action.action,
      model: r.churn.model,
      confidence: r.churn.confidence,
    },
    clv: r.clv && {
      clv_score: r.clv.clv_score, clv_1_year: r.clv.clv_1_year, clv_3_year: r.clv.clv_3_year, clv_5_year: r.clv.clv_5_year,
      segment_bucket: r.clv.tier, wallet_share_pct: r.clv.wallet_share_pct, revenue_potential: r.clv.revenue_potential,
      model: r.clv.model, confidence: r.clv.confidence,
    },
    next_best_action: r.nba && {
      top_action: r.nba.top_action && mapAction(r.nba.top_action),
      all_actions: (r.nba.all_actions || []).map(mapAction),
      personalization_score: Math.round((r.nba.confidence || 0.8) * 100),
      model: r.nba.model,
    },
    credit_risk: r.credit_risk && {
      probability_of_default: r.credit_risk.probability_of_default, internal_grade: r.credit_risk.internal_grade,
      credit_score: r.credit_risk.credit_score, bureau_band: r.credit_risk.bureau_band,
      loss_given_default: r.credit_risk.loss_given_default, exposure_at_default: r.credit_risk.exposure_at_default,
      expected_loss: r.credit_risk.expected_loss, debt_to_income: r.credit_risk.debt_to_income,
      max_eligible_loan: r.credit_risk.max_eligible_loan, recommended_rate: r.credit_risk.recommended_rate,
      risk_appetite: r.credit_risk.risk_appetite, decline_reasons: r.credit_risk.decline_reasons,
      scorecards: r.credit_risk.scorecard_breakdown, model: r.credit_risk.model,
    },
    fraud: r.fraud,
    sentiment: r.sentiment,
    segment: r.clv && { segment: r.clv.tier, strategy: r.clv.tier_strategy, model: 'CLV tier' },
    ai_summary: s.executive_narrative,
    risk_alerts: s.risk_alerts,
    data_quality: pipeline.data_quality,
  };
}

function mapAction(a) {
  return {
    action: a.product_name, category: a.category, propensity_score: a.propensity_score,
    expected_revenue: a.expected_revenue, channel: a.delivery_channel, urgency: a.urgency,
    offer: a.llm_pitch || a.rationale, icon: { Deposits: '💰', Loans: '🏠', Cards: '💳', Insurance: '🛡️', Investments: '📈' }[a.category] || '📋',
  };
}
