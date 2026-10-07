"""
Regression suite — every audit finding from the 2026-09-09 review is a test here.
Run:  DEV_MODE=1 pytest tests/ -q
If any of these fail again, a previously-fixed weakness has regressed.
"""
import inspect, os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DEV_MODE", "1")

from core.agent_base import AgentContext, load_bank_config
from core.data_quality import gate
from agents.churn.agent import ChurnRiskAgent
from agents.fraud.agent import FraudDetectionAgent
from agents.nba.agent import NextBestActionAgent
from agents.credit_risk.agent import CreditRiskAgent
from agents.clv.agent import CLVAgent
from agents.sentiment.agent import SentimentAgent
from agents.orchestrator.agent import OrchestratorAgent

BASE = {'profile': {'customer_id': 'X', 'annual_income': 80000, 'credit_score': 720, 'nps_score': 7, 'segment': 'Mass Retail',
                    'is_active': True, 'onboarding_date': '2020-01-01', 'date_of_birth': '1985-01-01'},
        'accounts': [{'account_type': 'Savings', 'balance': 10000, 'status': 'Active'}],
        'transactions': [], 'interactions': [], 'sentiment': [], 'bureau': {}, 'persona': {}, 'fraud': [], 'nudges': []}


def ctx(raw, prior=None, meta=None):
    cleaned, _ = gate.clean(raw)
    c = AgentContext(run_id='t', customer_id=raw['profile'].get('customer_id', 'X'), customer_data=cleaned, metadata=meta or {})
    if prior: c.results_so_far = prior
    return c


# ── Determinism (findings 1, 2, 5, 24) ─────────────────────────────────────────
@pytest.mark.parametrize("cls", [ChurnRiskAgent, FraudDetectionAgent, NextBestActionAgent, CreditRiskAgent, CLVAgent, SentimentAgent])
@pytest.mark.asyncio
async def test_agents_are_deterministic(cls):
    a, b = await cls().run(ctx(BASE)), await cls().run(ctx(BASE))
    assert a.success and a.output == b.output


def test_no_random_module_in_agents():
    for a in ['churn', 'fraud', 'nba', 'clv', 'credit_risk', 'sentiment']:
        src = open(f'agents/{a}/agent.py').read()
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", "•", "*")) and '"""' not in l)
        assert 'import random' not in code and 'random.' not in code, a


# ── Fraud (findings 3, 4, 34) ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_resolved_false_positives_do_not_flag():
    raw = {**BASE, 'transactions': [{'amount': -100, 'date': '2026-08-01', 'category': 'Groceries'}] * 10,
           'fraud': [{'severity': 'Critical', 'status': 'False Positive', 'fraud_type': 'Identity Theft', 'confidence_score': 0.9}] * 3}
    o = (await FraudDetectionAgent().run(ctx(raw))).output
    assert o['recommendation'] == 'Allow' and o['composite_risk_score'] < 30


@pytest.mark.asyncio
async def test_open_critical_alert_blocks():
    raw = {**BASE, 'transactions': [{'amount': -100, 'date': '2026-08-01', 'category': 'Groceries', 'is_flagged': True}],
           'fraud': [{'severity': 'Critical', 'status': 'Open', 'fraud_type': 'Account Takeover', 'confidence_score': 0.9}]}
    assert (await FraudDetectionAgent().run(ctx(raw))).output['recommendation'] == 'Block'


def test_fraud_type_weights_are_used():
    assert 'FRAUD_TYPE_WEIGHTS.get' in inspect.getsource(FraudDetectionAgent)


@pytest.mark.asyncio
async def test_inflight_transaction_is_evaluated():
    hist = {**BASE, 'transactions': [{'amount': -a, 'date': f'2026-08-0{i % 9 + 1}', 'merchant': 'Costco', 'channel': 'POS Terminal', 'category': 'Groceries'}
                                     for i, a in enumerate([120, 80, 95, 110, 70, 130, 90, 100, 85, 105])]}
    bad = (await FraudDetectionAgent().run(ctx(hist, meta={'inflight_transaction': {'amount': 9500, 'hour_of_day': 3, 'is_international': True,
                                                                                    'merchant': 'x', 'channel': 'Online', 'transaction_count_1h': 6, 'merchant_category': 'Gift Cards'}}))).output
    ok = (await FraudDetectionAgent().run(ctx(hist, meta={'inflight_transaction': {'amount': 95, 'hour_of_day': 14, 'merchant': 'Costco', 'channel': 'POS Terminal'}}))).output
    assert bad['recommendation'] == 'Block' and ok['recommendation'] == 'Allow'


# ── Credit (findings 6, 11) ────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_hard_decline_on_delinquencies():
    raw = {**BASE, 'profile': {**BASE['profile'], 'annual_income': 66000, 'credit_score': 659},
           'bureau': {'debt_to_income': 42.0, 'delinquent_accounts': 3, 'utilization_pct': 62.7, 'total_credit_limit': 191627, 'total_balance': 120000,
                      'payment_history_pct': 94.5, 'total_accounts': 10, 'open_accounts': 10}}
    o = (await CreditRiskAgent().run(ctx(raw))).output
    assert o['risk_appetite'] == 'Decline' and o['max_eligible_loan'] == 0


@pytest.mark.asyncio
async def test_loan_capped_by_grade_income_multiple():
    raw = {**BASE, 'profile': {**BASE['profile'], 'annual_income': 90000, 'credit_score': 790},
           'bureau': {'debt_to_income': 18.0, 'delinquent_accounts': 0, 'utilization_pct': 20, 'total_credit_limit': 60000, 'total_balance': 12000,
                      'payment_history_pct': 100, 'total_accounts': 6, 'open_accounts': 4, 'oldest_account_years': 12}}
    o = (await CreditRiskAgent().run(ctx(raw))).output
    assert o['risk_appetite'] == 'Within Limit' and o['max_eligible_loan'] <= 90000 * 5.0


@pytest.mark.asyncio
async def test_credit_consumes_fraud_prior():
    raw = {**BASE, 'profile': {**BASE['profile'], 'credit_score': 790}, 'bureau': {'debt_to_income': 18.0, 'delinquent_accounts': 0, 'utilization_pct': 20, 'payment_history_pct': 100}}
    o = (await CreditRiskAgent().run(ctx(raw, prior={'fraud': {'recommendation': 'Block', 'open_fraud_alerts': 2, 'composite_risk_score': 85}}))).output
    assert o['risk_appetite'] == 'Decline' and o['fraud_adjustment_applied']


# ── CLV (findings 7, 8) ────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_clv_uses_balances_when_history_thin():
    raw = {**BASE, 'profile': {**BASE['profile'], 'segment': 'High Net Worth', 'onboarding_date': '2017-01-01', 'lifetime_value': 283681},
           'accounts': [{'account_type': 'Savings', 'balance': 165000, 'status': 'Active'}, {'account_type': 'Money Market', 'balance': 214000, 'status': 'Active'}],
           'transactions': [{'amount': -2500, 'date': '2026-05-01', 'category': 'Shopping'}]}
    o = (await CLVAgent().run(ctx(raw))).output
    assert o['clv_3_year'] > 20000 and o['model_selection']['primary_model'] == 'relationship_value'
    assert 0 < o['wallet_share_pct'] < 95


# ── NBA (findings 9, 10, 22, 23, F) ────────────────────────────────────────────
@pytest.mark.asyncio
async def test_nba_reads_churn_prior_and_uses_config_revenue():
    o = (await NextBestActionAgent().run(ctx(BASE, prior={'churn': {'risk_level': 'High', 'churn_probability': 0.7}, 'clv': {'tier': 'Gold'}}))).output
    t = o['top_action']
    assert t['urgency'] == 'High' and t['delivery_channel'] == 'RM Call' and t['expected_revenue'] % 1 == 0 and t['rationale']


@pytest.mark.asyncio
async def test_nba_held_product_filter():
    raw = {**BASE, 'accounts': [{'account_type': 'Credit Card', 'balance': 5000, 'status': 'Active'}]}
    o = (await NextBestActionAgent().run(ctx(raw))).output
    assert not any('credit_card' in a['product_id'] for a in o['all_actions'])


def test_nba_maps_subset_of_products_for_every_profile():
    for prof in ['mid_scale_bank', 'community_bank', 'india_nbfc']:
        cfg, _, _ = load_bank_config(prof)
        for seg, prods in cfg['nba_segment_map'].items():
            assert set(prods) <= set(cfg['products']), (prof, seg)


def test_segment_alias_resolves_ultra_hnw():
    assert NextBestActionAgent().get_segment_config('Ultra High Net Worth')['clv_multiplier'] == 5.0


# ── Sentiment (findings 12, 13) ────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sentiment_analyses_text_and_reconciles():
    raw = {**BASE, 'profile': {**BASE['profile'], 'nps_score': 2},
           'interactions': [{'satisfaction_score': 5, 'sentiment': 'Neutral', 'resolution_status': 'Pending', 'intent': 'Transaction Dispute', 'duration_seconds': 600}],
           'sentiment': [{'feedback_text': 'Loan approval was quick and hassle-free.', 'sentiment': 'Very Negative', 'sentiment_score': 0.05, 'churn_risk': 0.84}]}
    o = (await SentimentAgent().run(ctx(raw))).output
    assert o['text_scored'][0]['label_source'] == 'text_analysis' and o['text_scored'][0]['text_score'] > 0.6
    assert o['sources_agree'] is False and o['contradiction_note']


# ── Orchestrator (findings 14, 15, 16, 25) ─────────────────────────────────────
@pytest.mark.asyncio
async def test_health_weights_renormalise_and_cache_keyed_on_data():
    o = OrchestratorAgent()
    a = await o.run_pipeline('X', BASE, task='full_360', use_cache=False)
    assert abs(sum(a['synthesis']['health_weights_applied'].values()) - 1) < 1e-6
    b = await o.run_pipeline('X', {**BASE, 'profile': {**BASE['profile'], 'nps_score': 1}}, task='full_360', use_cache=True)
    assert not b['cache_hit'] and a['data_fingerprint'] != b['data_fingerprint']
    assert 'trust_score' in a['data_quality']


def test_orchestrator_has_llm_planning_and_semaphore():
    src = inspect.getsource(OrchestratorAgent)
    assert '_llm_plan' in src and 'json.loads' in src and 'Semaphore' in src


# ── Data quality gate ──────────────────────────────────────────────────────────
def test_gate_repairs_known_dataset_defects():
    raw = {'profile': {'customer_id': 'Q', 'credit_score': 848, 'risk_level': 'Very High', 'annual_income': 50000, 'segment': 'Ultra High Net Worth'},
           'accounts': [{'account_id': 'A', 'account_type': 'Credit Card', 'balance': 214035, 'status': 'Active'},
                        {'account_id': 'B', 'account_type': 'Savings', 'balance': 227705, 'status': 'Closed'}],
           'transactions': [{'transaction_id': 'T', 'category': 'Salary', 'merchant': 'Costco', 'amount': 8720, 'channel': 'ATM', 'date': '2025-01-01'}],
           'bureau': {'total_accounts': 4, 'open_accounts': 7, 'delinquent_accounts': 3, 'total_credit_limit': 100000, 'total_balance': 50000, 'utilization_pct': 12, 'credit_score': 848},
           'sentiment': [{'feedback_text': 'The chatbot couldn\'t understand my query.', 'sentiment': 'Very Positive', 'sentiment_score': 0.9, 'churn_risk': 0.1}]}
    cleaned, rep = gate.clean(raw)
    assert cleaned['profile']['risk_level'] == 'Low'                       # contradiction with 848 fixed
    assert cleaned['profile']['segment_reliability'] == 'low'
    assert len(cleaned['accounts']) == 1 and cleaned['accounts'][0]['balance'] == -50000   # closed excluded, CC negative + capped
    assert cleaned['bureau']['total_accounts'] == 7 and cleaned['bureau']['utilization_pct'] == 50.0
    assert cleaned['sentiment'][0]['sentiment'] in ('Negative', 'Very Negative') and cleaned['sentiment'][0]['label_source'] == 'text_analysis'
    assert cleaned['transactions'][0]['merchant'] == 'Employer (unverified)'
    assert rep.trust_score < 1.0 and rep.fields_repaired >= 5 and rep.records_excluded == 1


# ── Gateway / infra (findings 17–21, 26–33) ────────────────────────────────────
def test_gateway_security_controls_present():
    g = open('api/gateway.py').read()
    assert g.count('Depends(require_key)') >= 10 and '_buckets' in g and 'allow_origins=["*"]' not in g
    assert 'transcript' in g and 'conversation_memory.add' in g and 'model_dump' in g


def test_config_and_infra():
    assert 'os.getenv("BANK_PROFILE")' in open('core/agent_base.py').read()
    m = open('core/memory.py').read(); assert '_sweep' in m and 'chat:' in m
    assert os.path.exists('deploy/nginx.conf')
    rq = open('requirements.txt').read()
    assert all(x not in rq for x in ['scikit', 'numpy', 'asyncio-throttle', 'json-logger', 'httpx'])


# ── NBA suppression under fraud block / credit decline (found during dashboard review) ─────
@pytest.mark.asyncio
async def test_nba_suppresses_cross_sell_under_fraud_block():
    o = (await NextBestActionAgent().run(ctx(BASE, prior={'fraud': {'recommendation': 'Block', 'open_fraud_alerts': 2}}))).output
    assert o['suppressed'] and o['top_action']['category'] == 'Service' and o['eligible_count'] == 0


@pytest.mark.asyncio
async def test_nba_never_offers_credit_to_declined_customer_or_investments_without_assets():
    raw = {**BASE, 'profile': {**BASE['profile'], 'segment': 'Mass Affluent', 'annual_income': 150000, 'credit_score': 700},
           'accounts': [{'account_type': 'Checking', 'balance': 500, 'status': 'Active'}]}
    o = (await NextBestActionAgent().run(ctx(raw, prior={'credit_risk': {'risk_appetite': 'Decline'}}))).output
    cats = {a['category'] for a in o['all_actions']}
    ids = {a['product_id'] for a in o['all_actions']}
    assert 'Loans' not in cats and 'Cards' not in cats and 'investment_advisory' not in ids


def test_orchestrator_runs_credit_before_nba():
    from agents.orchestrator.agent import PHASE2, TASK_PLANS
    assert PHASE2.index('credit_risk') < PHASE2.index('nba')
    seq = TASK_PLANS['nba_only']['sequential']; assert seq.index('credit_risk') < seq.index('nba')


@pytest.mark.asyncio
async def test_escalation_requires_negative_sentiment_not_just_open_cases():
    """Dataset has ~50% random 'Pending' statuses; a backlog with positive sentiment must not escalate."""
    raw = {**BASE, 'profile': {**BASE['profile'], 'nps_score': 9},
           'interactions': [{'satisfaction_score': 5, 'sentiment': 'Positive', 'resolution_status': 'Pending', 'intent': 'Balance Inquiry'}] * 3,
           'sentiment': [{'feedback_text': 'Customer support was helpful and resolved my issue.', 'sentiment': 'Positive', 'sentiment_score': 0.8}]}
    o = (await SentimentAgent().run(ctx(raw))).output
    assert o['requires_escalation'] is False and o['open_case_backlog'] is True
    raw2 = {**BASE, 'profile': {**BASE['profile'], 'nps_score': 2},
            'interactions': [{'satisfaction_score': 1, 'sentiment': 'Very Negative', 'resolution_status': 'Escalated', 'intent': 'Complaint'}] * 3,
            'sentiment': [{'feedback_text': 'Long wait times at the branch, needs improvement.', 'sentiment': 'Negative', 'sentiment_score': 0.2}]}
    assert (await SentimentAgent().run(ctx(raw2))).output['requires_escalation'] is True


def test_gateway_serves_dashboard_root():
    g = open('api/gateway.py').read()
    assert '_REACT_BUILD' in g and '@app.get("/", include_in_schema=False)' in g
    assert os.path.exists('client/react-app/src/App.js') and os.path.exists('client/react-app/package.json')


# ── Second audit (2026-09-17): time reference, currency parameters, negation ──────────────────
@pytest.mark.asyncio
async def test_recency_measured_from_as_of_not_wall_clock():
    """Dataset ends 2025-12-31. Measured against the wall clock every customer looked ≥300 days stale."""
    raw = {**BASE, 'interactions': [{'timestamp': '2025-12-20', 'resolution_status': 'Resolved'}],
           'transactions': [{'amount': -50, 'date': '2025-12-28', 'category': 'Groceries'}]}
    o = (await ChurnRiskAgent().run(ctx(raw))).output
    assert o['rfm_scores']['as_of'] == '2025-12-28' and o['rfm_scores']['days_since_interaction'] == 0
    rec = next(a for a in o['attributions'] if a['feature'] == 'Interaction recency')
    assert rec['contribution'] == 0.0


def test_currency_parameters_come_from_profile():
    from core.agent_base import load_bank_config
    usd, _, _ = load_bank_config('mid_scale_bank'); inr, _, _ = load_bank_config('india_nbfc')
    assert inr['clv']['product_fee'] > 20 * usd['clv']['product_fee']
    assert inr['quality_gate']['cc_cap'] > 10 * usd['quality_gate']['cc_cap']
    assert 'mutual_fund_sip' in inr['nba']['min_income_for'] and 'investment_advisory' not in inr['nba']['min_income_for']
    for f in ['agents/clv/agent.py', 'agents/nba/agent.py', 'agents/fraud/agent.py']:
        assert 'self.param(' in open(f).read(), f


@pytest.mark.asyncio
async def test_sentiment_negation():
    a = SentimentAgent()
    assert a._score_text('the app is not helpful') < 0.5
    assert a._score_text('never had a problem, great service') > 0.5
    assert a._score_text('not good at all') < 0.5


def test_gate_memoises_by_fingerprint():
    a, _ = gate.clean(BASE); b, _ = gate.clean(dict(BASE))
    assert a is b


def test_docker_image_excludes_secrets():
    di = open('.dockerignore').read().split()
    assert '.env' in di and 'client/react-app/node_modules' in di
    assert 'npm ci' in open('deploy/Dockerfile').read()


def test_react_ui_tolerates_disabled_agents():
    c = open('client/react-app/src/components/Customer.js').read()
    assert "off('Lifetime value'" in c and 'NotRun' in c
    assert 'ErrorBoundary' in open('client/react-app/src/App.js').read()
    assert "qs.get('key')" not in open('client/c360_dashboard.html').read()
