# Analytical Audit — Banking Agentic AI Framework
**Audit 1 (2026-09-09):** all source files + input dataset (500 customers, 12,811 records) · 36 weaknesses found and fixed.
**Audit 2 (2026-09-17):** framework v2.1, data source, React front end, deployment · 12 further weaknesses found and fixed.
**Regression suite:** 43 tests (`pytest tests/ -q`).

## 1. Input data — what the audit found (and what the framework now does about it)

| # | Finding | Volume | Gate action |
|---|---------|--------|-------------|
| D1 | `risk_level` contradicts `credit_score` (Low with <650; Very High with >780) | 58 / 500 | **Repaired** from score |
| D2 | Ultra-HNW customers with income < $100K | 66 / 85 | **Flagged** `segment_reliability=low`; CLV/NBA weight balances over segment |
| D3 | "Salary" credited from Costco / Uber / Shell / Apple | 83 / 201 | **Repaired** merchant → "Employer (unverified)" |
| D4 | "Travel" category at DoorDash / McDonald's; "ATM" channel for Netflix / PayPal | 38 + 135 | **Repaired** category / channel |
| D5 | Credit-card balances stored positive, 69 of them > $150K | 156 | **Repaired** sign; capped at $50K when > $100K |
| D6 | Closed accounts still holding > $1K | 297 / 298 | **Excluded** from balances |
| D7 | Bureau `open_accounts > total_accounts`; `delinquent > open`; `utilization ≠ balance/limit` | 130 / 33 / 395 | **Repaired** |
| D8 | Bureau score > 800 with ≥ 3 delinquencies | 32 | **Flagged** `score_reliability=low` — PD floor applied |
| D9 | Sentiment label contradicts feedback text (only 12 unique texts) | 385 / 1511 (25%) | **Repaired** from text; positive-but-churn>0.6 down-weighted |
| D10 | Microsegment name contradicts GenAI persona summary | 56 / 500 | **Excluded** summary |

Every pipeline response carries `data_quality.trust_score`, `fields_repaired`, `records_excluded` and the issue list, so an RM can see what the model did and didn't trust.

## 2. Code — weaknesses and fixes

| # | Area | Weakness | Fix |
|---|------|----------|-----|
| 1-2 | All agents | `random.uniform` confidence → identical input, different output | Confidence = data completeness × trust; `core/deterministic.py` replaces every `random.*` |
| 3 | Fraud | 3 resolved false-positives scored 66 → "Flag for Review" | Status weights (Open 1.0 … False Positive 0.05); now 4.3 → Allow |
| 4 | Fraud | `FRAUD_TYPE_WEIGHTS` dead code | Applied in alert weighting |
| 5 | Fraud | ±5 random noise on anomaly score → decision could flip on retry | Removed |
| 6 | Credit | PD 14% + 3 delinquencies → $172K approved (2.6× income) | Hard gates from config (`credit_hard_decline_pd`, `credit_hard_decline_delinquencies`, max DTI) → Decline / $0; grade-based income-multiple caps |
| 7 | CLV | HNW with $607K deposits scored $1,200 (BG/NBD on 1 txn) | Hybrid: relationship-value model (deposit + lending yield + fees) below `min_txns_for_bgnbd`; known LTV blended 30% |
| 8 | CLV | Wallet share = balance ÷ income×1.8 → 80% cap for everyone | Deposits ÷ segment investable-asset estimate |
| 9 | NBA | revenue / urgency / channel were `random.choice` | Revenue from `product_revenue_annual` × income tier; urgency from churn prior; channel from persona |
| 10 | NBA | Never read `results_so_far` — Phase-2 dependency was fictional | Reads `ctx.prior("churn")`, `ctx.prior("clv")`; exposed in `cross_agent_inputs` |
| 11 | Credit | Same — "reads fraud signals" was false | Reads `ctx.prior("fraud")`; open alerts raise PD, Block → Decline |
| 12 | Sentiment | Lexicons defined, never used; "FinBERT" copied dataset labels | Every text scored; label/text agreement % reported |
| 13 | Sentiment | "Negative" with CSAT 5.0/5 unremarked | Four-source reconciliation with `sources_agree` + `contradiction_note` |
| 14 | Orchestrator | "Claude plans" was a dict lookup | `task="auto"` → Claude JSON plan, validated; signal-driven heuristic offline |
| 15 | Orchestrator | Health score padded defaults (CSAT=3, CLV=60) when agents disabled | Weights renormalised over agents that ran; exposed in `health_weights_applied` |
| 16 | Orchestrator | Cache key ignored data → stale for 5 min after data changed | Key includes payload fingerprint |
| 17 | Config | `BANK_PROFILE` env var ignored | `load_bank_config()` honours env |
| 18 | Gateway | `/chat` built history then discarded it; borrowed a Churn agent for LLM | Transcript passed to Claude; prior agent results injected; Redis-backed history |
| 19 | Gateway | "Rate limiting" claimed, none implemented | Per-key token bucket (`RATE_LIMIT_RPM`) |
| 20 | Gateway | No authentication — fraud decisions and PII open | `X-API-Key` on every `/api/*` route; refuses to boot without keys outside DEV_MODE |
| 21 | Gateway | CORS `*` + credentials | Explicit `ALLOWED_ORIGINS` |
| 22 | NBA | india_nbfc got US products (checking, credit_card) | Per-profile `nba_segment_map` ⊆ `products` (tested) |
| 23 | Config | "Ultra High Net Worth" → `ultra_high_net_worth` never matched `ultra_hnw` | `SEGMENT_ALIASES` table |
| 24 | Concurrency | Global `random.seed` under `asyncio.gather` → cross-agent contamination | No global RNG |
| 25 | Orchestrator | `max_parallel_agents` never enforced | `asyncio.Semaphore` |
| 26 | Memory | Expired entries never evicted | Sweep + evict-on-read |
| 27 | Memory | Chat history in-process with 2 uvicorn workers → lost randomly | Redis-backed; warns if `WORKERS>1` without Redis |
| 28 | Deploy | `nginx.conf` referenced, didn't exist | Added |
| 29-33 | Requirements | 5 unused packages (sklearn, numpy, httpx …) | Trimmed to 6 |
| 34 | Gateway | `/fraud/realtime` accepted `hour_of_day`, `is_international` then discarded them | In-flight transaction passed to agent; z-score, off-hours, intl, velocity, first-seen merchant, MCC signals; extreme anomaly hard-blocks |
| INT | Integration | React dashboard still called `/api/customers` | `client/agentsApi.js` adapter + `toDashboardML()` mapper |
| 36 | Sentiment | Found in portfolio view: `requires_escalation` fired for 21 of 40 customers because the dataset marks ~50% of interactions Pending at random | Escalation now needs negative sentiment alongside repeated unresolved disputes; open-case backlog reported separately (3 of 40 escalate) |
| 35 | NBA | Found during dashboard review: Investment Advisory pitched to a customer under fraud Block, credit Decline, $0 deposits | Cross-sell suppressed under fraud Block (top action = clear the hold); no Loans/Cards for Decline; asset floors for investment products; `credit_risk` now runs before `nba` |

## 3. Before / after on real customers

| Customer | Before | After |
|----------|--------|-------|
| Sandra King — 2 **open** Critical alerts (SIM swap, AML) | Fraud 37 → Allow | **Block 85.2**; Credit Decline (CCC) |
| Andrew King — 3 delinquencies, PD 14% | $172K eligible | **Decline, $0** |
| Andrew King — $379K live deposits, 1 txn | CLV 3Y $1,785 | **$24,499** (relationship model) |
| 3 resolved false-positive alerts | 66 → Flag | **4.3 → Allow** |
| $9.5K, 03:00, international, 6/hr, gift cards vs $100 baseline | 37 → Allow | **Block 75** (7 signals) |

## 4. Run the regression suite
```
DEV_MODE=1 pytest tests/ -q        # 29 passed
```

## 5. Second audit (2026-09-17) — findings and fixes

| # | Area | Weakness | Evidence | Fix |
|---|------|----------|----------|-----|
| 37 | Time reference | Agents measured recency, tenure and alert age against the wall clock, but the dataset ends 2025-12-31 | "Interaction recency" pinned at its +25 cap for **60/60** customers → churn's top driver carried no information; 5 of 40 "high churn" were artefacts | `as_of` date resolved per run: explicit → profile `as_of_date` → dataset's latest activity → today. Exposed as `run.as_of_date`. Recency now varies (median 28 d, p90 246 d); churn median 17%, 7/200 above 60% |
| 38 | Currency | USD constants hard-coded in agents and the quality gate (product fee $95, card cap $50K, advisory income floor $40K, thin-baseline $10K) | `india_nbfc` would score ₹95 fees and cap cards at ₹50K | New per-profile sections `quality_gate:`, `clv:`, `nba:`, `fraud:` with INR values for the NBFC; agents read them via `self.param()` (tested) |
| 39 | React UI | `Customer.js` dereferenced every agent's output unconditionally | `community_bank` (CLV + sentiment off) → `TypeError`, white screen | Tiles render "Not run"; insights degrade to "—"; tabs show which agents are off; `ErrorBoundary` around the main pane |
| 40 | Portfolio | Ranking by `clv_3y × churn` with CLV disabled → every row 0, order arbitrary | — | `value_at_risk` is `null` without CLV; ranking falls back to churn probability; `clv_enabled` flag drives the UI label |
| 41 | Realtime | UI sent `customer_data: {}` → the "baseline" was empty; every merchant was "first-seen" | earlier signal list showed no z-score for any transaction | Gateway loads the customer's bundle from the data source when none is supplied; response reports `baseline_quality` |
| 42 | Security | `.env` (API keys, Anthropic key) was copied into the Docker image by `COPY . .` | — | `.dockerignore` excludes `.env`, logs, node_modules, build |
| 43 | Security | API-key comparison was a set lookup (timing side-channel); `?key=` in the fallback dashboard URL leaked keys into browser history | — | `hmac.compare_digest`; URL key parameter removed |
| 44 | Rate limit | Per-process token bucket with 2 workers → effective limit 2× and reset on restart | — | Redis fixed-window counter shared across workers when Redis is present; in-process fallback |
| 45 | Concurrency | `/portfolio` gathered up to 500 pipelines with no bound; data tables loaded lazily on first request | — | `PORTFOLIO_CONCURRENCY` semaphore (default 16); tables loaded in the app lifespan |
| 46 | Sentiment | Lexicon had no negation: "not helpful" scored positive, "never had a problem" negative | `_score_text('not good at all') = 0.70` | 3-token negation window; multi-word phrases keep their own polarity (tested) |
| 47 | Build | `npm install` without lockfile in the image (non-reproducible) | — | `npm ci` with the committed `package-lock.json` |
| 48 | Quality gate | Cleaned the same payload twice per request (gateway + orchestrator) | — | Gate memoises by payload fingerprint (LRU 512) |
| — | Validation | `customer_360` accepted any `task` string | — | Regex-validated query parameter |
| — | Tests | No tests hit the HTTP layer | — | `tests/test_api.py` via FastAPI TestClient: auth, rate limit, 404s, portfolio ordering, realtime baseline, dashboard at `/` |

**Data note.** After fixing the time reference, the mid-scale profile shows 0 of 40 customers above the 60% churn threshold and 22 credit declines. Both are properties of this dataset (activity is dense up to its 2025-12-31 horizon; a third of bureau rows carry ≥3 delinquencies). They are not tuned away — the thresholds are the bank's policy in `bank_config.yaml`.
