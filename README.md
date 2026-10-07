# Banking Agentic AI Framework v2.2

Quality-gated, deterministic, config-driven multi-agent Customer 360 for mid-scale banks.
**One container + Redis. `docker-compose up --build` — live in ~90 s.**

## What changed in v2.1 (post-audit)
An adversarial review of every file and the input dataset found **34 weaknesses** — all fixed.
Full findings, fixes and before/after numbers: [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md) — two audits, 48 findings.
Regression suite: `pip install -r requirements-dev.txt && pytest tests/ -q`.

Headline changes:
- **Data Quality Gate** (`core/data_quality.py`) runs before any agent — repairs, excludes and flags contradictory records; every response carries a `data_quality` report with a `trust_score`.
- **Deterministic agents** — no `random`; same input → same decision. Confidence = data completeness × trust.
- **Fraud** discounts resolved/false-positive alerts, uses fraud-type weights, and scores the actual in-flight transaction.
- **Credit** has hard decline gates and grade-based exposure caps from bank config; consumes Fraud output.
- **CLV** uses a relationship-value model when transaction history is thin; blends known LTV.
- **NBA** consumes Churn + CLV output, uses per-profile product maps and a revenue table.
- **LangGraph orchestration** — the six existing agents are graph nodes; independent scoring runs concurrently, followed by Credit Risk and NBA dependencies. The existing result cache remains separate; durable graph checkpoints are not enabled.
- **Security** — API-key auth on every `/api/*` route, token-bucket rate limiting, explicit CORS.
- **Deploy** — `BANK_PROFILE` env var actually switches profiles; Redis-backed chat memory for multi-worker.

## Quick start
```bash
cp .env.example .env               # set API_KEYS (required) and optionally ANTHROPIC_API_KEY
./deploy.sh mid_scale_bank         # or community_bank | india_nbfc
open http://localhost:8000/        # dashboard — every number is a live agent execution
```
Local dev without keys: `DEV_MODE=1 uvicorn api.gateway:app --reload` then open http://localhost:8000/

## Front end — React 17 (JavaScript)
`client/react-app/` is a Create-React-App project (react 17.0.2, react-dom 17.0.2, react-scripts 5, plain JSX, no TypeScript).
It holds **no customer data**: every number is fetched from the running service.

```
src/
  index.js                 entry
  App.js                   boot: /health → /api/v2/customers → /api/v2/portfolio; API-key prompt on 401
  services/api.js          all service calls (X-API-Key header, REACT_APP_AGENTS_URL base)
  lib/format.js            money / percent / tone helpers
  components/
    CustomerRail.js        portfolio + customer list with health dots
    Portfolio.js           value at risk, revenue opportunity, ranked "act on these first", by-segment
    Customer.js            hero, six verdict tiles, insight row, tabs; "Re-run all agents" (refresh=1)
    Tile.js Insight.js Bar.js
    Overview.js            churn attributions, next-best actions, balances, spend, data trusted
    Risk.js                fraud verdict + alert weights, credit gates, PD drivers, scorecard
    Authorise.js           live in-flight transaction scoring via /api/v2/agents/fraud/realtime
    Value.js               CLV horizons, model used, sentiment sources, interactions
    DataQuality.js         every repair / exclusion / flag
  styles.css
```

**Develop** (service on :8000, UI on :3000 with the CRA proxy — no CORS setup needed):
```bash
DEV_MODE=1 uvicorn api.gateway:app --reload          # terminal 1
cd client/react-app && npm install && npm start      # terminal 2 → http://localhost:3000
```
**Ship**: `npm run build` produces `client/react-app/build/`; the gateway serves it at `/` automatically
(the Docker image builds it in a node stage, so `./deploy.sh` needs no local Node). Without a build,
the gateway falls back to the single-file `client/c360_dashboard.html`.
Config: `.env.example` → `REACT_APP_AGENTS_URL` (leave empty when served by the gateway or using the dev proxy)
and `REACT_APP_API_KEY` (or enter the key in the prompt the app shows on 401).

The service owns the customer data (`core/data_source.py`, tables under `DATA_DIR`, default `./data`).
Replace that module with a warehouse/CDP adapter to go to production.

## Endpoints (all under `X-API-Key`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v2/portfolio?limit=` | Execute all agents across the portfolio; value at risk, revenue opportunity, holds |
| GET / POST | `/api/v2/customers` · `/api/v2/customers/{id}/360?refresh=` | List customers · execute the pipeline for one customer from the service's data |
| POST | `/api/v2/pipeline` | Full 360 on a caller-supplied payload · tasks: `full_360` `churn_only` `fraud_only` `nba_only` `risk_only` `custom` `auto` (Claude plans) |
| POST | `/api/v2/agents/{churn,fraud,nba,credit-risk,clv,sentiment}` | Single agent |
| POST | `/api/v2/agents/fraud/realtime` | Authorise an in-flight transaction |
| POST | `/api/v2/data-quality` | Preview what the gate would repair/exclude |
| POST | `/api/v2/chat` | BankAI, history + prior agent results in context |
| GET  | `/api/v2/bank-config` · `/agents` · `/health` | Introspection |

## Layout
```
core/       agent_base.py  data_quality.py  data_source.py  deterministic.py  memory.py
agents/     churn  fraud  nba  credit_risk  clv  sentiment  orchestrator
api/        gateway.py
config/     bank_config.yaml     (3 profiles; thresholds, gates, products, NBA maps, revenue table)
client/     react-app/           (React 17 JavaScript project — the dashboard)
            c360_dashboard.html  (single-file fallback, same UI, no build step)
            agentsApi.js         (JS adapter if embedding the agents in another React app)
data/       bank export tables (JSON) read by core/data_source.py
tests/      test_audit_regressions.py
docs/       AUDIT_REPORT.md
deploy/     Dockerfile  nginx.conf
```

## Time reference
Every recency, tenure and alert-age calculation is measured from `run.as_of_date`, never the wall clock:
explicit API argument → profile `as_of_date` in `bank_config.yaml` → latest activity date in the data → today.
For a live feed leave `as_of_date: null`; for a historical extract it resolves to the extract's own horizon.

## Adapting to a new bank
Add a profile in `config/bank_config.yaml`: enable/disable agents, set thresholds and credit gates,
list `products`, give each segment an `nba_segment_map` (must be ⊆ products — enforced by test), a
`product_revenue_annual` table, and the currency-dependent `quality_gate:`, `clv:`, `nba:` and `fraud:` parameters
(see `india_nbfc` for INR values). Set `BANK_PROFILE=<name>`. No code changes.
