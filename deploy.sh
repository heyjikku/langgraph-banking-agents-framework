#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Banking Agentic AI Framework — One-Command Deploy
# Usage:  ./deploy.sh [mid_scale_bank|community_bank|india_nbfc]
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

PROFILE="${1:-mid_scale_bank}"
echo "🏦 Deploying Banking Agentic AI Framework"
echo "   Bank profile: $PROFILE"
echo ""

# docker compose v2 ("docker compose") or v1 ("docker-compose")
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi

# Env setup
[ ! -f .env ] && cp .env.example .env && echo "⚙️  Created .env from template — edit API_KEYS before exposing this beyond localhost"
export BANK_PROFILE="$PROFILE"

# Build and launch (the Dockerfile compiles the React UI in a node stage)
echo "🐳 Building containers..."
$DC build

echo "🚀 Starting services..."
$DC up -d

echo ""
echo "⏳ Waiting for health check..."
sleep 8

if curl -sf http://localhost:8000/health > /dev/null; then
    echo "✅ Framework is LIVE"
    echo ""
    echo "  Dashboard:     http://localhost:8000/   (enter the API key from .env when prompted)"
    echo "  API Gateway:   http://localhost:8000"
    echo "  Swagger docs:  http://localhost:8000/docs"
    echo "  Health:        http://localhost:8000/health"
    echo "  Agents list:   http://localhost:8000/agents"
    echo "  Bank config:   http://localhost:8000/api/v2/bank-config"
    echo ""
    echo "  Quick test:"
    echo "  curl -X POST http://localhost:8000/api/v2/agents/churn \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo '    -d '"'"'{"customer_id":"C001","customer_data":{"profile":{"nps_score":4,"annual_income":85000}}}'"'"
else
    echo "❌ Health check failed. Check logs:"
    echo "   $DC logs banking-agents"
    exit 1
fi
