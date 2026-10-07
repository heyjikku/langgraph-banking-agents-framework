"""Gateway tests through FastAPI's TestClient — auth, rate limit, data-backed endpoints, realtime baseline."""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DEV_MODE"] = ""
os.environ["API_KEYS"] = "test-key-1"
os.environ["RATE_LIMIT_RPM"] = "1000"
os.environ.pop("REDIS_URL", None)

from fastapi.testclient import TestClient  # noqa: E402
from api.gateway import app  # noqa: E402

H = {"X-API-Key": "test-key-1"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_open_and_reports_as_of(client):
    r = client.get("/health"); assert r.status_code == 200
    assert r.json()["customers_loaded"] == 500 and r.json()["as_of_date"] == "2025-12-31"


def test_api_requires_key(client):
    assert client.get("/api/v2/customers").status_code == 401
    assert client.get("/api/v2/customers", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v2/customers?limit=2", headers=H).status_code == 200


def test_customer_360_runs_from_service_data(client):
    r = client.post("/api/v2/customers/CUST-00007/360?refresh=1", headers=H); assert r.status_code == 200
    run = r.json()["run"]
    assert run["as_of_date"] == "2025-12-31" and run["synthesis"]["fraud_summary"]["recommendation"] == "Block"
    assert client.post("/api/v2/customers/NOPE/360", headers=H).status_code == 404
    assert client.post("/api/v2/customers/CUST-00007/360?task=bogus", headers=H).status_code == 422


def test_realtime_uses_customer_baseline_when_no_data_sent(client):
    body = {"customer_id": "CUST-00001", "transaction": {"transaction_id": "T1", "amount": 95, "merchant": "Costco", "channel": "POS Terminal", "hour_of_day": 14}}
    r = client.post("/api/v2/agents/fraud/realtime", json=body, headers=H); assert r.status_code == 200
    assert r.json()["baseline_quality"] in ("ok", "thin") and r.json()["decision"] in ("Allow", "Flag for Review", "Block")
    r2 = client.post("/api/v2/agents/fraud/realtime", json={**body, "customer_id": "NOPE"}, headers=H); assert r2.status_code == 404


def test_portfolio_ranked_and_bounded(client):
    r = client.post("/api/v2/portfolio?limit=12", headers=H); assert r.status_code == 200
    d = r.json(); rows = d["rows"]
    assert d["summary"]["customers"] == 12 and d["as_of_date"] == "2025-12-31"
    vals = [x["value_at_risk"] for x in rows]
    assert vals == sorted(vals, reverse=True)


def test_rate_limit_enforced(client, monkeypatch):
    import api.gateway as gw
    monkeypatch.setattr(gw, "RATE_LIMIT_RPM", 3)
    monkeypatch.setattr(gw, "API_KEYS", {"test-key-1", "burst-key"})
    gw._buckets.pop("burst-key", None)
    codes = [client.get("/api/v2/bank-config", headers={"X-API-Key": "burst-key"}).status_code for _ in range(6)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]


def test_dashboard_served_at_root(client):
    r = client.get("/"); assert r.status_code == 200 and "<div id=\"root\"></div>" in r.text
