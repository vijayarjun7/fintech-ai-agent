import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")
os.environ.setdefault("PINECONE_API_KEY", "test_key")

from api_server import app  # noqa: E402

client = TestClient(app)


def test_import():
    """api_server imports without errors."""
    from api_server import app as _app  # noqa: F401
    assert _app is not None


def test_endpoints_registered():
    """All 6 endpoints are registered."""
    routes = {r.path for r in app.routes}
    expected = {
        "/api/health",
        "/api/metrics",
        "/api/regulations",
        "/api/fraud/detect",
        "/api/compliance/query",
        "/api/risk/report",
    }
    assert expected <= routes, f"Missing routes: {expected - routes}"


def test_health_schema():
    """GET /api/health returns correct schema."""
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["model"] == "claude-sonnet-4-20250514"
    assert body["version"] == "1.0.0"
    assert isinstance(body["uptime_seconds"], int)


def test_metrics_schema():
    """GET /api/metrics returns correct schema."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    body = res.json()
    expected_keys = {
        "total_requests",
        "fraud_checks",
        "compliance_queries",
        "risk_reports",
        "high_risk_alerts",
        "hallucinations_caught",
        "avg_response_time_ms",
    }
    assert expected_keys <= set(body.keys())


def test_regulations_schema():
    """GET /api/regulations returns all 5 regulations."""
    res = client.get("/api/regulations")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 5
    assert set(body["regulations"]) == {"KYC", "AML", "GDPR", "SOX", "PCI-DSS"}


def test_fraud_detect_missing_body():
    """POST /api/fraud/detect with empty body returns 422."""
    res = client.post("/api/fraud/detect", json={})
    assert res.status_code == 422


def test_compliance_query_missing_body():
    """POST /api/compliance/query with empty body returns 422."""
    res = client.post("/api/compliance/query", json={})
    assert res.status_code == 422


def test_risk_report_missing_body():
    """POST /api/risk/report with empty body returns 422."""
    res = client.post("/api/risk/report", json={})
    assert res.status_code == 422
