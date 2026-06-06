import json
import logging
import re
import time

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fintech AI Agent API",
    description="Fraud detection, AML compliance Q&A, and risk report generation powered by Claude.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──────────────────────────────────────────────────────────────────
START_TIME = time.time()

metrics = {
    "total_requests": 0,
    "fraud_checks": 0,
    "compliance_queries": 0,
    "risk_reports": 0,
    "high_risk_alerts": 0,
    "hallucinations_caught": 0,
    "response_times_ms": [],
}

# ── Claude client ──────────────────────────────────────────────────────────
client = Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a senior fintech compliance AI. "
    "Be precise. Cite specific regulations. "
    "Never hallucinate. Return valid JSON only."
)

# ── Regulations ────────────────────────────────────────────────────────────
REGULATIONS = {
    "KYC": (
        "Know Your Customer: verify identity, govt ID, "
        "proof of address, source of funds >$10K, EDD for PEPs. "
        "[FinCEN CDD Rule, FATF]"
    ),
    "AML": (
        "Anti-Money Laundering: file SARs within 30 days, "
        "CTRs for cash >$10K, structuring is illegal. "
        "[Bank Secrecy Act, USA PATRIOT Act]"
    ),
    "GDPR": (
        "Data protection: consent required, 72hr breach "
        "notification, right to erasure. [EU Regulation 2016/679]"
    ),
    "SOX": (
        "Section 302: CEO/CFO certify financials. "
        "Section 404: internal controls audit. "
        "Records retained 7 years. [Pub.L. 107-204]"
    ),
    "PCI-DSS": (
        "Never store CVV, encrypt cardholder data "
        "TLS 1.2+, AES-256, quarterly scans, annual pen test. "
        "[PCI SSC v4.0]"
    ),
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _call_claude(prompt: str) -> dict:
    raw = (
        client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        .content[0]
        .text
    )
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
    return json.loads(cleaned)


def _track(
    endpoint_counter: str,
    elapsed_ms: int,
    high_risk: bool = False,
    hallucination: bool = False,
):
    metrics["total_requests"] += 1
    metrics[endpoint_counter] += 1
    metrics["response_times_ms"].append(elapsed_ms)
    if high_risk:
        metrics["high_risk_alerts"] += 1
    if hallucination:
        metrics["hallucinations_caught"] += 1
    log.info("endpoint=%s response_time_ms=%d", endpoint_counter, elapsed_ms)


# ── Schemas ────────────────────────────────────────────────────────────────


class FraudRequest(BaseModel):
    transaction: str
    threshold: int = 7


class FraudResponse(BaseModel):
    risk_score: int
    risk_level: str
    red_flags: list[str]
    recommendation: str
    reasoning: str
    faithfulness_score: float
    confidence: float
    processing_time_ms: int


class ComplianceRequest(BaseModel):
    question: str


class ComplianceResponse(BaseModel):
    answer: str
    source: str
    confidence: float
    hallucination_risk: str


class RiskReportRequest(BaseModel):
    customer_name: str
    account_type: str
    transactions: str
    country: str
    income_range: str


class RiskReportSection(BaseModel):
    customer_profile: str
    transaction_analysis: str
    red_flags: list[str]
    regulatory_considerations: str
    recommended_actions: list[str]
    compliance_notes: str


class RiskReportResponse(BaseModel):
    risk_level: str
    risk_score: int
    report: RiskReportSection
    faithfulness_score: float
    hallucination_risk: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    uptime_seconds: int


class MetricsResponse(BaseModel):
    total_requests: int
    fraud_checks: int
    compliance_queries: int
    risk_reports: int
    high_risk_alerts: int
    hallucinations_caught: int
    avg_response_time_ms: float


class RegulationsResponse(BaseModel):
    regulations: list[str]
    count: int
    last_updated: str


# ── Lifecycle ──────────────────────────────────────────────────────────────


@app.on_event("startup")
async def on_startup():
    log.info("Fintech AI Agent API starting — model=%s", MODEL)


@app.on_event("shutdown")
async def on_shutdown():
    log.info("Shutting down — total_requests=%d", metrics["total_requests"])


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.post("/api/fraud/detect", response_model=FraudResponse)
async def fraud_detect(req: FraudRequest):
    t0 = time.time()
    prompt = f"""Analyze this financial transaction for fraud risk.

Transaction: {req.transaction}

Respond with ONLY valid JSON:
{{
  "risk_score": <int 0-10>,
  "red_flags": ["flag1"],
  "recommendation": "approve|review|reject",
  "reasoning": "<explanation>",
  "faithfulness_score": <float 0-1>,
  "confidence": <float 0-1>
}}"""

    try:
        data = _call_claude(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude error: {e}")

    score = int(data.get("risk_score", 0))
    level = "HIGH" if score >= 7 else "MEDIUM" if score >= 4 else "LOW"
    elapsed = int((time.time() - t0) * 1000)

    _track("fraud_checks", elapsed, high_risk=score >= req.threshold)

    return FraudResponse(
        risk_score=score,
        risk_level=level,
        red_flags=data.get("red_flags", []),
        recommendation=data.get("recommendation", "review"),
        reasoning=data.get("reasoning", ""),
        faithfulness_score=float(data.get("faithfulness_score", 0.8)),
        confidence=float(data.get("confidence", 0.8)),
        processing_time_ms=elapsed,
    )


@app.post("/api/compliance/query", response_model=ComplianceResponse)
async def compliance_query(req: ComplianceRequest):
    t0 = time.time()
    context = "\n\n".join(f"[{k}]\n{v}" for k, v in REGULATIONS.items())
    prompt = f"""Use ONLY the regulatory context below to answer the question.

REGULATORY CONTEXT:
{context}

QUESTION: {req.question}

Respond with ONLY valid JSON:
{{
  "answer": "<detailed answer citing regulation>",
  "source": "<KYC|AML|GDPR|SOX|PCI-DSS>",
  "confidence": <float 0-1>,
  "hallucination_risk": "LOW|MEDIUM|HIGH"
}}"""

    try:
        data = _call_claude(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude error: {e}")

    hal = data.get("hallucination_risk", "MEDIUM").upper()
    elapsed = int((time.time() - t0) * 1000)
    _track("compliance_queries", elapsed, hallucination=hal == "HIGH")

    return ComplianceResponse(
        answer=data.get("answer", ""),
        source=data.get("source", "Unknown"),
        confidence=float(data.get("confidence", 0.7)),
        hallucination_risk=hal,
    )


@app.post("/api/risk/report", response_model=RiskReportResponse)
async def risk_report(req: RiskReportRequest):
    t0 = time.time()
    prompt = f"""Generate a formal AML risk assessment report.

CUSTOMER PROFILE:
- Name: {req.customer_name}
- Account Type: {req.account_type}
- Country: {req.country}
- Monthly Income: {req.income_range}

TRANSACTIONS:
{req.transactions}

Respond with ONLY valid JSON:
{{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "risk_score": <int 0-100>,
  "report": {{
    "customer_profile": "<text>",
    "transaction_analysis": "<text>",
    "red_flags": ["flag1"],
    "regulatory_considerations": "<text>",
    "recommended_actions": ["action1"],
    "compliance_notes": "<text>"
  }},
  "faithfulness_score": <float 0-1>,
  "hallucination_risk": "LOW|MEDIUM|HIGH"
}}"""

    try:
        data = _call_claude(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude error: {e}")

    level = data.get("risk_level", "MEDIUM").upper()
    hal = data.get("hallucination_risk", "MEDIUM").upper()
    elapsed = int((time.time() - t0) * 1000)
    _track(
        "risk_reports",
        elapsed,
        high_risk=level in ("HIGH", "CRITICAL"),
        hallucination=hal == "HIGH",
    )

    raw_report = data.get("report", {})
    return RiskReportResponse(
        risk_level=level,
        risk_score=int(data.get("risk_score", 50)),
        report=RiskReportSection(
            customer_profile=raw_report.get("customer_profile", ""),
            transaction_analysis=raw_report.get("transaction_analysis", ""),
            red_flags=raw_report.get("red_flags", []),
            regulatory_considerations=raw_report.get("regulatory_considerations", ""),
            recommended_actions=raw_report.get("recommended_actions", []),
            compliance_notes=raw_report.get("compliance_notes", ""),
        ),
        faithfulness_score=float(data.get("faithfulness_score", 0.8)),
        hallucination_risk=hal,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model=MODEL,
        version="1.0.0",
        uptime_seconds=int(time.time() - START_TIME),
    )


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    times = metrics["response_times_ms"]
    avg = sum(times) / len(times) if times else 0.0
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        fraud_checks=metrics["fraud_checks"],
        compliance_queries=metrics["compliance_queries"],
        risk_reports=metrics["risk_reports"],
        high_risk_alerts=metrics["high_risk_alerts"],
        hallucinations_caught=metrics["hallucinations_caught"],
        avg_response_time_ms=round(avg, 2),
    )


@app.get("/api/regulations", response_model=RegulationsResponse)
async def get_regulations():
    return RegulationsResponse(
        regulations=list(REGULATIONS.keys()),
        count=len(REGULATIONS),
        last_updated="2026-05-22",
    )


# ── EXAMPLE CURL COMMANDS ──────────────────────────────────────────────────
#
# Health:
# curl http://localhost:8000/api/health
#
# Fraud detect:
# curl -X POST http://localhost:8000/api/fraud/detect \
#   -H "Content-Type: application/json" \
#   -d '{"transaction":"Transfer $9800 to Cayman Islands at 3am"}'
#
# Compliance:
# curl -X POST http://localhost:8000/api/compliance/query \
#   -H "Content-Type: application/json" \
#   -d '{"question":"When must we file a SAR under BSA?"}'
#
# Risk report:
# curl -X POST http://localhost:8000/api/risk/report \
#   -H "Content-Type: application/json" \
#   -d '{"customer_name":"John Smith","account_type":"Personal",
#        "transactions":"$9800 cash deposit\n$9700 wire to Mexico",
#        "country":"USA","income_range":"$5K-$25K"}'
#
# Swagger UI: http://localhost:8000/docs
# ReDoc:      http://localhost:8000/redoc
