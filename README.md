# Fintech AI Agent 🏦

> AI-powered fraud detection, compliance Q&A,
> and AML risk reports — with hallucination eval layer.
> Powered by Claude + Pinecone + LangSmith

![CI](https://github.com/vijayarjun7/fintech-ai-agent/actions/workflows/ci.yml/badge.svg)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204-orange?logo=anthropic)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green?logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Evidently](https://img.shields.io/badge/Evidently-Drift%20Monitor-purple)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)

## 🤗 Live Demo
https://huggingface.co/spaces/Vijayarv07/fintech-ai-agent

## GitHub
https://github.com/vijayarjun7/fintech-ai-agent

---

## What It Does

| Section | Function |
|---|---|
| 🛡️ Fraud Detector | Risk score 0-10, red flags, approve/review/reject |
| ⚖️ Compliance Q&A | RAG over KYC/AML/GDPR/SOX/PCI-DSS |
| 📋 Risk Report Generator | Formal 6-section AML assessment with regulatory citations |
| 📊 Eval Dashboard | Real-time quality score, hallucination flags, risk alerts |

---

## Key Results

**Live demo — 3 queries across all sections:**

| Section | Input | Result |
|---|---|---|
| 🛡️ Fraud Detector | `$9,800 to Cayman Islands at 3:47am` | 9/10 HIGH RISK → **REJECT** |
| ⚖️ Compliance Q&A | `When must we file a SAR under BSA?` | BSA + FATF cited, 95% confidence |
| 📋 Risk Report | Sarah Johnson, Personal, USA | 15/100 LOW — 6-section formal report |

**Eval scores:**

| Metric | Value |
|---|---|
| Avg quality score | 98% |
| Hallucinations flagged | 0 |
| Faithfulness — Fraud Detector | 95–100% |
| Faithfulness — Risk Report | 92% |
| Confidence — Compliance Q&A | 95–96% |
| Hallucination risk | LOW across all runs |

**MLflow tracked runs:**

| Run | Confidence | Risk Score | Response Time |
|---|---|---|---|
| FraudDetector | 85% | 9/10 | 1200 ms |
| ComplianceQA | 92% | — | 980 ms |
| RiskReportGenerator | — | 78/100 | 2100 ms |
| **Avg** | | | **1427 ms** |

---

## Stack

```
app.py                 Gradio UI (4 accordions, dark theme)
api_server.py          FastAPI REST API (6 endpoints)
mlflow_tracker.py      MLflow experiment tracking (3 runs)
evidently_monitor.py   Evidently AI drift monitoring (3 HTML reports)
Dockerfile             Production container (python:3.11-slim)
docker-compose.yml     api + ui services
requirements.txt       All dependencies
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/fraud/detect` | Score + red flags + recommendation |
| POST | `/api/compliance/query` | RAG over 5 regulations |
| POST | `/api/risk/report` | Full AML assessment report |
| GET  | `/api/health` | Uptime + model info |
| GET  | `/api/metrics` | Request counts + avg latency |
| GET  | `/api/regulations` | Regulation index |

Swagger UI: `http://localhost:8000/docs`

---

## Quick Start

```bash
git clone https://github.com/vijayarjun7/fintech-ai-agent
cd fintech-ai-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=your_key
export PINECONE_API_KEY=your_key

# Gradio UI
python app.py                                        # localhost:7860

# FastAPI
uvicorn api_server:app --reload --port 8000          # localhost:8000

# MLflow
python mlflow_tracker.py
mlflow ui --port 5050                                # localhost:5050

# Evidently drift monitor (requires Python 3.11)
python evidently_monitor.py
# open reports/data_drift_report.html
```

---

## Docker

```bash
# create .env
echo "ANTHROPIC_API_KEY=your_key" > .env
echo "PINECONE_API_KEY=your_key" >> .env

# build + run both services
docker compose up --build

# FastAPI only
docker run -p 8000:8000 --env-file .env fintech-ai-agent

# Gradio only
docker run -p 7860:7860 --env-file .env fintech-ai-agent python app.py

# health check
curl http://localhost:8000/api/health
```

---

## Regulations Covered

| Regulation | Coverage |
|---|---|
| KYC | Identity verification, EDD for PEPs, source of funds |
| AML | SARs within 30 days, CTRs >$10K, structuring detection |
| GDPR | Consent, 72hr breach notification, right to erasure |
| SOX | Section 302/404, 7-year document retention |
| PCI-DSS | CVV rules, TLS 1.2+, AES-256, quarterly scans |

---

## Why the Eval Layer Matters

In fintech, AI cannot hallucinate. A fabricated regulation reference = legal liability.
Every Claude response is scored for faithfulness and hallucination risk before returning to the user.
The Evidently drift monitor catches model degradation before it reaches production.
