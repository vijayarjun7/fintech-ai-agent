# Fintech AI Agent 🏦

> AI-powered fraud detection, compliance Q&A,
> and AML risk reports — with hallucination eval layer.
> Powered by Claude + Pinecone + LangSmith

![Claude](https://img.shields.io/badge/Claude-Sonnet%204-orange?logo=anthropic)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green?logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)

## 🤗 Live Demo
https://huggingface.co/spaces/Vijayarv07/fintech-ai-agent

## What It Does

| Tab | Function |
|---|---|
| 🛡️ Fraud Detector | Risk score 0-10, red flags, approve/review/reject |
| ⚖️ Compliance Q&A | RAG over KYC/AML/GDPR/SOX/PCI-DSS |
| 📋 Risk Report | Formal 6-section AML assessment |
| 📊 Eval Dashboard | Real-time quality metrics |

## Key Results

**Live demo run — 3 queries across all sections:**

| Section | Input | Result |
|---|---|---|
| 🛡️ Fraud Detector | `$9,800 to Cayman Islands at 3:47am` | 9/10 HIGH RISK → **REJECT** |
| ⚖️ Compliance Q&A | `When must we file a SAR under BSA?` | Answer cited BSA + FATF, 95% confidence |
| 📋 Risk Report | Sarah Johnson, Personal, USA | 15/100 LOW — 6-section formal report |

**Eval scores:**

| Metric | Value |
|---|---|
| Avg quality score | 98% |
| Hallucinations flagged | 0 |
| Faithfulness — Fraud Detector | 95–100% |
| Faithfulness — Risk Report | 92% |
| Hallucination risk | LOW across all runs |
| Confidence — Compliance Q&A | 95–96% |

**MLflow tracked runs:**

| Run | Confidence | Risk Score | Response Time |
|---|---|---|---|
| FraudDetector | 85% | 9/10 | 1200 ms |
| ComplianceQA | 92% | — | 980 ms |
| RiskReportGenerator | — | 78/100 | 2100 ms |
| **Avg** | | | **1427 ms** |

## Stack

```
app.py            Gradio UI (4 accordions)
api_server.py     FastAPI REST API (6 endpoints)
mlflow_tracker.py MLflow experiment tracking
requirements.txt  Dependencies
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/fraud/detect` | Score + red flags + recommendation |
| POST | `/api/compliance/query` | RAG over 5 regulations |
| POST | `/api/risk/report` | Full AML assessment report |
| GET  | `/api/health` | Uptime + model info |
| GET  | `/api/metrics` | Request counts + avg latency |
| GET  | `/api/regulations` | Regulation index |

## Quick Start

```bash
git clone https://github.com/vijayarjun7/fintech-ai-agent
cd fintech-ai-agent
pip install -r requirements.txt

export ANTHROPIC_API_KEY=your_key
export PINECONE_API_KEY=your_key

# Gradio UI
python app.py

# FastAPI server
uvicorn api_server:app --reload --port 8000
# Swagger: http://localhost:8000/docs

# MLflow tracking
python mlflow_tracker.py
mlflow ui --port 5050
```

## Regulations Covered

| Regulation | Coverage |
|---|---|
| KYC | Identity verification, EDD for PEPs |
| AML | SARs, CTRs, structuring detection |
| GDPR | Consent, breach notification, erasure |
| SOX | Section 302/404, 7-year retention |
| PCI-DSS | CVV rules, TLS 1.2+, AES-256 |

## Docker

```bash
# Build image
docker build -t fintech-ai-agent .

# Run FastAPI
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  fintech-ai-agent

# Test health
curl http://localhost:8000/api/health

# Run with docker-compose
docker-compose up

# Run Gradio UI instead
docker run -p 7860:7860 \
  -e ANTHROPIC_API_KEY=your_key \
  fintech-ai-agent \
  python app.py
```

## Why the Eval Layer Matters

In fintech, AI cannot hallucinate. A fabricated regulation reference = legal liability.
Every Claude response is scored for faithfulness and hallucination risk before returning to the user.
