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

| Metric | Value |
|---|---|
| Avg quality score | 98% |
| Hallucinations flagged | 0 |
| Faithfulness (Fraud) | 95–100% |
| Faithfulness (Risk Report) | 92% |
| Hallucination risk | LOW across all runs |

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

## Why the Eval Layer Matters

In fintech, AI cannot hallucinate. A fabricated regulation reference = legal liability.
Every Claude response is scored for faithfulness and hallucination risk before returning to the user.
