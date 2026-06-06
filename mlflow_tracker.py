import mlflow

EXPERIMENT = "fintech-ai-agent-eval"

mlflow.set_experiment(EXPERIMENT)

# ── Run 1: Fraud Detector ──────────────────────────────────────────────────
with mlflow.start_run(run_name="FraudDetector"):
    mlflow.log_params({"model": "claude-sonnet-4-6", "threshold": 7})
    mlflow.log_metrics({
        "risk_score":       9,
        "faithfulness":     0.95,
        "confidence":       0.85,
        "response_time_ms": 1200,
    })
    mlflow.set_tags({"recommendation": "reject", "risk_level": "HIGH"})
    print("✓ Run 1 logged — FraudDetector")

# ── Run 2: Compliance Q&A ──────────────────────────────────────────────────
with mlflow.start_run(run_name="ComplianceQA"):
    mlflow.log_params({"model": "claude-sonnet-4-6", "regulation": "AML"})
    mlflow.log_metrics({
        "confidence":       0.92,
        "response_time_ms": 980,
    })
    mlflow.set_tags({"hallucination_risk": "LOW", "source": "BSA"})
    print("✓ Run 2 logged — ComplianceQA")

# ── Run 3: Risk Report Generator ───────────────────────────────────────────
with mlflow.start_run(run_name="RiskReportGenerator"):
    mlflow.log_params({
        "model":        "claude-sonnet-4-6",
        "account_type": "Personal",
        "country":      "USA",
    })
    mlflow.log_metrics({
        "risk_score":       78,
        "faithfulness":     0.92,
        "response_time_ms": 2100,
    })
    mlflow.set_tags({"risk_level": "HIGH"})
    print("✓ Run 3 logged — RiskReportGenerator")

# ── Comparison summary ─────────────────────────────────────────────────────
runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
runs = runs.head(3)

conf_col  = "metrics.confidence"
risk_col  = "metrics.risk_score"
time_col  = "metrics.response_time_ms"
name_col  = "tags.mlflow.runName"

best_conf = runs.loc[runs[conf_col].idxmax()] if conf_col in runs else None
best_risk = runs.loc[runs[risk_col].idxmax()] if risk_col in runs else None
avg_time  = runs[time_col].mean() if time_col in runs else None

print("\n── Comparison ─────────────────────────────────────────")
if best_conf is not None:
    print(f"Best confidence : {best_conf[conf_col]:.0%}  ({best_conf[name_col]})")
if best_risk is not None:
    print(f"Highest risk    : {int(best_risk[risk_col])}/100  ({best_risk[name_col]})")
if avg_time is not None:
    print(f"Avg response    : {avg_time:.0f} ms")
print("───────────────────────────────────────────────────────")
print(f"MLflow UI → mlflow ui --port 5050")
