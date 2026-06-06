import os

import numpy as np
import pandas as pd
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite

os.makedirs("reports", exist_ok=True)
rng = np.random.default_rng(42)

# ── Reference dataset: Week 1 baseline ────────────────────────────────────
# 80% legitimate (risk 0-4), 20% suspicious (risk 7-10)

n = 50
legit = int(n * 0.8)  # 40 rows
suspic = n - legit  # 10 rows

ref_legit = pd.DataFrame(
    {
        "transaction_amount": rng.uniform(100, 8_000, legit),
        "hour_of_day": rng.integers(8, 20, legit),  # business hours
        "is_overseas": rng.choice([0, 1], legit, p=[0.8, 0.2]).astype(float),
        "risk_score": rng.uniform(0, 4, legit),
        "recommendation": rng.choice(["approve", "review"], legit, p=[0.85, 0.15]),
        "fraud_label": np.zeros(legit, dtype=int),
    }
)

ref_suspic = pd.DataFrame(
    {
        "transaction_amount": rng.uniform(8_000, 50_000, suspic),
        "hour_of_day": rng.integers(0, 6, suspic),  # odd hours
        "is_overseas": np.ones(suspic),
        "risk_score": rng.uniform(7, 10, suspic),
        "recommendation": rng.choice(["review", "reject"], suspic, p=[0.3, 0.7]),
        "fraud_label": np.ones(suspic, dtype=int),
    }
)

reference = (
    pd.concat([ref_legit, ref_suspic], ignore_index=True)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

# ── Current dataset: Week 3 — drifted ─────────────────────────────────────
# 50% high-risk, 60% overseas, larger amounts

n_cur = 50
cur_legit = int(n_cur * 0.5)
cur_suspic = n_cur - cur_legit

cur_l = pd.DataFrame(
    {
        "transaction_amount": rng.uniform(5_000, 20_000, cur_legit),
        "hour_of_day": rng.integers(6, 22, cur_legit),
        "is_overseas": rng.choice([0, 1], cur_legit, p=[0.4, 0.6]).astype(float),
        "risk_score": rng.uniform(1, 6, cur_legit),
        "recommendation": rng.choice(["approve", "review"], cur_legit, p=[0.6, 0.4]),
        "fraud_label": np.zeros(cur_legit, dtype=int),
    }
)

cur_s = pd.DataFrame(
    {
        "transaction_amount": rng.uniform(20_000, 80_000, cur_suspic),
        "hour_of_day": rng.integers(0, 5, cur_suspic),
        "is_overseas": np.ones(cur_suspic),
        "risk_score": rng.uniform(6, 10, cur_suspic),
        "recommendation": np.full(cur_suspic, "reject"),
        "fraud_label": np.ones(cur_suspic, dtype=int),
    }
)

current = (
    pd.concat([cur_l, cur_s], ignore_index=True)
    .sample(frac=1, random_state=99)
    .reset_index(drop=True)
)

# ── Numeric columns used across reports ───────────────────────────────────
num_cols = ["transaction_amount", "hour_of_day", "is_overseas", "risk_score"]

# ── Report 1: Data Drift ───────────────────────────────────────────────────
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference[num_cols], current_data=current[num_cols])
drift_report.save_html("reports/data_drift_report.html")
print("✓ reports/data_drift_report.html saved")

# ── Report 2: Data Quality ─────────────────────────────────────────────────
quality_report = Report(metrics=[DataQualityPreset()])
quality_report.run(reference_data=reference[num_cols], current_data=current[num_cols])
quality_report.save_html("reports/data_quality_report.html")
print("✓ reports/data_quality_report.html saved")

# ── Report 3: Classification Performance ──────────────────────────────────
# Add target + prediction columns required by ClassificationPreset
ref_clf = reference[["fraud_label"]].copy()
ref_clf["prediction"] = reference["fraud_label"]  # baseline: perfect on reference

cur_clf = current[["fraud_label"]].copy()
# Simulate degraded model: flip ~30% of fraud predictions
flip_idx = rng.choice(cur_clf.index, size=int(n_cur * 0.3), replace=False)
cur_clf["prediction"] = current["fraud_label"].copy()
cur_clf.loc[flip_idx, "prediction"] = 1 - cur_clf.loc[flip_idx, "prediction"]

clf_report = Report(metrics=[ClassificationPreset()])
clf_report.run(
    reference_data=ref_clf.rename(columns={"fraud_label": "target"}),
    current_data=cur_clf.rename(columns={"fraud_label": "target"}),
)
clf_report.save_html("reports/classification_report.html")
print("✓ reports/classification_report.html saved")

# ── Test Suite: Data Drift ─────────────────────────────────────────────────
suite = TestSuite(tests=[DataDriftTestPreset()])
suite.run(reference_data=reference[num_cols], current_data=current[num_cols])
suite_dict = suite.as_dict()

# ── Parse test results ─────────────────────────────────────────────────────
tests = suite_dict.get("tests", [])
n_pass = sum(1 for t in tests if t.get("status") == "SUCCESS")
n_fail = len(tests) - n_pass
failed = [t.get("name", "unknown") for t in tests if t.get("status") != "SUCCESS"]

if n_fail == 0:
    overall_status = "HEALTHY ✅"
    action = "No action required"
elif n_fail <= 2:
    overall_status = "DEGRADED ⚠️"
    action = "Retrain fraud detection model"
else:
    overall_status = "CRITICAL 🔴"
    action = "Immediate retraining required — alert compliance team"

# ── Per-column drift lookup ────────────────────────────────────────────────
drift_result = drift_report.as_dict()
col_drift: dict[str, bool] = {}
try:
    for metric in drift_result.get("metrics", []):
        result = metric.get("result", {})
        drift_by_col = result.get("drift_by_columns", {})
        for col, info in drift_by_col.items():
            col_drift[col] = info.get("drift_detected", False)
except Exception:
    pass


def drift_line(col: str) -> str:
    detected = col_drift.get(col, False)
    return f"  - {col}: {'DRIFT DETECTED 🔴' if detected else 'NO DRIFT ✅'}"


# ── Console summary ────────────────────────────────────────────────────────
print("""
=== Evidently AI Monitoring Report ===
Reference period : Week 1 (50 transactions)
Current period   : Week 3 (50 transactions)

Data Drift:""")
for col in num_cols:
    print(drift_line(col))

print(f"""
Test Suite Results : {n_pass} passed / {n_fail} failed""")
for t in tests:
    icon = "✅" if t.get("status") == "SUCCESS" else "❌"
    print(f"  {icon} {t.get('name', 'unknown')}")

print(f"""
Overall Status   : {overall_status}
Action Required  : {action}

Reports saved to reports/ folder
  → reports/data_drift_report.html
  → reports/data_quality_report.html
  → reports/classification_report.html
""")
