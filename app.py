import os
import json
import time
import re
import gradio as gr
from anthropic import Anthropic

# ── LangSmith tracing (optional, no-op if not configured) ──────────────────
try:
    from langsmith import traceable
    LANGSMITH_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
except ImportError:
    def traceable(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f
    LANGSMITH_ENABLED = False

# ── Pinecone (optional, graceful degradation) ──────────────────────────────
try:
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    _pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    PINECONE_ENABLED = True
except Exception:
    PINECONE_ENABLED = False

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a fintech AI assistant. Be precise, cite sources, "
    "never hallucinate. If uncertain, say so explicitly."
)

# ── Shared metrics state ────────────────────────────────────────────────────
metrics = {
    "total_queries": 0,
    "quality_scores": [],
    "hallucinations_flagged": 0,
    "risk_alerts": 0,
}


def _call_claude(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _update_metrics(quality_score: float = None, hallucination: bool = False, risk_alert: bool = False):
    metrics["total_queries"] += 1
    if quality_score is not None:
        metrics["quality_scores"].append(quality_score)
    if hallucination:
        metrics["hallucinations_flagged"] += 1
    if risk_alert:
        metrics["risk_alerts"] += 1


# ── Compliance knowledge base ───────────────────────────────────────────────
REGULATIONS = {
    "KYC": (
        "Know Your Customer (KYC): Financial institutions must verify customer identity "
        "before onboarding. Requirements include: government-issued photo ID, proof of address "
        "(utility bill <3 months old), date of birth verification, source of funds declaration "
        "for transactions >$10,000, and enhanced due diligence (EDD) for PEPs (Politically "
        "Exposed Persons). Ongoing monitoring required. [FinCEN CDD Rule, FATF Recommendations]"
    ),
    "AML": (
        "Anti-Money Laundering (AML): Institutions must file Suspicious Activity Reports (SARs) "
        "within 30 days of detecting suspicious activity. Currency Transaction Reports (CTRs) "
        "required for cash transactions >$10,000. Structuring (breaking transactions to avoid "
        "reporting) is illegal. Required: AML compliance officer, employee training, independent "
        "audit, customer risk profiling. [Bank Secrecy Act, USA PATRIOT Act, FATF 40 Recommendations]"
    ),
    "GDPR": (
        "General Data Protection Regulation (GDPR): EU/EEA customers' personal data must be "
        "processed lawfully. Rights: access, rectification, erasure ('right to be forgotten'), "
        "portability, restriction, objection. Consent must be explicit and withdrawable. "
        "Data breaches must be reported to supervisory authority within 72 hours. "
        "Data retention limited to necessary period. DPO required for large-scale processing. "
        "[EU Regulation 2016/679]"
    ),
    "SOX": (
        "Sarbanes-Oxley Act (SOX): Applies to publicly traded companies. Section 302: CEO/CFO "
        "must personally certify financial reports. Section 404: Annual assessment of internal "
        "controls over financial reporting, audited by external auditors. Section 802: "
        "Document retention — financial records 7 years, audit workpapers 7 years. "
        "Whistleblower protections mandatory. Criminal penalties for willful violations. "
        "[Pub.L. 107-204, 116 Stat. 745]"
    ),
    "PCI-DSS": (
        "Payment Card Industry Data Security Standard (PCI-DSS v4.0): 12 requirements across "
        "6 goals. Key rules: never store CVV after authorization, encrypt cardholder data in "
        "transit (TLS 1.2+) and at rest (AES-256), restrict access on need-to-know basis, "
        "maintain firewall configuration, use unique IDs per user, quarterly network scans, "
        "annual penetration testing. Levels 1-4 based on transaction volume. "
        "Level 1 (>6M transactions/year) requires QSA audit. [PCI SSC]"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — FRAUD DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="fraud_detection")
def analyze_fraud(transaction: str):
    if not transaction.strip():
        return _empty_fraud_result("Please enter a transaction description.")

    prompt = f"""Analyze this financial transaction for fraud risk.

Transaction: {transaction}

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "risk_score": <integer 0-10>,
  "red_flags": ["flag1", "flag2"],
  "recommendation": "approve" | "review" | "reject",
  "reasoning": "<brief explanation>",
  "faithfulness_score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>
}}"""

    try:
        raw = _call_claude(prompt)
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
    except Exception as e:
        return _empty_fraud_result(f"Analysis error: {str(e)}")

    risk_score = int(data.get("risk_score", 0))
    red_flags = data.get("red_flags", [])
    recommendation = data.get("recommendation", "review").lower()
    reasoning = data.get("reasoning", "")
    faithfulness = float(data.get("faithfulness_score", 0.8))
    confidence = float(data.get("confidence", 0.8))

    is_alert = risk_score >= 7
    _update_metrics(
        quality_score=(faithfulness + confidence) / 2,
        risk_alert=is_alert,
    )

    # Build colored progress bar HTML
    if risk_score <= 3:
        bar_color = "#22c55e"
        risk_label = "LOW RISK"
        label_color = "#22c55e"
    elif risk_score <= 6:
        bar_color = "#f59e0b"
        risk_label = "MEDIUM RISK"
        label_color = "#f59e0b"
    else:
        bar_color = "#ef4444"
        risk_label = "HIGH RISK"
        label_color = "#ef4444"

    fill_pct = risk_score * 10
    risk_html = f"""
<div style="margin:8px 0">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
    <span style="color:#94a3b8;font-size:13px">Risk Score</span>
    <span style="color:{label_color};font-weight:700;font-size:14px">{risk_score}/10 — {risk_label}</span>
  </div>
  <div style="background:#1e293b;border-radius:8px;height:18px;overflow:hidden">
    <div style="width:{fill_pct}%;background:{bar_color};height:100%;border-radius:8px;
                transition:width 0.4s ease"></div>
  </div>
</div>"""

    flags_md = "\n".join(f"- {f}" for f in red_flags) if red_flags else "- No significant flags detected"

    rec_color = {"approve": "#22c55e", "review": "#f59e0b", "reject": "#ef4444"}.get(recommendation, "#94a3b8")
    rec_html = f'<span style="color:{rec_color};font-weight:700;font-size:16px;text-transform:uppercase">⬤ {recommendation}</span>'

    eval_md = (
        f"**Faithfulness:** {faithfulness:.0%}  \n"
        f"**Confidence:** {confidence:.0%}  \n"
        f"**Reasoning:** {reasoning}"
    )

    return risk_html, flags_md, rec_html, eval_md


def _empty_fraud_result(msg):
    return (
        f'<p style="color:#ef4444">{msg}</p>',
        "",
        "",
        "",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPLIANCE Q&A
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="compliance_qa")
def answer_compliance(question: str):
    if not question.strip():
        return "Please enter a compliance question.", "", ""

    context = "\n\n".join(f"[{k}]\n{v}" for k, v in REGULATIONS.items())

    prompt = f"""You are a compliance expert. Use ONLY the provided regulatory context to answer.

REGULATORY CONTEXT:
{context}

QUESTION: {question}

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "answer": "<detailed answer citing specific regulation>",
  "source": "<regulation name, e.g. KYC/AML/GDPR/SOX/PCI-DSS>",
  "confidence": <float 0.0-1.0>,
  "hallucination_risk": "LOW" | "MEDIUM" | "HIGH"
}}"""

    try:
        raw = _call_claude(prompt)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
    except Exception as e:
        return f"Error: {str(e)}", "", ""

    answer = data.get("answer", "")
    source = data.get("source", "Unknown")
    confidence = float(data.get("confidence", 0.7))
    hal_risk = data.get("hallucination_risk", "MEDIUM").upper()

    hal_color = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(hal_risk, "#94a3b8")
    hal_html = f'<span style="color:{hal_color};font-weight:700">⬤ {hal_risk} hallucination risk</span>'

    is_hallucination = hal_risk == "HIGH"
    _update_metrics(quality_score=confidence, hallucination=is_hallucination)

    answer_md = f"**Source: {source}**\n\n{answer}\n\n**Confidence:** {confidence:.0%}"
    conf_html = f"""
<div style="margin:8px 0">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="color:#94a3b8;font-size:13px">Confidence</span>
    <span style="color:#60a5fa;font-weight:600">{confidence:.0%}</span>
  </div>
  <div style="background:#1e293b;border-radius:6px;height:12px;overflow:hidden">
    <div style="width:{confidence*100:.0f}%;background:#3b82f6;height:100%;border-radius:6px"></div>
  </div>
</div>"""

    return answer_md, conf_html, hal_html


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — TEST CASE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

_test_case_store: list[dict] = []


def _check_pinecone_duplicate(description: str) -> str:
    if not PINECONE_ENABLED:
        return "Pinecone not configured — duplicate check skipped."
    try:
        index_name = os.environ.get("PINECONE_INDEX", "fintech-testcases")
        index = _pc.Index(index_name)
        vec = _embed_model.encode(description).tolist()
        results = index.query(vector=vec, top_k=1, include_metadata=True)
        if results.matches and results.matches[0].score > 0.92:
            meta = results.matches[0].metadata or {}
            return f"Similar test case exists (score {results.matches[0].score:.2f}): {meta.get('description', 'N/A')}"
        return "No duplicate found in vector store."
    except Exception as e:
        return f"Duplicate check error: {str(e)}"


@traceable(name="test_case_generation")
def generate_test_cases(feature: str):
    if not feature.strip():
        return "", "", "Please enter a feature description.", ""

    prompt = f"""Generate exactly 5 QA test cases for this banking app feature: {feature}

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "test_cases": [
    {{
      "id": "TC001",
      "title": "<short title>",
      "steps": "<numbered steps>",
      "expected": "<expected result>",
      "quality_score": <float 0.0-1.0>,
      "specificity_score": <float 0.0-1.0>
    }}
  ]
}}

Generate exactly 5 test cases covering: happy path, edge cases, error handling, security, performance."""

    try:
        raw = _call_claude(prompt)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
        cases = data.get("test_cases", [])
    except Exception as e:
        return "", "", f"Generation error: {str(e)}", ""

    if not cases:
        return "", "", "No test cases generated.", ""

    # Build markdown table
    table_md = "| ID | Title | Steps | Expected | Quality | Specificity |\n"
    table_md += "|---|---|---|---|---|---|\n"
    for tc in cases:
        q = float(tc.get("quality_score", 0.8))
        s = float(tc.get("specificity_score", 0.8))
        table_md += (
            f"| {tc.get('id','')} | {tc.get('title','')} | "
            f"{tc.get('steps','')} | {tc.get('expected','')} | "
            f"{q:.0%} | {s:.0%} |\n"
        )

    # Eval scores summary
    avg_q = sum(float(tc.get("quality_score", 0.8)) for tc in cases) / len(cases)
    avg_s = sum(float(tc.get("specificity_score", 0.8)) for tc in cases) / len(cases)
    eval_md = f"**Avg Quality:** {avg_q:.0%}  \n**Avg Specificity:** {avg_s:.0%}  \n**Cases Generated:** {len(cases)}"

    _update_metrics(quality_score=(avg_q + avg_s) / 2)

    # Store for export
    _test_case_store.clear()
    _test_case_store.extend(cases)

    dup_status = _check_pinecone_duplicate(feature)

    return table_md, eval_md, dup_status, json.dumps(cases, indent=2)


def export_test_cases():
    if not _test_case_store:
        return None
    path = "/tmp/test_cases.json"
    with open(path, "w") as f:
        json.dump(_test_case_store, f, indent=2)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — EVAL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def get_dashboard():
    total = metrics["total_queries"]
    scores = metrics["quality_scores"]
    avg_q = (sum(scores) / len(scores)) if scores else 0.0
    hal = metrics["hallucinations_flagged"]
    alerts = metrics["risk_alerts"]

    gauge_pct = avg_q * 100

    html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:8px">

  <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;padding:20px;text-align:center">
    <div style="color:#94a3b8;font-size:13px;margin-bottom:8px">TOTAL QUERIES</div>
    <div style="color:#60a5fa;font-size:48px;font-weight:700">{total}</div>
  </div>

  <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;padding:20px;text-align:center">
    <div style="color:#94a3b8;font-size:13px;margin-bottom:8px">AVG QUALITY SCORE</div>
    <div style="color:#a78bfa;font-size:48px;font-weight:700">{avg_q:.0%}</div>
    <div style="background:#1e293b;border-radius:6px;height:8px;margin-top:10px;overflow:hidden">
      <div style="width:{gauge_pct:.0f}%;background:#7c3aed;height:100%;border-radius:6px"></div>
    </div>
  </div>

  <div style="background:#0f172a;border:1px solid #3b1f1f;border-radius:12px;padding:20px;text-align:center">
    <div style="color:#94a3b8;font-size:13px;margin-bottom:8px">HALLUCINATIONS FLAGGED</div>
    <div style="color:#f87171;font-size:48px;font-weight:700">{hal}</div>
  </div>

  <div style="background:#0f172a;border:1px solid #3b2f0a;border-radius:12px;padding:20px;text-align:center">
    <div style="color:#94a3b8;font-size:13px;margin-bottom:8px">RISK ALERTS TRIGGERED</div>
    <div style="color:#fb923c;font-size:48px;font-weight:700">{alerts}</div>
  </div>

</div>
<p style="color:#475569;font-size:12px;text-align:center;margin-top:12px">
  Metrics update automatically after each query across all tabs.
</p>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

DARK_CSS = """
body, .gradio-container { background:#0a0f1e !important; color:#e2e8f0 !important; }
.tab-nav button { background:#0f172a !important; color:#94a3b8 !important; border:1px solid #1e293b !important; }
.tab-nav button.selected { background:#1e3a5f !important; color:#60a5fa !important; border-color:#3b82f6 !important; }
.block { background:#0f172a !important; border:1px solid #1e293b !important; border-radius:10px !important; }
textarea, input[type=text] { background:#1e293b !important; color:#e2e8f0 !important; border:1px solid #334155 !important; border-radius:8px !important; }
textarea:focus, input[type=text]:focus { border-color:#3b82f6 !important; outline:none !important; box-shadow:0 0 0 2px rgba(59,130,246,0.3) !important; }
button.primary { background:linear-gradient(135deg,#1d4ed8,#7c3aed) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
button.secondary { background:#1e293b !important; color:#94a3b8 !important; border:1px solid #334155 !important; border-radius:8px !important; }
.label { color:#94a3b8 !important; font-size:13px !important; }
.markdown-body { color:#e2e8f0 !important; }
table { border-collapse:collapse; width:100%; }
th { background:#1e3a5f; color:#93c5fd; padding:8px 12px; text-align:left; font-size:13px; }
td { padding:8px 12px; border-bottom:1px solid #1e293b; font-size:13px; color:#cbd5e1; }
tr:hover td { background:#0f1f35; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="violet",
        neutral_hue="slate",
    ),
    css=DARK_CSS,
    title="Fintech AI Agent",
) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
      <h1 style="font-size:28px;font-weight:800;background:linear-gradient(90deg,#60a5fa,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
        Fintech AI Agent
      </h1>
      <p style="color:#64748b;font-size:14px;margin-top:6px">
        Powered by Claude · Pinecone · LangSmith
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: FRAUD DETECTOR ──────────────────────────────────────────
        with gr.Tab("🛡️ Fraud Detector"):
            gr.Markdown(
                "**Analyze financial transactions for fraud patterns.** "
                "Receive a risk score, red flags, and a clear approve/review/reject recommendation."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    fraud_input = gr.Textbox(
                        label="Transaction Description",
                        placeholder='e.g. "Transfer $9,800 to overseas account at 3am"',
                        lines=4,
                        max_lines=8,
                    )
                    fraud_btn = gr.Button("Analyze Transaction", variant="primary")
                    gr.Examples(
                        examples=[
                            ["Transfer $9,800 to overseas account at 3am"],
                            ["$50 grocery purchase at local supermarket"],
                            ["Wire transfer of $9,500 to 5 different accounts within 1 hour"],
                            ["Monthly Netflix subscription charge $15.99"],
                        ],
                        inputs=fraud_input,
                    )
                with gr.Column(scale=1):
                    fraud_risk_bar = gr.HTML(label="Risk Score")
                    fraud_flags = gr.Markdown(label="Red Flags")
                    fraud_rec = gr.HTML(label="Recommendation")
                    fraud_eval = gr.Markdown(label="Eval Scores")

            fraud_btn.click(
                fn=analyze_fraud,
                inputs=fraud_input,
                outputs=[fraud_risk_bar, fraud_flags, fraud_rec, fraud_eval],
                api_name="fraud_detect",
            )

        # ── TAB 2: COMPLIANCE Q&A ──────────────────────────────────────────
        with gr.Tab("⚖️ Compliance Q&A"):
            gr.Markdown(
                "**RAG-powered compliance assistant** covering KYC, AML, GDPR, SOX, and PCI-DSS. "
                "Each answer cites the source regulation and flags hallucination risk."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    compliance_input = gr.Textbox(
                        label="Compliance Question",
                        placeholder='e.g. "What are KYC requirements for new customers?"',
                        lines=3,
                    )
                    compliance_btn = gr.Button("Get Answer", variant="primary")
                    gr.Examples(
                        examples=[
                            ["What are KYC requirements for new customers?"],
                            ["When must we file a Suspicious Activity Report?"],
                            ["What does GDPR say about data breach notification?"],
                            ["What are SOX Section 404 requirements?"],
                            ["What encryption standard does PCI-DSS require?"],
                        ],
                        inputs=compliance_input,
                    )
                with gr.Column(scale=1):
                    compliance_answer = gr.Markdown(label="Answer")
                    compliance_conf = gr.HTML(label="Confidence")
                    compliance_hal = gr.HTML(label="Hallucination Risk")

            compliance_btn.click(
                fn=answer_compliance,
                inputs=compliance_input,
                outputs=[compliance_answer, compliance_conf, compliance_hal],
                api_name="compliance_qa",
            )

        # ── TAB 3: TEST CASE GENERATOR ─────────────────────────────────────
        with gr.Tab("🧪 Test Case Generator"):
            gr.Markdown(
                "**Generate QA test cases for banking app features.** "
                "Covers happy path, edge cases, error handling, security, and performance. "
                "Includes duplicate detection via Pinecone."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    tc_input = gr.Textbox(
                        label="Feature Description",
                        placeholder='e.g. "Login feature with 2FA authentication"',
                        lines=3,
                    )
                    tc_btn = gr.Button("Generate Test Cases", variant="primary")
                    gr.Examples(
                        examples=[
                            ["Login feature with 2FA authentication"],
                            ["International wire transfer with FX conversion"],
                            ["Credit card spend limit enforcement"],
                            ["Password reset via email verification"],
                        ],
                        inputs=tc_input,
                    )
                with gr.Column(scale=1):
                    tc_dup = gr.Textbox(label="Duplicate Check (Pinecone)", interactive=False)
                    tc_eval = gr.Markdown(label="Eval Scores")
                    tc_export_btn = gr.Button("Export as JSON", variant="secondary")
                    tc_file = gr.File(label="Download JSON", visible=False)

            tc_table = gr.Markdown(label="Generated Test Cases")
            tc_json_store = gr.State("")

            tc_btn.click(
                fn=generate_test_cases,
                inputs=tc_input,
                outputs=[tc_table, tc_eval, tc_dup, tc_json_store],
                api_name="generate_tests",
            )
            tc_export_btn.click(
                fn=export_test_cases,
                inputs=[],
                outputs=tc_file,
            ).then(lambda: gr.update(visible=True), outputs=tc_file)

        # ── TAB 4: EVAL DASHBOARD ──────────────────────────────────────────
        with gr.Tab("📊 Eval Dashboard"):
            gr.Markdown(
                "**Real-time evaluation metrics** aggregated across all tabs. "
                "Refresh to see the latest counts after running queries."
            )
            dashboard_html = gr.HTML(value=get_dashboard)
            refresh_btn = gr.Button("↻ Refresh Metrics", variant="secondary")
            refresh_btn.click(fn=get_dashboard, inputs=[], outputs=dashboard_html)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
