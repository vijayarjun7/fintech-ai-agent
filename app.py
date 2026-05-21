import os
import json
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
MODEL = "claude-sonnet-4-5"
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
# SECTION 3 — RISK REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

RISK_SYSTEM_PROMPT = (
    "You are a senior AML compliance officer. "
    "Generate a formal risk assessment report. "
    "Be precise, cite specific regulations (BSA, FATF, FinCEN). "
    "Never fabricate regulatory references. "
    "Return valid JSON only."
)


@traceable(name="risk_report_generation")
def generate_risk_report(
    customer_name: str,
    account_type: str,
    transaction_history: str,
    country: str,
    income_range: str,
):
    if not customer_name.strip() or not transaction_history.strip():
        return (
            '<p style="color:#ef4444">Please fill in Customer Name and Transaction History.</p>',
            "",
            "",
            "",
        )

    prompt = f"""Generate a formal AML risk assessment report for this customer profile.

CUSTOMER PROFILE:
- Name: {customer_name}
- Account Type: {account_type}
- Country of Origin: {country}
- Monthly Income Range: {income_range}

TRANSACTION HISTORY:
{transaction_history}

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "risk_score": <integer 0-100>,
  "report": {{
    "customer_profile": "<text>",
    "transaction_analysis": "<text>",
    "red_flags": ["flag1", "flag2"],
    "regulatory_considerations": "<text>",
    "recommended_actions": ["action1", "action2"],
    "compliance_notes": "<text>"
  }},
  "faithfulness_score": <float 0.0-1.0>,
  "hallucination_risk": "LOW" | "MEDIUM" | "HIGH"
}}"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=RISK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
    except Exception as e:
        return (
            f'<p style="color:#ef4444">Error: {str(e)}</p>',
            "",
            "",
            "",
        )

    risk_level = data.get("risk_level", "MEDIUM").upper()
    risk_score = int(data.get("risk_score", 50))
    report = data.get("report", {})
    faithfulness = float(data.get("faithfulness_score", 0.8))
    hal_risk = data.get("hallucination_risk", "MEDIUM").upper()

    _update_metrics(
        quality_score=faithfulness,
        hallucination=hal_risk == "HIGH",
        risk_alert=risk_level in ("HIGH", "CRITICAL"),
    )

    # Risk level badge
    badge_colors = {
        "LOW":      ("#22c55e", "#052e16"),
        "MEDIUM":   ("#f59e0b", "#1c1400"),
        "HIGH":     ("#ef4444", "#1c0000"),
        "CRITICAL": ("#7f1d1d", "#1c0000"),
    }
    fg, bg = badge_colors.get(risk_level, ("#94a3b8", "#0f172a"))
    badge_html = (
        f'<div style="display:inline-block;background:{bg};border:2px solid {fg};'
        f'border-radius:8px;padding:10px 24px;font-size:22px;font-weight:800;color:{fg}">'
        f'{risk_level}</div>'
    )

    # Progress bar (0-100)
    bar_color = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#7f1d1d"}.get(risk_level, "#94a3b8")
    score_html = f"""
<div style="margin:12px 0">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
    <span style="color:#94a3b8;font-size:13px">Overall Risk Score</span>
    <span style="color:{bar_color};font-weight:700">{risk_score}/100</span>
  </div>
  <div style="background:#1e293b;border-radius:8px;height:18px;overflow:hidden">
    <div style="width:{risk_score}%;background:{bar_color};height:100%;border-radius:8px;transition:width 0.4s ease"></div>
  </div>
</div>"""

    # Formal report markdown
    red_flags = report.get("red_flags", [])
    actions = report.get("recommended_actions", [])
    flags_md = "\n".join(f"- {f}" for f in red_flags) if red_flags else "- None identified"
    actions_md = "\n".join(f"- {a}" for a in actions) if actions else "- No actions required"

    report_md = f"""## Customer Risk Profile
{report.get("customer_profile", "")}

## Transaction Pattern Analysis
{report.get("transaction_analysis", "")}

## Red Flags Identified
{flags_md}

## Regulatory Considerations
{report.get("regulatory_considerations", "")}

## Recommended Actions
{actions_md}

## Compliance Officer Notes
{report.get("compliance_notes", "")}"""

    # Eval scores
    hal_color = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(hal_risk, "#94a3b8")
    eval_html = (
        f'<div style="font-size:13px;line-height:1.8">'
        f'<b>Faithfulness:</b> {faithfulness:.0%}<br>'
        f'<b>Hallucination Risk:</b> <span style="color:{hal_color};font-weight:700">{hal_risk}</span>'
        f'</div>'
    )

    return badge_html + score_html, report_md, eval_html, ""


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
.block { background:#0f172a !important; border:1px solid #1e293b !important; border-radius:10px !important; }
textarea, input[type=text] { background:#1e293b !important; color:#e2e8f0 !important; border:1px solid #334155 !important; border-radius:8px !important; }
textarea:focus, input[type=text]:focus { border-color:#3b82f6 !important; outline:none !important; box-shadow:0 0 0 2px rgba(59,130,246,0.3) !important; }
button.primary { background:linear-gradient(135deg,#1d4ed8,#7c3aed) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
button.secondary { background:#1e293b !important; color:#94a3b8 !important; border:1px solid #334155 !important; border-radius:8px !important; }
.label { color:#94a3b8 !important; font-size:13px !important; }
.markdown-body, .prose { color:#e2e8f0 !important; }
table { border-collapse:collapse; width:100%; }
th { background:#1e3a5f; color:#93c5fd; padding:8px 12px; text-align:left; font-size:13px; }
td { padding:8px 12px; border-bottom:1px solid #1e293b; font-size:13px; color:#cbd5e1; }
tr:hover td { background:#0f1f35; }
"""

with gr.Blocks(title="Fintech AI Agent") as demo:

    gr.Markdown("# Fintech AI Agent\nPowered by Claude · Pinecone · LangSmith")

    # ── SECTION 1: FRAUD DETECTOR ─────────────────────────────────────────
    with gr.Accordion("🛡️ Fraud Detector", open=True):
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

    # ── SECTION 2: COMPLIANCE Q&A ─────────────────────────────────────────
    with gr.Accordion("⚖️ Compliance Q&A", open=False):
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

    # ── SECTION 3: RISK REPORT GENERATOR ─────────────────────────────────
    with gr.Accordion("📋 Risk Report Generator", open=False):
        gr.Markdown(
            "**Generate formal AML risk assessment reports for customer profiles.** "
            "Used by compliance officers for documentation and audit trails."
        )
        with gr.Row():
            with gr.Column(scale=1):
                rr_name = gr.Textbox(
                    label="Customer Name",
                    placeholder="e.g. John Smith",
                )
                rr_account_type = gr.Dropdown(
                    label="Account Type",
                    choices=["Personal", "Business", "Corporate", "PEP"],
                    value="Personal",
                )
                rr_transactions = gr.Textbox(
                    label="Transaction History",
                    placeholder="Paste recent transactions, one per line\ne.g.\n$9,800 cash deposit\n$9,700 wire to Mexico",
                    lines=5,
                )
                rr_country = gr.Textbox(
                    label="Country of Origin",
                    placeholder="e.g. USA",
                )
                rr_income = gr.Dropdown(
                    label="Monthly Income Range",
                    choices=["<$5K", "$5K-$25K", "$25K-$100K", "$100K+"],
                    value="$5K-$25K",
                )
                rr_btn = gr.Button("Generate Risk Report", variant="primary")
                gr.Examples(
                    label="Example Profiles",
                    examples=[
                        ["John Smith",       "Personal",  "$9,800 cash deposit\n$9,700 wire to Mexico\n$9,500 cash withdrawal", "USA",            "$5K-$25K"],
                        ["Acme Trading LLC", "Business",  "$500K wire from offshore\n$450K split to 10 accounts\n$480K crypto purchase", "Cayman Islands", "$100K+"],
                    ],
                    inputs=[rr_name, rr_account_type, rr_transactions, rr_country, rr_income],
                )
            with gr.Column(scale=1):
                rr_badge = gr.HTML(label="Risk Level")
                rr_eval  = gr.HTML(label="Eval Scores")

        rr_report = gr.Markdown(label="Formal Risk Assessment Report")
        rr_dummy  = gr.State("")

        rr_btn.click(
            fn=generate_risk_report,
            inputs=[rr_name, rr_account_type, rr_transactions, rr_country, rr_income],
            outputs=[rr_badge, rr_report, rr_eval, rr_dummy],
            api_name="risk_report",
        )

    # ── SECTION 4: EVAL DASHBOARD ─────────────────────────────────────────
    with gr.Accordion("📊 Eval Dashboard", open=False):
        gr.Markdown(
            "**Real-time evaluation metrics** aggregated across all sections. "
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
        ssr_mode=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="violet", neutral_hue="slate"),
        css=DARK_CSS,
    )
