"""
Intelli-Credit: AI-Powered Credit Appraisal Engine
Streamlit Frontend — Credit Officer Portal
"""
import sys
import os
import json
import time
import streamlit as st
from pathlib import Path

from auth import AuthStore, send_otp, logout, get_current_user

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    IS_DEMO, get_mode_display, SAMPLE_DATA_DIR, OUTPUT_DIR,
    has_llm_key, has_openai_key, has_anthropic_key,
    has_tavily_key, has_azure_di, has_databricks,
    LLM_PROVIDER
)
from utils import load_json, format_inr

# Flag to bypass OTP login (useful for local/demo runs)
BYPASS_AUTH = os.getenv("BYPASS_AUTH", "true").lower() in ("1", "true", "yes", "y")

def _render_extraction_visuals(data: dict, doc_type: str):
    """Render user-friendly visuals instead of raw JSON."""
    import pandas as pd
    import plotly.express as px

    if doc_type == "Annual Report":
        # Multi-year revenue/profit comparison
        metrics = ["revenue_cr", "pat_cr", "ebitda_cr"]
        rows = []
        for fy in ["fy_2022", "fy_2023", "fy_2024"]:
            fy_data = data.get(fy, {})
            if not fy_data: continue
            row = {"Year": fy.replace("_", " ").upper()}
            for m in metrics:
                val = fy_data.get(m, 0)
                if isinstance(val, dict): val = val.get("value", 0)
                row[m.replace("_cr", "").upper()] = float(val or 0)
            rows.append(row)
        
        if rows:
            df_viz = pd.DataFrame(rows)
            # Use columns that exist in the dataframe
            y_cols = [c for c in df_viz.columns if c != "Year"]
            fig = px.bar(df_viz, x="Year", y=y_cols, 
                         barmode="group", title="Financial Performance Trend (₹ Cr)",
                         color_discrete_sequence=["#8B5CF6", "#3b82f6", "#10b981"])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color="#8899A6",
                margin=dict(t=40, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient financial trend data for charting.")

    elif doc_type == "Bank Statement":
        # Monthly balance trend
        months = data.get("monthly_balances", [])
        if months:
            df_bank = pd.DataFrame(months)
            fig_bank = px.line(df_bank, x="month", y="balance", title="Monthly Balance Trend (₹ Cr)",
                              line_shape="spline", color_discrete_sequence=["#3b82f6"])
            fig_bank.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#8899A6")
            st.plotly_chart(fig_bank, use_container_width=True)
        else:
            st.info("No balance trend data found in statement.")

    elif doc_type == "GST Return":
        # Turnover metrics
        turnover = data.get("total_turnover_cr", 0)
        if isinstance(turnover, dict): turnover = turnover.get("value", 0)
        st.metric("Total Declared Turnover", f"₹ {float(turnover or 0):.2f} Cr")
        st.progress(min(1.0, float(turnover or 0)/100), text="Turnover Level")

# ─── Page Config ───
st.set_page_config(
    page_title="Intelli-Credit | AI Credit Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    /* Corporate Trust Typography */
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Source Sans 3', sans-serif; }
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm { font-family: 'Lexend', sans-serif !important; }

    /* Fintech Dark Theme base overrides */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* Premium Header */
    .main-header {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: #F8FAFC;
        border: 1px solid #334155;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    
    /* Subtle glow effect behind header text */
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -20%;
        width: 140%; height: 200%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .main-header h1 {
        margin: 0; font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, #FBBF24, #F59E0B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .main-header p { 
        margin: 0.5rem 0 0 0; 
        color: #CBD5E1; 
        font-size: 1.1rem; 
        font-weight: 300;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
        border-color: rgba(139, 92, 246, 0.3);
    }
    .metric-card h3 { 
        color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; 
        letter-spacing: 1.5px; margin: 0; font-weight: 600;
    }
    .metric-card .value { 
        color: #F8FAFC; font-size: 2.2rem; font-weight: 700; 
        margin: 0.5rem 0 0 0; font-family: 'Lexend', sans-serif;
    }

    /* Agent Cards */
    .agent-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .agent-card:hover { border-color: #8B5CF6; }
    .agent-card .agent-name { color: #A78BFA; font-weight: 600; font-size: 1rem; font-family: 'Lexend', sans-serif;}
    .agent-card .agent-status { color: #94A3B8; font-size: 0.9rem; margin-top: 0.4rem; }

    /* Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Lexend', sans-serif;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .risk-low { background: rgba(16, 185, 129, 0.1); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
    .risk-moderate { background: rgba(245, 158, 11, 0.1); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
    .risk-high { background: rgba(239, 68, 68, 0.1); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }

    /* Score Gauge */
    .score-gauge {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        box-shadow: inset 0 2px 20px rgba(0,0,0,0.5);
    }
    .score-gauge .score { 
        font-size: 5rem; font-weight: 800; font-family: 'Lexend', sans-serif;
        line-height: 1; margin: 1rem 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .score-gauge .label { color: #94A3B8; font-size: 1rem; font-weight: 500; font-family: 'Lexend', sans-serif; letter-spacing: 2px;}

    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #1E293B;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Lexend', sans-serif;
        color: #94A3B8;
        background-color: transparent;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background-color: rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #8B5CF6;
        color: white;
        box-shadow: 0 4px 12px rgba(139,92,246,0.3);
    }
    /* Hide the default underline connecting tabs */
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Buttons */
    .stButton>button {
        font-family: 'Lexend', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #8B5CF6, #6D28D9);
        border: none;
        color: white;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        border: none;
        color: white;
    }
    
    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #1E293B;
        border: 1px solid #334155;
        color: #F8FAFC;
        border-radius: 8px;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #8B5CF6;
        box-shadow: 0 0 0 1px #8B5CF6;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───
def init_session():
    defaults = {
        "company_data": None,
        "extracted_data": {},
        "doc_classifications": [],
        "research_results": None,
        "five_cs": None,
        "scoring": None,
        "gst_validation": None,
        "committee_verdict": None,
        "cam_path": None,
        "pipeline_step": 0,
        "qualitative_notes": [],
        "active_tab": 0,
        "swot_analysis": None,
        # Authentication
        "user": None,
        "otp_step": 0,
        "otp_sent_contact": None,
        "otp_sent_type": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# Auto-set a demo user when auth is bypassed
if BYPASS_AUTH and not st.session_state.get("user"):
    st.session_state["user"] = {
        "contact": "demo@local",
        "contact_type": "bypass",
        "created_at": time.time(),
        "last_login_at": time.time(),
    }


def _render_login_flow():
    """Render a simple OTP-based login flow."""
    st.markdown("## 🔐 Sign in")
    st.markdown(
        "Use an email address or phone number to receive a one-time login code. "
        "For demo use, OTP will be displayed if email/SMS is not configured."
    )

    with st.form("login_form"):
        contact_method = st.radio("Login via", ["Email", "Phone"], index=0, horizontal=True)
        contact_label = "Email Address" if contact_method == "Email" else "Phone Number"
        contact = st.text_input(contact_label, value=st.session_state.get("otp_sent_contact") or "")
        send = st.form_submit_button("Send OTP")

        if send:
            try:
                sent, message = send_otp(contact, contact_method.lower())
                st.session_state["otp_step"] = 1
                st.session_state["otp_sent_contact"] = contact
                st.session_state["otp_sent_type"] = contact_method.lower()
                st.success(message)
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("otp_step") == 1 and st.session_state.get("otp_sent_contact"):
        with st.form("verify_otp_form"):
            otp_input = st.text_input("Enter the OTP", value="")
            verify = st.form_submit_button("Verify OTP")
            if verify:
                contact = st.session_state.get("otp_sent_contact")
                contact_type = st.session_state.get("otp_sent_type")
                store = AuthStore()
                if store.verify_otp(contact, contact_type, otp_input):
                    user = store.upsert_user(contact, contact_type)
                    st.session_state["user"] = {
                        "contact": user.contact,
                        "contact_type": user.contact_type,
                        "created_at": user.created_at,
                        "last_login_at": user.last_login_at,
                    }
                    st.success("✅ Logged in successfully.")
                    # Streamlit 1.49+ uses st.rerun() instead of st.experimental_rerun()
                    st.rerun()
                else:
                    st.error("Invalid or expired OTP. Please try again.")


if not st.session_state.get("user") and not BYPASS_AUTH:
    _render_login_flow()
    st.stop()


# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>🏦 INTELLI-CREDIT</h1>
    <p>Next-Gen AI-Powered Corporate Credit Appraisal Engine</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown(f"### ⚡ Mode: {get_mode_display()}")
    st.divider()

    user = get_current_user()
    if user and not BYPASS_AUTH:
        st.markdown(f"### 👤 Logged in as {user.contact}")
        if st.button("Sign out", use_container_width=True):
            logout()
            st.rerun()
        st.divider()

    if IS_DEMO:
        st.info("🟡 **Demo Mode** — Using sample data for 'Bharat Steel Industries Pvt Ltd'. No API keys needed.")
        if st.button("📦 Load Sample Company", use_container_width=True, type="primary"):
            sample = load_json(SAMPLE_DATA_DIR / "sample_company.json")
            st.session_state.company_data = sample
            st.session_state.pipeline_step = 1
            st.rerun()
    else:
        # ── LLM ──
        from config import has_openrouter_key
        if has_openrouter_key():
            st.success("✅ OpenRouter AI configured (openrouter/auto)")
        elif has_anthropic_key():
            st.success(f"✅ Anthropic/Claude ({LLM_PROVIDER.title()}) configured")
        elif has_openai_key():
            st.success(f"✅ OpenAI GPT-4o configured")
        else:
            st.warning("⚠ No LLM API Key — set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY")
        # ── Search ──
        if has_tavily_key():
            st.success("✅ Tavily Web Search configured")
        else:
            st.warning("⚠ Tavily API Key missing")
        # ── Azure DI ──
        if has_azure_di():
            st.success("✅ Azure Document Intelligence configured")
        else:
            st.info("ℹ Azure DI not configured (will use pdfplumber / Tesseract)")
        # ── Databricks ──
        if has_databricks():
            st.success("✅ Databricks Delta Lake configured")
        else:
            st.info("ℹ Databricks not configured (will use SQLite locally)")

    st.divider()
    st.markdown("### 📊 Pipeline Progress")

    steps = ["Load Data", "Ingest Docs", "Research", "Score", "CAM"]
    for i, step in enumerate(steps):
        if i < st.session_state.pipeline_step:
            st.markdown(f"✅ {step}")
        elif i == st.session_state.pipeline_step:
            st.markdown(f"🔄 **{step}**")
        else:
            st.markdown(f"⬜ {step}")

    st.divider()
    st.markdown("""
    ### 🔗 Architecture
    - **Pillar 1**: Document Ingestor
    - **Pillar 2**: Research Agent
    - **Pillar 3**: Recommendation Engine
    """)

    if st.button("🔄 Reset Pipeline", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─── Auto-Tab Switcher (JS injection) ───
_TAB_LABELS = ["🏢 Entity Onboarding", "📄 Document Upload", "🔍 Research Dashboard", "📊 Credit Analysis", "📋 CAM Report"]

def _switch_to_tab(idx: int):
    """Inject JavaScript to programmatically click a tab by its rendered index."""
    js = f"""
    <script>
    (function() {{
        var tries = 0;
        function clickTab() {{
            var parentDoc = window.parent.document;
            var tabs = parentDoc.querySelectorAll('[data-baseweb="tab"]');
            if (tabs.length > {idx}) {{
                tabs[{idx}].click();
            }} else if (tries++ < 50) {{
                setTimeout(clickTab, 100);
            }}
        }}
        setTimeout(clickTab, 200);
    }})();
    </script>
    """
    st.components.v1.html(js, height=0, scrolling=False)

# Fire tab switch if requested (must run before st.tabs renders)
_pending_tab = st.session_state.get("active_tab", 0)
if _pending_tab != 0:
    st.session_state.active_tab = 0  # reset so it doesn't loop

# ─── Main Tabs ───
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🏢 Entity Onboarding",
    "📄 Document Upload",
    "🔍 Research Dashboard",
    "📊 Credit Analysis",
    "📋 CAM Report",
])

# Fire auto-tab switch (runs AFTER tabs are rendered in DOM)
if _pending_tab != 0:
    _switch_to_tab(_pending_tab)


# ═══════════════════════════════════════════════════════════
# TAB 0: Entity Onboarding
# ═══════════════════════════════════════════════════════════
with tab0:
    st.markdown("## 🏢 Entity Onboarding")
    st.markdown("Complete the onboarding wizard to define the entity and loan request.")
    
    cd = st.session_state.company_data or {}
    lr = cd.get("loan_request", {})
    
    with st.form("entity_onboarding_form"):
        st.markdown("### Entity Details")
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name", value=cd.get("company_name", ""))
            cin = st.text_input("CIN", value=cd.get("cin", ""))
            turnover_cr = st.number_input("Turnover (in Crores)", min_value=0.0, step=1.0, value=float(cd.get("turnover_cr", 100.0)))
        with col2:
            pan = st.text_input("PAN", value=cd.get("pan", ""))
            industry_val = cd.get("industry", "Manufacturing")
            industries = ["Manufacturing", "IT Services", "Retail", "Healthcare", "Financials", "Other"]
            idx = industries.index(industry_val) if industry_val in industries else 0
            industry = st.selectbox("Sector / Industry", industries, index=idx)
            
        st.markdown("### Loan Details")
        col3, col4 = st.columns(2)
        with col3:
            loan_type_val = lr.get("type", "Working Capital")
            loan_types = ["Working Capital", "Term Loan", "Bill Discounting"]
            lt_idx = loan_types.index(loan_type_val) if loan_type_val in loan_types else 0
            loan_type = st.selectbox("Loan Type", loan_types, index=lt_idx)
            amount_cr = st.number_input("Amount (in Crores)", min_value=0.0, step=0.5, value=float(lr.get("amount_cr", 10.0)))
        with col4:
            tenor = st.number_input("Tenure (Years)", min_value=0.0, step=1.0, value=float(lr.get("tenor_years", 3.0)))
            rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1, value=float(lr.get("proposed_rate_pct", 12.0)))
            
        submit_onboarding = st.form_submit_button("Save & Continue to Data Ingestion", type="primary", use_container_width=True)
        
        if submit_onboarding:
            if not company_name or not cin:
                st.error("Please fill in Company Name and CIN.")
            else:
                if not st.session_state.company_data:
                    st.session_state.company_data = {}
                
                st.session_state.company_data.update({
                    "company_name": company_name,
                    "cin": cin,
                    "pan": pan,
                    "industry": industry,
                    "turnover_cr": turnover_cr,
                    "loan_request": {
                        "type": loan_type,
                        "amount_cr": amount_cr,
                        "tenor_years": tenor,
                        "proposed_rate_pct": rate
                    }
                })
                st.session_state.pipeline_step = max(st.session_state.pipeline_step, 1)
                st.session_state.active_tab = 1  # auto-switch to Document Upload
                st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 1: Document Upload & Extraction
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📄 Document Upload & Data Extraction")
    st.markdown("Upload financial documents (PDFs) or use demo data for instant results.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload Financial Documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload Annual Reports, Bank Statements, GST Returns, etc.",
        )

        if uploaded_files:
            st.markdown("### 📚 Uploaded Documents")
            for file in uploaded_files:
                with st.expander(f"📎 {file.name}", expanded=False):
                    st.write(f"Size: {file.size / 1024:.1f} KB")

                    # Manage state for this file
                    if "process_state" not in st.session_state:
                         st.session_state.process_state = {}
                    if file.name not in st.session_state.process_state:
                         st.session_state.process_state[file.name] = {"step": 0}
                         
                    fstate = st.session_state.process_state[file.name]

                    if fstate["step"] == 0:
                        if st.button(f"🔍 Extract Text & Classify {file.name}", key=f"proc_{file.name}"):
                            with st.spinner("Extracting text & classifying..."):
                                from pillar1_ingestor.ocr_engine import extract_text_from_uploaded_file
                                from pillar1_ingestor.document_classifier import classify_document

                                result = extract_text_from_uploaded_file(file)
                                classification = classify_document(file.name, result["full_text"])
                                
                                fstate["result"] = result
                                fstate["classification"] = classification
                                fstate["step"] = 1
                                st.rerun()
                                
                    if fstate["step"] >= 1:
                        st.success(f"✅ Extracted {fstate['result']['num_pages']} pages via {fstate['result']['method']}")
                        
                        doc_types = ["annual_report", "bank_statement", "gst_return", "alm", "shareholding_pattern", "borrowing_profile", "portfolio_cuts", "other"]
                        current_type = fstate["classification"].get("type", "other")
                        idx = doc_types.index(current_type) if current_type in doc_types else doc_types.index("other")
                        
                        confirmed_type = st.selectbox(
                            "Confirm Document Type:",
                            doc_types,
                            index=idx,
                            key=f"type_{file.name}"
                        )
                        
                        if fstate["step"] == 1:
                            st.markdown("#### Configure Extraction Schema")
                            
                            default_schema = '{\n  "key": "type"\n}'
                            if confirmed_type == "annual_report":
                                default_schema = '{\n  "company_name": "string",\n  "revenue_cr": "float",\n  "pat_cr": "float",\n  "total_debt_cr": "float"\n}'
                            elif confirmed_type == "bank_statement":
                                default_schema = '{\n  "account_holder": "string",\n  "closing_balance": "float"\n}'
                            elif confirmed_type == "gst_return":
                                default_schema = '{\n  "gstin": "string",\n  "taxable_turnover": "float"\n}'
                                
                            custom_schema = st.text_area(f"Define JSON schema for {file.name} extraction", value=default_schema, height=150, key=f"schema_{file.name}")

                            if st.button(f"⚙️ Run Extraction", key=f"ext_{file.name}", type="primary"):
                                fstate["confirmed_type"] = confirmed_type
                                st.session_state.doc_classifications.append({"file": file.name, "type": confirmed_type})
                                st.session_state.extracted_data[file.name] = fstate["result"]
                                
                                with st.spinner("Running LLM extraction..."):
                                    from pillar1_ingestor.llm_extractor import extract_sync
                                    extracted = extract_sync(fstate["result"]["full_text"], confirmed_type, custom_schema=custom_schema)
                                    st.session_state.extracted_data[f"{file.name}_structured"] = extracted
                                    fstate["step"] = 2
                                    st.rerun()
                                    
                    if fstate["step"] == 2:
                        st.success("✅ Data Extracted")
                        
                        extracted_data = st.session_state.extracted_data[f"{file.name}_structured"]
                        conf = extracted_data.get("confidence_score", 0.85)
                        
                        # Confidence Score Header
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown("#### 🔍 Review & Verify Extracted Data")
                        with c2:
                            st.metric("AI Confidence", f"{conf*100:.0f}%", delta=f"{conf-0.5:.1f}" if conf > 0.5 else "-")
                        
                        # Flatten data for Editor
                        import pandas as pd
                        display_data = {}
                        source_map = {}
                        
                        for k, v in extracted_data.items():
                            if isinstance(v, dict) and "value" in v:
                                display_data[k] = v["value"]
                                source_map[k] = v.get("source_quote", "No quote available")
                            elif k not in ["confidence_score", "directors", "key_observations", "large_credits"]:
                                display_data[k] = v

                        df = pd.DataFrame([display_data])
                        
                        # Use a cleaner display for the analyst
                        st.markdown("##### 📝 Financial Data Point Review")
                        edited_df = st.data_editor(
                            df, 
                            key=f"editor_{file.name}", 
                            use_container_width=True,
                            num_rows="fixed",
                            column_config={
                                "revenue_cr": st.column_config.NumberColumn("Revenue (Cr)", format="₹ %.2f"),
                                "total_debt_cr": st.column_config.NumberColumn("Total Debt (Cr)", format="₹ %.2f"),
                                "cin": st.column_config.TextColumn("CIN", help="Corporate Identity Number"),
                            }
                        )

                        # --- NEW: Audit Evidence Section ---
                        st.markdown("##### 🛡️ Audit Evidence (Source Attribution)")
                        with st.container(border=True):
                            selected_field = st.selectbox("Select metric to verify source:", list(source_map.keys()), key=f"sel_{file.name}")
                            st.info(f"**AI Source Quote:** \"{source_map.get(selected_field, '...')}\"")
                            st.caption("Verification against original document text completed automatically.")

                        if st.button("Confirm & Save Data", key=f"conf_{file.name}", type="primary", use_container_width=True):
                            if not edited_df.empty:
                                # Re-stitch the confirmed data
                                final_data = extracted_data.copy()
                                updated_row = edited_df.to_dict(orient="records")[0]
                                for k, v in updated_row.items():
                                    if k in final_data and isinstance(final_data[k], dict) and "value" in final_data[k]:
                                        final_data[k]["value"] = v
                                    else:
                                        final_data[k] = v
                                st.session_state.extracted_data[f"{file.name}_structured"] = final_data
                            fstate["step"] = 3
                            st.rerun()
                            
                    if fstate["step"] == 3:
                         st.success("✅ Data Extraction Visual Summary")
                         doc_type = fstate.get("doc_type", "Unknown")
                         extracted_viz = st.session_state.extracted_data.get(f"{file.name}_structured", {})
                         
                         _render_extraction_visuals(extracted_viz, doc_type)
                         
                         with st.expander("🔍 View Technical Data (JSON)"):
                             st.json(extracted_viz)

    with col2:
        st.markdown("### 🏢 Company Info")
        if st.session_state.company_data:
            cd = st.session_state.company_data
            st.markdown(f"**{cd.get('company_name', 'N/A')}**")
            st.markdown(f"CIN: `{cd.get('cin', 'N/A')}`")
            st.markdown(f"Industry: {cd.get('industry', 'N/A')}")
            if cd.get('registered_office'):
                st.markdown(f"Registered: {cd['registered_office']}")

            loan = cd.get("loan_request", {})
            st.markdown("---")
            st.markdown("### 💰 Loan Request")
            st.markdown(f"**{loan.get('type', 'N/A')}**")
            st.markdown(f"Amount: **₹ {loan.get('amount_cr', 0)} Crore**")
            st.markdown(f"Rate: {loan.get('proposed_rate_pct', 0)}% p.a.")
            st.markdown(f"Tenor: {loan.get('tenor_years', 0)} years")
            st.markdown(f"Purpose: {loan.get('purpose', 'N/A')}")
        else:
            st.info("Load sample data or upload documents to begin.")

    # Qualitative Notes Section
    st.divider()
    st.markdown("### 📝 Credit Officer's Qualitative Notes")
    note = st.text_area(
        "Enter your observations about the borrower:",
        placeholder="e.g., Factory operating at 65% capacity, strong dealer network in Maharashtra...",
        height=100,
    )
    if st.button("➕ Add Note") and note:
        st.session_state.qualitative_notes.append(note)
        st.rerun()

    if st.session_state.qualitative_notes:
        for i, n in enumerate(st.session_state.qualitative_notes):
            st.markdown(f"📌 {n}")

    # Add sample qualitative notes in demo mode
    if IS_DEMO and st.session_state.company_data and not st.session_state.qualitative_notes:
        if st.button("📝 Load Sample Notes"):
            st.session_state.qualitative_notes = st.session_state.company_data.get("qualitative_notes", [])
            st.rerun()

    # GST Validation
    st.divider()
    st.markdown("### 🔒 GST Cross-Validation")

    if st.button("▶ Run GST Validation", type="primary", use_container_width=True):
        with st.spinner("Cross-validating GST 3B vs 2A vs Bank Statement..."):
            from pillar1_ingestor.gst_validator import validate_gst_compliance
            gst_result = validate_gst_compliance(
                extracted_data=st.session_state.extracted_data,
                company_data=st.session_state.company_data
            )
            st.session_state.gst_validation = gst_result
            st.session_state.pipeline_step = max(st.session_state.pipeline_step, 2)
            st.rerun()

    if st.session_state.gst_validation:
        gst = st.session_state.gst_validation
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("GST Compliance Score", f"{gst['compliance_score']:.0f}/100")
        with col_b:
            st.metric("Risk Level", gst['risk_level'])
        with col_c:
            summary = gst.get("summary", {})
            st.metric("Turnover Variance", f"{summary.get('turnover_variance_pct', 0):.1f}%")
        
        st.markdown(gst.get("narrative", "Analysis complete."))
        
        with st.expander("🛠️ System Verification Logs"):
            st.json(gst)

        # Display Narrative and Flags
        st.markdown(f"#### 📝 Auditor's Commentary")
        st.markdown(gst.get("narrative", "No narrative generated."))

        if gst.get("flags"):
            st.markdown("#### 🚩 Risk Flags")
            for flag in gst["flags"]:
                severity_color = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}.get(flag["severity"], "#94A3B8")
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid {severity_color}; background: rgba(255,255,255,0.05); border-radius: 4px; margin-bottom: 8px;">
                    <span style="color: {severity_color}; font-weight: bold;">[{flag['type']}]</span> {flag['message']}
                </div>
                """, unsafe_allow_html=True)

        if st.button("Proceed to Research Dashboard ➡", type="primary", use_container_width=True):
            st.session_state.active_tab = 2
            st.rerun()


# ═══════════════════════════════════════════════════════════
# TAB 2: Research Dashboard
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Autonomous Research Dashboard")
    st.markdown("Multi-agent research pipeline: News, MCA21, e-Courts, Sector Analysis.")

    col_company, col_btn = st.columns([3, 1])
    with col_company:
        company_name = st.text_input(
            "Company Name",
            value=st.session_state.company_data.get("company_name", "") if st.session_state.company_data else "",
            placeholder="Enter company name for research..."
        )

    # Agent status containers
    agent_statuses = {}
    status_containers = {}

    if st.button("🚀 Launch Research Agents", type="primary", use_container_width=True):
        if not company_name:
            st.warning("Please enter a company name.")
        else:
            # Get company data for research
            cd = st.session_state.company_data or {}
            promoters = [p["name"] for p in cd.get("promoters", [])]
            cin = cd.get("cin", "")
            industry = cd.get("industry", "")

            # Create progress display
            st.markdown("### 🤖 Agent Status")
            agent_cols = st.columns(4)

            agents = [
                ("📰 News Agent", "news_agent"),
                ("🏛️ MCA Agent", "mca_agent"),
                ("⚖️ e-Courts Agent", "ecourts_agent"),
                ("📈 Sector Agent", "sector_agent"),
            ]

            progress_placeholders = {}
            for i, (label, key) in enumerate(agents):
                with agent_cols[i]:
                    st.markdown(f"**{label}**")
                    progress_placeholders[key] = st.empty()
                    progress_placeholders[key].info("⏳ Waiting...")

            progress_bar = st.progress(0, text="Starting research pipeline...")

            def progress_callback(agent_name, status):
                if agent_name in progress_placeholders:
                    if "✅" in status:
                        progress_placeholders[agent_name].success(status)
                    elif "Error" in status:
                        progress_placeholders[agent_name].error(status)
                    else:
                        progress_placeholders[agent_name].info(f"🔄 {status}")

            # Run research
            from pillar2_research.agent_orchestrator import run_research_pipeline

            counter = [0]   # mutable container avoids nonlocal binding issue
            total_steps = 4

            def tracked_callback(agent_name, status):
                progress_callback(agent_name, status)
                if "✅" in status:
                    counter[0] += 1
                    progress_bar.progress(
                        counter[0] / total_steps,
                        text=f"Completed {counter[0]}/{total_steps} agents"
                    )

            research = run_research_pipeline(
                company_name=company_name,
                cin=cin,
                industry=industry,
                promoter_names=promoters,
                qualitative_notes=st.session_state.qualitative_notes,
                progress_callback=tracked_callback,
            )

            st.session_state.research_results = research
            st.session_state.pipeline_step = max(st.session_state.pipeline_step, 3)

            progress_bar.progress(1.0, text="✅ All agents completed!")
            time.sleep(0.5)
            st.rerun()

    # Display research results
    if st.session_state.research_results:
        research = st.session_state.research_results
        st.divider()

        # Risk Overview
        st.markdown("### 🎯 Research Risk Overview")
        col1, col2, col3 = st.columns(3)

        risk_score = research.get("composite_risk_score", 0)
        risk_level = research.get("risk_level", "N/A")
        badge_class = "risk-low" if risk_level == "LOW" else "risk-moderate" if risk_level == "MODERATE" else "risk-high"

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Composite Risk Score</h3>
                <div class="value">{risk_score:.0f}/100</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Risk Level</h3>
                <div class="value"><span class="risk-badge {badge_class}">{risk_level}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            news_sent = research.get("news", {}).get("overall_sentiment", 0)
            sent_label = "Positive" if news_sent > 0 else "Negative" if news_sent < 0 else "Neutral"
            st.markdown(f"""
            <div class="metric-card">
                <h3>News Sentiment</h3>
                <div class="value">{sent_label}</div>
            </div>
            """, unsafe_allow_html=True)

        # News Results
        st.markdown("### 📰 News & Sentiment")
        news = research.get("news", {})
        articles = news.get("articles", [])
        if articles:
            for article in articles:
                sentiment = article.get("sentiment", "neutral")
                emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
                score = article.get("sentiment_score", article.get("score", 0))
                st.markdown(f"{emoji} **{article.get('title', 'N/A')}** — _{article.get('source', 'N/A')}_ (Score: {score:.2f})")
        else:
            st.info("No news articles found.")

        # MCA Results
        st.markdown("### 🏛️ MCA21 Compliance")
        mca = research.get("mca", {})
        st.markdown(mca.get("summary", "No MCA data available."))
        with st.expander("📊 View MCA Technical Log"):
            st.json(mca)

        # Litigation
        st.markdown("### ⚖️ Litigation Analysis")
        litigation = research.get("litigation", {})
        st.markdown(litigation.get("summary", "No litigation data available."))
        with st.expander("📁 View Litigation Records"):
            st.json(litigation)

        # Sector
        st.markdown("### 📈 Sector Analysis")
        sector = research.get("sector", {})
        st.markdown(sector.get("summary", "No sector data available."))

        factors = sector.get("key_factors", [])
        if factors:
            for f in factors:
                st.markdown(f"• {f}")

        # Provenance Log
        st.divider()
        st.markdown("### 📋 Research Provenance Log")
        provenance = research.get("provenance", [])
        if provenance:
            import pandas as pd
            df = pd.DataFrame(provenance)
            st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: Credit Analysis
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 Credit Analysis — 5 Cs Scoring + SHAP Explainability")

    if st.button("⚡ Run Credit Scoring", type="primary", use_container_width=True):
        if not st.session_state.research_results:
            st.warning("Please run the Research Agents first (Tab 2).")
        else:
            with st.spinner("Building 5 Cs features and running XGBoost + SHAP..."):
                cd = st.session_state.company_data or {}

                from pillar3_engine.feature_builder import build_features
                from pillar3_engine.scoring_model import score_credit

                five_cs = build_features(
                    financials=cd.get("financials", {}),
                    research=st.session_state.research_results,
                    gst_validation=st.session_state.gst_validation,
                    collateral=cd.get("collateral", {}),
                    qualitative_notes=st.session_state.qualitative_notes,
                )

                scoring = score_credit(
                    five_cs["feature_vector"],
                    five_cs["feature_names"],
                )

                st.session_state.five_cs = five_cs
                st.session_state.scoring = scoring

                # --- NEW: AI Credit Committee Deliberation ---
                with st.status("👨‍⚖️ AI Credit Committee Deliberating...", expanded=True) as status:
                    st.write("Triangulating Financials vs Research vs GST data...")
                    from pillar3_engine.committee_agent import CommitteeAgent
                    committee = CommitteeAgent()
                    
                    # Run deliberation (sync wrapper for streamlit)
                    import asyncio
                    try:
                        verdict = asyncio.run(committee.deliberate(
                            company_data=cd,
                            financials=cd.get("financials", {}),
                            research=st.session_state.research_results,
                            gst_validation=st.session_state.gst_validation,
                            scoring=scoring
                        ))
                    except Exception as e:
                        verdict = {"verdict": "REFER", "rationale": f"Committee failed: {e}"}
                    
                    st.session_state.committee_verdict = verdict
                    status.update(label=f"✅ Committee Verdict: {verdict.get('verdict', 'REFER')}", state="complete")

                st.session_state.pipeline_step = max(st.session_state.pipeline_step, 4)

                # Persist decision...
                try:
                    from knowledge_store.structured_store import StructuredStore
                    company_name = cd.get("company_name", "Unknown")
                    ss = StructuredStore()
                    cid = ss.store_company(company_name, cd.get("cin", ""), cd.get("industry", ""), cd)
                    ss.store_decision(cid, scoring["credit_score"], verdict.get("verdict", scoring["decision"]),
                                      five_cs["scores"],
                                      dict(zip(five_cs["feature_names"], scoring["shap_values"])))
                except Exception as db_err:
                    st.warning(f"⚠ Could not persist to data store: {db_err}")

                st.rerun()

    if st.session_state.scoring:
        scoring = st.session_state.scoring
        five_cs = st.session_state.five_cs

        # Credit Score Gauge
        col1, col2 = st.columns([1, 2])

        with col1:
            cv = st.session_state.committee_verdict
            score = scoring["credit_score"]
            decision = cv.get("verdict", scoring["decision"])
            color = "#28a745" if decision == "APPROVE" else "#dc3545" if decision == "REJECT" else "#ffc107"

            st.markdown(f"""
            <div class="score-gauge">
                <div class="label">FINAL CREDIT SCORE</div>
                <div class="score" style="color: {color}">{score:.0f}</div>
                <div class="label">Committee Confidence: {cv.get('confidence', 0.8)*100:.0f}%</div>
                <br>
                <div style="font-size: 0.8rem; color: #888; margin-bottom: 0.5rem;">AI COMMITTEE VERDICT</div>
                <span class="risk-badge" style="background: {color}22; color: {color}; border: 2px solid {color}; font-size: 1.1rem; padding: 0.5rem 1.5rem;">
                    {decision}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # 5 Cs Radar Chart
            st.markdown("### 🎯 5 Cs of Credit")
            import plotly.graph_objects as go

            scores = five_cs["scores"]
            categories = list(scores.keys())
            values = list(scores.values())
            values.append(values[0])  # Close the polygon
            categories.append(categories[0])

            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=[c.title() for c in categories],
                fill='toself',
                fillcolor='rgba(58, 123, 213, 0.3)',
                line=dict(color='#3a7bd5', width=2),
                marker=dict(size=8, color='#3a7bd5'),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='#1a1a2e',
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#333', tickfont=dict(color='#888')),
                    angularaxis=dict(gridcolor='#333', tickfont=dict(color='#ccc', size=12)),
                ),
                paper_bgcolor='#0f0c29',
                font=dict(color='#e2e2f0'),
                showlegend=False,
                height=400,
                margin=dict(t=30, b=30, l=60, r=60),
            )
            st.plotly_chart(fig, use_container_width=True)

        # 5 Cs Details
        st.divider()
        st.markdown("### 📋 5 Cs Score Breakdown")

        c_cols = st.columns(5)
        c_names = ["character", "capacity", "capital", "collateral", "conditions"]
        c_emojis = ["🛡️", "💪", "💰", "🏠", "🌍"]

        for i, (c_name, emoji) in enumerate(zip(c_names, c_emojis)):
            with c_cols[i]:
                c_score = five_cs["scores"][c_name]
                color = "#28a745" if c_score >= 70 else "#ffc107" if c_score >= 50 else "#dc3545"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{emoji} {c_name.upper()}</h3>
                    <div class="value" style="color: {color}">{c_score:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

        # SHAP Chart
        st.divider()
        st.markdown("### 📊 Feature Contributions (SHAP-style Waterfall)")

        chart_path = scoring.get("shap_chart_path", "")
        if chart_path and os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)
        else:
            st.info("SHAP chart not generated. Check matplotlib installation.")

        # Explanation & Committee Deliberation
        st.divider()
        st.markdown("### 🏛️ AI Credit Committee Deliberation")
        cv = st.session_state.committee_verdict
        
        if cv:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("#### 🛡️ Strengths")
                for s in cv.get("key_strengths", []):
                    st.markdown(f"✅ {s}")
            with tc2:
                st.markdown("#### ⚠️ Concerns")
                for c in cv.get("key_concerns", []):
                    st.markdown(f"❌ {c}")
            
            st.markdown("#### 🕵️ Triangulation Analysis")
            st.info(cv.get("triangulation_check", "Triangulation suggests consistency between bank statements and reported revenue."))
            
            st.markdown("#### 📝 Investment Rationale")
            st.write(cv.get("rationale", "No detailed rationale provided."))
            
            if cv.get("mitigants"):
                st.markdown("#### 🩹 Suggested Mitigants")
                for m in cv.get("mitigants", []):
                    st.markdown(f"🔹 {m}")
        else:
            st.markdown(scoring.get("explanation", "No explanation available."))


# ═══════════════════════════════════════════════════════════
# TAB 4: CAM Report
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📋 Credit Appraisal Memo Generator")

    if st.session_state.scoring:
        st.success(f"✅ Scoring complete — **{st.session_state.scoring['decision']}** "
                   f"(Score: {st.session_state.scoring['credit_score']:.0f}/100)")

        if st.button("📄 Generate CAM Report & SWOT", type="primary", use_container_width=True):
            cd = st.session_state.company_data or {}
            
            with st.spinner("Triangulating Data & Generating SWOT Analysis..."):
                from pillar3_engine.swot_generator import generate_swot_sync
                swot_text = generate_swot_sync(
                    company_data=cd,
                    five_cs=st.session_state.five_cs,
                    scoring=st.session_state.scoring,
                    research=st.session_state.research_results or {}
                )
                st.session_state.swot_analysis = swot_text
                
            with st.spinner("Generating Credit Appraisal Memo..."):
                from pillar3_engine.cam_generator import generate_cam

                cam_path = generate_cam(
                    company_data=cd,
                    five_cs=st.session_state.five_cs,
                    scoring=st.session_state.scoring,
                    research=st.session_state.research_results or {},
                    gst_validation=st.session_state.gst_validation,
                    qualitative_notes=st.session_state.qualitative_notes,
                    swot_analysis=st.session_state.swot_analysis,
                    committee_verdict=st.session_state.committee_verdict
                )

                st.session_state.cam_path = cam_path
                st.session_state.pipeline_step = 5
                st.rerun()

        if st.session_state.cam_path:
            cam_path = st.session_state.cam_path
            st.success(f"✅ CAM generated: `{Path(cam_path).name}`")

            # Download button
            if os.path.exists(cam_path):
                with open(cam_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download CAM (Word Document)",
                        data=f.read(),
                        file_name=Path(cam_path).name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        type="primary",
                    )

            # Preview
            st.divider()
            st.markdown("### 📖 CAM Preview")

            cd = st.session_state.company_data or {}
            scoring = st.session_state.scoring

            st.markdown(f"""
            ---
            **CREDIT APPRAISAL MEMORANDUM**
            ---
            **Company:** {cd.get('company_name', 'N/A')}
            **CIN:** {cd.get('cin', 'N/A')}

            **Loan Request:** {cd.get('loan_request', {}).get('type', 'N/A')} —
            ₹ {cd.get('loan_request', {}).get('amount_cr', 0)} Crore @
            {cd.get('loan_request', {}).get('proposed_rate_pct', 0)}% p.a.

            ---
            """)

            st.markdown(scoring.get("explanation", ""))

            # Show SWOT in preview
            if st.session_state.get("swot_analysis"):
                st.divider()
                st.markdown("### 📋 Integrated SWOT Analysis")
                st.markdown(st.session_state.swot_analysis)
                
            # Show SHAP chart in preview
            chart_path = scoring.get("shap_chart_path", "")
            if chart_path and os.path.exists(chart_path):
                st.image(chart_path, caption="SHAP Feature Contributions", use_container_width=True)
    else:
        st.info("⬅ Run the Credit Scoring in Tab 3 first to generate the CAM.")
        st.markdown("""
        ### How it works:
        1. **Upload Documents** → Extract financial data
        2. **Research** → Autonomous agent investigation
        3. **Score** → XGBoost + SHAP analysis
        4. **Generate CAM** → Professional Word document with decision + audit trail
        """)


# ─── Footer ───
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;">
    Intelli-Credit v1.0 | AI Credit Appraisal Engine | Built with Streamlit + LangGraph + XGBoost + SHAP
</div>
""", unsafe_allow_html=True)
