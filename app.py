"""
AI-Powered Personal Finance Analyzer
=====================================
Main Streamlit dashboard integrating all modules:
  data_analysis -> ml_predictor -> ai_advisor

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import pdfplumber
import pikepdf

from data_analysis import ExpenseAnalyzer
from ml_predictor import ExpensePredictor
from ai_advisor import FinancialAdvisor
from generate_sample_data import generate_expenses

# ================================================================
# PDF PARSER  —  Supports password-protected bank statements
# ================================================================
CATEGORY_MAP = {
    "Food":          ["zomato","swiggy","dominos","pizza","haldiram","blinkit",
                      "jiomart","food forum","cooks corner","nescafe","magicpin",
                      "sirohi chap","krishnasweets","dunzo"],
    "Travel":        ["delhi metro","metro qr","metro rai","rapido","ola","uber",
                      "irctc","redbus","paytm travel","uttrakhand","railway","bus"],
    "Shopping":      ["amazon","flipkart","myntra","mr diy","smart bazaar",
                      "meesho","nykaa","ajio","geeta traders","rainbow super"],
    "Bills":         ["airtel","jio","bsnl","vodafone","recharge","prepaid",
                      "electricity","blue dart","cashfree"],
    "Entertainment": ["netflix","hotstar","spotify","bookmyshow","pvr"],
    "Health":        ["pharma","medical","hospital","clinic","apollo",
                      "1mg","netmeds","panchtirthi"],
}

def detect_category(description: str) -> str:
    desc = description.lower()
    for category, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in desc:
                return category
    return "Others"

def unlock_pdf(pdf_bytes: bytes, password: str) -> io.BytesIO:
    """Remove password protection from PDF and return unlocked bytes."""
    with pikepdf.open(io.BytesIO(pdf_bytes), password=password) as pdf:
        out = io.BytesIO()
        pdf.save(out)
        out.seek(0)
        return out

def parse_kotak_pdf(pdf_bytes: bytes, password: str = "") -> pd.DataFrame:
    """
    Parse Kotak Bank PDF (with or without password).
    Returns DataFrame with Date, Category, Amount columns.
    """
    # Step 1: Unlock if password provided
    if password:
        try:
            pdf_source = unlock_pdf(pdf_bytes, password)
        except pikepdf.PasswordError:
            raise ValueError("Incorrect password. Please check and try again.")
        except Exception as e:
            raise ValueError(f"Could not unlock PDF: {e}")
    else:
        pdf_source = io.BytesIO(pdf_bytes)

    # Step 2: Extract transactions
    rows = []
    with pdfplumber.open(pdf_source) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 6:
                        continue
                    serial = str(row[0]).strip() if row[0] else ""
                    if not serial.isdigit():
                        continue
                    date_str  = str(row[1]).strip() if row[1] else ""
                    desc_str  = str(row[2]).replace("\n", " ").strip() if row[2] else ""
                    debit_str = str(row[4]).strip() if row[4] else "0"
                    debit_clean = debit_str.replace(",", "").replace("₹", "").strip()
                    try:
                        amount = float(debit_clean)
                    except ValueError:
                        amount = 0.0
                    if amount <= 0:
                        continue
                    rows.append({
                        "Date":     date_str,
                        "Category": detect_category(desc_str),
                        "Amount":   int(amount),
                    })

    if not rows:
        raise ValueError("No transactions found in PDF.")

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.strftime("%d-%m-%y")
    return df.reset_index(drop=True)


# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="AI Personal Finance Analyzer",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI-Powered Personal Finance Analyzer")
st.caption("Upload your bank statement and get smart AI-driven financial insights — 100% free, no API key required.")

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("📥 Data Source")

data_source = st.sidebar.radio(
    "Select data source:",
    ["Sample Data (Demo)", "Upload Bank PDF", "Upload CSV"]
)

if data_source == "Upload Bank PDF":
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload Bank Statement (PDF)",
        type=["pdf"],
        help="Supports Kotak, SBI, HDFC, ICICI and most Indian bank statements."
    )

    # Password field
    pdf_password = st.sidebar.text_input(
        "PDF Password (if protected)",
        type="password",
        placeholder="Enter password or leave blank",
        help="Most bank statements are password protected."
    )

    # Bank-wise password hints
    with st.sidebar.expander("🔑 Common Bank Password Formats"):
        st.markdown("""
| Bank | Password Format |
|------|----------------|
| **Kotak** | First 4 letters of name (CAPS) + DOB (DDMMYYYY) |
| **SBI** | Account number |
| **HDFC** | Date of Birth (DDMMYYYY) |
| **ICICI** | Date of Birth (DDMMYYYY) |
| **Axis** | PAN number (CAPS) |
| **Yes Bank** | Date of Birth (DDMMYYYY) |

*Example (Kotak): Name = Swarna Sonam, DOB = 19 Dec 1912 → **SWAR19121912***
        """)

    if uploaded_pdf is not None:
        with st.spinner("Processing PDF..."):
            try:
                pdf_bytes = uploaded_pdf.read()
                raw_df = parse_kotak_pdf(pdf_bytes, password=pdf_password)
                st.sidebar.success(f"✅ {len(raw_df)} transactions extracted!")
            except ValueError as e:
                st.sidebar.error(f"❌ {e}")
                if "password" in str(e).lower() or "Incorrect" in str(e):
                    st.sidebar.warning("💡 Check the password format table above.")
                st.stop()
            except Exception as e:
                st.sidebar.error(f"❌ Unexpected error: {e}")
                st.stop()
    else:
        st.sidebar.info("Upload your bank statement PDF to get started.")
        st.stop()

elif data_source == "Upload CSV":
    uploaded_csv = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain columns: Date, Category, Amount"
    )
    if uploaded_csv is not None:
        raw_df = pd.read_csv(uploaded_csv)
    else:
        st.sidebar.warning("Please upload a CSV file to continue.")
        st.stop()

else:
    raw_df = generate_expenses(start_date="2026-01-01", num_months=6)

st.sidebar.divider()
monthly_income = st.sidebar.number_input(
    "Monthly Income (₹)",
    min_value=0,
    value=50000,
    step=1000,
    help="Used to calculate savings rate and generate recommendations."
)

with st.sidebar.expander("➕ Add Manual Expense"):
    with st.form("manual_entry_form", clear_on_submit=True):
        m_date     = st.date_input("Date")
        m_category = st.selectbox(
            "Category",
            ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Health", "Others"]
        )
        m_amount   = st.number_input("Amount (₹)", min_value=0, step=10)
        submitted  = st.form_submit_button("Add Expense")
        if submitted and m_amount > 0:
            new_row = pd.DataFrame([{
                "Date":     m_date.strftime("%d-%m-%y"),
                "Category": m_category,
                "Amount":   m_amount
            }])
            raw_df = pd.concat([raw_df, new_row], ignore_index=True)
            st.success(f"Added: {m_category} — ₹{m_amount}")

# ================================================================
# CORE PROCESSING
# ================================================================
try:
    analyzer = ExpenseAnalyzer(df=raw_df)
except Exception as e:
    st.error(f"Failed to process data: {e}")
    st.info("Ensure your file contains 'Date', 'Category', and 'Amount' columns.")
    st.stop()

monthly_spend = analyzer.monthly_spend()

predictor = None
if len(monthly_spend) >= 3:
    predictor = ExpensePredictor(monthly_spend)
    predictor.train()

advisor = FinancialAdvisor(analyzer, predictor)

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Charts", "🤖 AI Insights", "💬 Chatbot", "📄 Raw Data"]
)

# ----------------------------------------------------------------
# TAB 1 — OVERVIEW
# ----------------------------------------------------------------
with tab1:
    total_expense       = analyzer.total_spend()
    avg_monthly_expense = monthly_spend.mean() if len(monthly_spend) > 0 else 0
    savings             = monthly_income - avg_monthly_expense
    savings_rate        = (savings / monthly_income * 100) if monthly_income > 0 else 0
    savings_emoji       = "✅" if savings_rate >= 20 else ("⚠️" if savings_rate >= 0 else "🚨")
    bar_pct             = min(max(savings_rate, 0), 100)
    bar_color           = "#27ae60" if savings_rate >= 20 else ("#f39c12" if savings_rate >= 10 else "#e74c3c")

    st.markdown("## Financial Summary")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
        <div style="background:#1e3a5f;border-radius:12px;padding:20px;text-align:center">
            <div style="color:#7eb8f7;font-size:12px;font-weight:600;letter-spacing:1px">MONTHLY INCOME</div>
            <div style="color:white;font-size:26px;font-weight:700;margin-top:6px">₹{monthly_income:,.0f}</div>
            <div style="color:#7eb8f7;font-size:11px;margin-top:4px">💼 Per Month</div>
        </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
        <div style="background:#5f1e1e;border-radius:12px;padding:20px;text-align:center">
            <div style="color:#f7a07e;font-size:12px;font-weight:600;letter-spacing:1px">AVG MONTHLY SPEND</div>
            <div style="color:white;font-size:26px;font-weight:700;margin-top:6px">₹{avg_monthly_expense:,.0f}</div>
            <div style="color:#f7a07e;font-size:11px;margin-top:4px">📉 Average</div>
        </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
        <div style="background:{'#1e5f2a' if savings>=0 else '#5f1e1e'};border-radius:12px;padding:20px;text-align:center">
            <div style="color:{'#7ef7a0' if savings>=0 else '#f7a07e'};font-size:12px;font-weight:600;letter-spacing:1px">MONTHLY SAVINGS</div>
            <div style="color:white;font-size:26px;font-weight:700;margin-top:6px">₹{savings:,.0f}</div>
            <div style="color:{'#7ef7a0' if savings>=0 else '#f7a07e'};font-size:11px;margin-top:4px">{savings_emoji} {savings_rate:.1f}% Rate</div>
        </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
        <div style="background:#3d1e5f;border-radius:12px;padding:20px;text-align:center">
            <div style="color:#c07ef7;font-size:12px;font-weight:600;letter-spacing:1px">TOTAL TRACKED</div>
            <div style="color:white;font-size:26px;font-weight:700;margin-top:6px">₹{total_expense:,.0f}</div>
            <div style="color:#c07ef7;font-size:11px;margin-top:4px">📊 All Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    goal_msg = "🎉 Excellent! You are above the 20% savings target." if savings_rate >= 20 \
        else f"₹{max(monthly_income * 0.2 - savings, 0):,.0f} more savings needed to reach the 20% goal."

    st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:10px;padding:16px">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                <span style="color:#ccc;font-size:13px">💾 Savings Rate Goal (20%)</span>
                <span style="color:white;font-weight:700">{savings_rate:.1f}%</span>
            </div>
            <div style="background:#333;border-radius:6px;height:12px">
                <div style="background:{bar_color};width:{bar_pct}%;height:12px;border-radius:6px"></div>
            </div>
            <div style="color:#888;font-size:11px;margin-top:6px">{goal_msg}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🏷️ Category Breakdown")
        cat_df = pd.DataFrame({
            "Category":   analyzer.category_spend().index,
            "Amount (₹)": [f"₹{v:,.0f}" for v in analyzer.category_spend().values],
            "Share":      [f"{v}%" for v in analyzer.category_percentage().values],
        })
        st.dataframe(cat_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("### 📅 Monthly Breakdown")
        month_df = pd.DataFrame({
            "Month":      [str(m) for m in monthly_spend.index],
            "Amount (₹)": [f"₹{v:,.0f}" for v in monthly_spend.values],
        })
        st.dataframe(month_df, hide_index=True, use_container_width=True)

    if predictor:
        st.markdown("---")
        st.markdown("### 🔮 Next Month Expense Prediction")
        pred = predictor.predict_next_month()
        p1, p2, p3 = st.columns(3)
        for col, label, color, val in [
            (p1, "Linear Regression", "#7eb8f7", pred["linear_regression"]),
            (p2, "Random Forest",     "#7ef7a0", pred["random_forest"]),
            (p3, "Best Estimate ✨",  "#f7c07e", pred["blended_estimate"]),
        ]:
            col.markdown(f"""
                <div style="background:#1a1a2e;border-radius:10px;padding:18px;text-align:center">
                    <div style="color:{color};font-size:12px;font-weight:600">{label}</div>
                    <div style="color:white;font-size:24px;font-weight:700;margin-top:6px">₹{val:,}</div>
                </div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# TAB 2 — CHARTS
# ----------------------------------------------------------------
with tab2:
    st.subheader("Visual Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(analyzer.plot_monthly_bar())
    with col2:
        st.pyplot(analyzer.plot_category_pie())
    st.pyplot(analyzer.plot_category_trend_line())

# ----------------------------------------------------------------
# TAB 3 — AI INSIGHTS
# ----------------------------------------------------------------
with tab3:
    st.subheader("🤖 AI Financial Advisor")
    st.caption("Smart insights automatically generated from your spending patterns — no external API required.")
    insights = advisor.generate_full_report(monthly_income=monthly_income)
    for insight in insights:
        st.info(insight)

# ----------------------------------------------------------------
# TAB 4 — CHATBOT
# ----------------------------------------------------------------
with tab4:
    st.subheader("💬 Finance Chatbot")
    st.caption("Ask anything about your finances in English or Hindi — get instant answers.")

    INTENT_KEYWORDS = {
        "save":     ["save","saving","savings","reduce","cut","bachao","bachat","bachau","bachana"],
        "predict":  ["predict","prediction","next month","forecast","future","agle mahine","agle"],
        "category": ["category","where","highest","most","spent","kaha","kahan","sabse","zyada"],
        "total":    ["total","how much","overall","kitna","kul","poora"],
        "analyze":  ["analyze","analysis","summary","report","overview","dikhao","batao"],
    }

    def get_intent(question: str) -> str:
        q = question.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    return intent
        return "unknown"

    def chatbot_response(question: str, income: float) -> str:
        intent = get_intent(question)

        if intent == "save":
            tips = advisor._savings_insight(income)
            advice = (
                "Here are personalized tips to save more:\n\n"
                "1. 🛍️ Cut Shopping & Entertainment by 20%\n"
                "2. 🍕 Cook at home more — reduce food delivery orders\n"
                "3. 💰 Auto-transfer 20% of salary to savings on payday\n"
                "4. 📱 Cancel unused subscriptions\n"
                "5. 📊 Review your budget at the end of every month"
            )
            return (str(tips[0]) + "\n\n" + advice) if tips else advice

        if intent == "predict":
            if predictor:
                pred = predictor.predict_next_month()
                return (
                    f"🔮 **Next Month Expense Forecast**\n\n"
                    f"**Best Estimate: ₹{pred['blended_estimate']:,}**\n\n"
                    f"• Linear Regression: ₹{pred['linear_regression']:,}\n"
                    f"• Random Forest: ₹{pred['random_forest']:,}\n\n"
                    f"Plan your budget accordingly."
                )
            return "At least 3 months of data is required to generate a prediction."

        if intent == "category":
            top_cat, amount = analyzer.top_category_last_month()
            cat_pct = analyzer.category_percentage()
            breakdown = "\n".join([f"• {c}: {p}%" for c, p in cat_pct.head(5).items()])
            return (
                f"Your highest spending category is **{top_cat}**.\n\n"
                f"Last month: ₹{amount:,.0f} | Overall share: {cat_pct.get(top_cat, 0)}%\n\n"
                f"**Top Categories:**\n{breakdown}"
            )

        if intent == "total":
            return f"Your total tracked expenditure is **₹{analyzer.total_spend():,.0f}**."

        if intent == "analyze":
            report = advisor.generate_full_report(income)
            return "\n\n".join([str(r) for r in report[:3]])

        return (
            "I can answer questions like:\n\n"
            "• **How can I save money?**\n"
            "• **What is my highest spending category?**\n"
            "• **Predict next month's expenses**\n"
            "• **What is my total spending?**\n"
            "• **Analyze my expenses**\n\n"
            "Try any of these!"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your AI Finance Assistant.\n\n"
                "Ask me anything about your spending:\n"
                "• *How can I save money?*\n"
                "• *What is my highest spending category?*\n"
                "• *Predict next month's expenses*"
            )
        }]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask your finance question...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        response = chatbot_response(user_q, monthly_income)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    st.markdown("**Quick Questions:**")
    q1, q2, q3, q4 = st.columns(4)
    quick = [
        (q1, "💰 How to save?",          "How can I save money?"),
        (q2, "📊 Top spending?",          "What is my highest spending category?"),
        (q3, "🔮 Next month prediction?", "Predict next month expenses"),
        (q4, "📋 Full analysis",          "Analyze my expenses"),
    ]
    for col, label, question in quick:
        if col.button(label):
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": chatbot_response(question, monthly_income)
            })
            st.rerun()

# ----------------------------------------------------------------
# TAB 5 — RAW DATA
# ----------------------------------------------------------------
with tab5:
    st.subheader("📄 Raw Transaction Data")
    st.dataframe(
        analyzer.df[["Date", "Category", "Amount"]],
        use_container_width=True,
        hide_index=True
    )
    buf = io.StringIO()
    analyzer.df[["Date", "Category", "Amount"]].to_csv(buf, index=False)
    st.download_button(
        "📥 Download as CSV",
        data=buf.getvalue(),
        file_name="expense_data.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("Built with Python · Pandas · Scikit-learn · Streamlit")