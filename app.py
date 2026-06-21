
import streamlit as st
import pandas as pd
import numpy as np
import io

from data_analysis import ExpenseAnalyzer
from ml_predictor import ExpensePredictor
from ai_advisor import FinancialAdvisor
from generate_sample_data import generate_expenses

# ===============================================================
# PAGE CONFIG
# ===============================================================
st.set_page_config(
    page_title="AI Personal Finance Analyzer",
    page_icon="💰",
    layout="wide"
)

st.title(" AI-Powered Personal Finance Analyzer")
st.caption("Apna expense data upload karo aur AI se smart financial advice paao — 100% free, no API key required.")

# ===============================================================
# SIDEBAR - DATA INPUT
# ===============================================================
st.sidebar.header(" Data Input")

data_source = st.sidebar.radio(
    "Data kahan se loge?",
    ["Use Sample Data (Demo)", "Upload Your CSV"]
)

if data_source == "Upload Your CSV":
    uploaded_file = st.sidebar.file_uploader(
        "CSV upload karo (columns: Date, Category, Amount)", type=["csv"]
    )
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        st.sidebar.warning("Pehle CSV upload karo, ya 'Use Sample Data' select karo.")
        st.stop()
else:
    raw_df = generate_expenses(start_date="2026-01-01", num_months=6)

st.sidebar.divider()
monthly_income = st.sidebar.number_input(
    "Monthly Income (₹)", min_value=0, value=50000, step=1000,
    help="Yeh savings calculation aur AI advice ke liye use hoga"
)

# Manual entry option
with st.sidebar.expander(" Manual Expense Add Karo"):
    with st.form("manual_entry_form", clear_on_submit=True):
        m_date = st.date_input("Date")
        m_category = st.selectbox(
            "Category",
            ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Health", "Others"]
        )
        m_amount = st.number_input("Amount (₹)", min_value=0, step=10)
        submitted = st.form_submit_button("Add Expense")

        if submitted and m_amount > 0:
            new_row = pd.DataFrame([{
                "Date": m_date.strftime("%d-%m-%y"),
                "Category": m_category,
                "Amount": m_amount
            }])
            raw_df = pd.concat([raw_df, new_row], ignore_index=True)
            st.success(f" Added: {m_category} - ₹{m_amount}")

# ===============================================================
# CORE PROCESSING (Step 2, 3, 4 ko yahan call karte hain)
# ===============================================================
try:
    analyzer = ExpenseAnalyzer(df=raw_df)
except Exception as e:
    st.error(f"Data process karne me error aaya: {e}")
    st.info("Confirm karo CSV me 'Date', 'Category', 'Amount' columns hain.")
    st.stop()

monthly_spend = analyzer.monthly_spend()

predictor = None
if len(monthly_spend) >= 3:
    predictor = ExpensePredictor(monthly_spend)
    predictor.train()

advisor = FinancialAdvisor(analyzer, predictor)

# ===============================================================
# TABS
# ===============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [" Overview", " Charts", " AI Insights", " Chatbot", " Raw Data"]
)

# ---------------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------------
with tab1:
    st.subheader("Financial Summary")

    total_expense = analyzer.total_spend()
    avg_monthly_expense = monthly_spend.mean() if len(monthly_spend) > 0 else 0
    savings = monthly_income - avg_monthly_expense

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monthly Income", f"₹{monthly_income:,.0f}")
    col2.metric("Avg Monthly Expense", f"₹{avg_monthly_expense:,.0f}")
    col3.metric(
        "Avg Monthly Savings",
        f"₹{savings:,.0f}",
        delta=f"{(savings/monthly_income*100):.1f}%" if monthly_income > 0 else None
    )
    col4.metric("Total Tracked Expense", f"₹{total_expense:,.0f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Category-wise Spending**")
        cat_df = pd.DataFrame({
            "Category": analyzer.category_spend().index,
            "Amount (₹)": analyzer.category_spend().values,
            "% Share": analyzer.category_percentage().values
        })
        st.dataframe(cat_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**Month-wise Spending**")
        month_df = pd.DataFrame({
            "Month": [str(m) for m in monthly_spend.index],
            "Amount (₹)": monthly_spend.values
        })
        st.dataframe(month_df, hide_index=True, use_container_width=True)

    if predictor:
        st.divider()
        st.markdown("**Next Month Prediction (ML Model)**")
        pred = predictor.predict_next_month()
        p1, p2, p3 = st.columns(3)
        p1.metric("Linear Regression", f"₹{pred['linear_regression']:,}")
        p2.metric("Random Forest", f"₹{pred['random_forest']:,}")
        p3.metric("Blended Estimate", f"₹{pred['blended_estimate']:,}")

# ---------------------------------------------------------------
# TAB 2: CHARTS
# ---------------------------------------------------------------
with tab2:
    st.subheader("Visual Analysis")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(analyzer.plot_monthly_bar())
    with chart_col2:
        st.pyplot(analyzer.plot_category_pie())

    st.pyplot(analyzer.plot_category_trend_line())

# ---------------------------------------------------------------
# TAB 3: AI INSIGHTS (rule-based Gen AI advisor)
# ---------------------------------------------------------------
with tab3:
    st.subheader(" AI Financial Advisor")
    st.caption("Yeh insights aapke data ke patterns se automatically generate hote hain — no external API used.")

    insights = advisor.generate_full_report(monthly_income=monthly_income)
    for insight in insights:
        st.info(insight)

# ---------------------------------------------------------------
# TAB 4: CHATBOT
# ---------------------------------------------------------------
with tab4:
    st.subheader(" Ask Your Finance Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hi! Mujhse apne expenses ke baare me kuch bhi pucho. Jaise: 'How can I save money?' ya 'Predict next month expense'"}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Apna sawaal type karo...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        response = advisor.answer_question(user_q, monthly_income=monthly_income)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    st.divider()
    st.caption("Quick questions:")
    qcol1, qcol2, qcol3 = st.columns(3)
    if qcol1.button("How can I save money?"):
        st.session_state.chat_history.append({"role": "user", "content": "How can I save money?"})
        resp = advisor.answer_question("How can I save money?", monthly_income=monthly_income)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()
    if qcol2.button("Highest spending category?"):
        st.session_state.chat_history.append({"role": "user", "content": "What is my highest spending category?"})
        resp = advisor.answer_question("What is my highest spending category?", monthly_income=monthly_income)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()
    if qcol3.button("Predict next month"):
        st.session_state.chat_history.append({"role": "user", "content": "Predict next month expense"})
        resp = advisor.answer_question("Predict next month expense", monthly_income=monthly_income)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()

# ---------------------------------------------------------------
# TAB 5: RAW DATA
# ---------------------------------------------------------------
with tab5:
    st.subheader("📄 Raw Expense Data")
    st.dataframe(analyzer.df[["Date", "Category", "Amount"]], use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    analyzer.df[["Date", "Category", "Amount"]].to_csv(csv_buffer, index=False)
    st.download_button(
        " Download this data as CSV",
        data=csv_buffer.getvalue(),
        file_name="expense_data.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("Built with Python, Pandas, Scikit-learn & Streamlit")