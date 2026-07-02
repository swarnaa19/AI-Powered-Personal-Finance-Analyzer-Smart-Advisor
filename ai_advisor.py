"""
AI Financial Advisor — Rule-Based Natural Language Generator
=============================================================
Generates human-readable financial insights from spending data
without requiring any paid API (OpenAI / Gemini).

Approach: Template-based NLG (Natural Language Generation).
Upgrade path: Replace generate_full_report() with an LLM API
call — the rest of the project stays identical.
"""

import pandas as pd


class FinancialAdvisor:
    def __init__(self, analyzer, predictor=None):
        self.analyzer  = analyzer
        self.predictor = predictor

    # ----------------------------------------------------------------
    # PUBLIC — Full insight report
    # ----------------------------------------------------------------
    def generate_full_report(self, monthly_income: float = None) -> list:
        """Return a list of insight strings for the dashboard."""
        insights = []
        insights.extend(self._spending_trend_insight())
        insights.extend(self._category_change_insights())
        insights.extend(self._top_category_insight())
        if monthly_income:
            insights.extend(self._savings_insight(monthly_income))
        if self.predictor:
            insights.extend(self._prediction_insight())
        insights.extend(self._budget_recommendations())
        return insights

    # ----------------------------------------------------------------
    # INSIGHT 1 — Overall spending trend (MoM)
    # ----------------------------------------------------------------
    def _spending_trend_insight(self) -> list:
        mom = self.analyzer.month_over_month_change().dropna()
        if len(mom) == 0:
            return []

        change = mom.iloc[-1]
        month  = str(mom.index[-1])

        if change > 15:
            msg = (f"⚠️ Your total spending in **{month}** jumped **{change}%** "
                   f"compared to the previous month — a significant increase worth reviewing.")
        elif change > 0:
            msg = (f"📈 Spending in **{month}** increased slightly ({change}%). "
                   f"Not alarming yet, but keep an eye on the trend.")
        elif change < -15:
            msg = (f"🎉 Great discipline! Spending in **{month}** dropped by "
                   f"**{abs(change)}%** — keep up the good work.")
        else:
            msg = (f"✅ Spending in **{month}** remained stable ({change}% change). "
                   f"Consistency is a positive sign.")

        return [msg]

    # ----------------------------------------------------------------
    # INSIGHT 2 — Category spikes
    # ----------------------------------------------------------------
    def _category_change_insights(self) -> list:
        trend_df = self.analyzer.category_trend_change()
        if trend_df.empty:
            return []

        insights = []
        top = trend_df.iloc[0]
        if top["PctChange"] > 20 and top["Current"] > 0:
            extra        = top["Current"] - top["Previous"]
            annual_save  = round(extra * 0.5 * 12, -2)
            msg = (f"🛍️ **{top['Category']}** spending rose **{top['PctChange']}%** "
                   f"last month (₹{top['Previous']:,.0f} → ₹{top['Current']:,.0f}). "
                   f"Cutting it by 50% could save you **₹{annual_save:,.0f} per year**.")
            insights.append(msg)

        bottom = trend_df.iloc[-1]
        if bottom["PctChange"] < -20:
            msg = (f"👍 **{bottom['Category']}** spending fell {abs(bottom['PctChange'])}% "
                   f"— well controlled!")
            insights.append(msg)

        return insights

    # ----------------------------------------------------------------
    # INSIGHT 3 — Top spending category
    # ----------------------------------------------------------------
    def _top_category_insight(self) -> list:
        cat_pct = self.analyzer.category_percentage()
        if len(cat_pct) == 0:
            return []

        top_cat = cat_pct.index[0]
        top_pct = cat_pct.iloc[0]

        if top_pct > 35:
            msg = (f"🔍 **{top_cat}** accounts for **{top_pct}%** of your total spending. "
                   f"Experts recommend keeping any single non-essential category below 30%.")
        else:
            msg = (f"📊 Your spending is reasonably balanced across categories. "
                   f"The largest share is **{top_cat}** at {top_pct}%.")

        return [msg]

    # ----------------------------------------------------------------
    # INSIGHT 4 — Savings rate
    # ----------------------------------------------------------------
    def _savings_insight(self, monthly_income: float) -> list:
        monthly = self.analyzer.monthly_spend()
        if len(monthly) == 0:
            return []

        avg_expense  = monthly.mean()
        savings      = monthly_income - avg_expense
        savings_rate = (savings / monthly_income * 100) if monthly_income > 0 else 0

        if savings_rate < 0:
            msg = (f"🚨 Your average monthly expense (₹{avg_expense:,.0f}) exceeds your "
                   f"income (₹{monthly_income:,.0f}). An immediate budget review is critical.")
        elif savings_rate < 20:
            msg = (f"💰 Your current savings rate is **{savings_rate:.1f}%** "
                   f"(₹{savings:,.0f}/month). Financial experts recommend 20%+ — "
                   f"there is room for improvement.")
        else:
            msg = (f"🌟 Excellent! Your savings rate is **{savings_rate:.1f}%** "
                   f"(₹{savings:,.0f}/month). This is a healthy financial habit.")

        return [msg]

    # ----------------------------------------------------------------
    # INSIGHT 5 — ML prediction
    # ----------------------------------------------------------------
    def _prediction_insight(self) -> list:
        try:
            pred = self.predictor.predict_next_month()
        except Exception:
            return []

        msg = (f"🔮 The ML model forecasts next month's expense at "
               f"**₹{pred['blended_estimate']:,}** "
               f"(Linear Regression: ₹{pred['linear_regression']:,} | "
               f"Random Forest: ₹{pred['random_forest']:,}). "
               f"Use this as your planning baseline.")
        return [msg]

    # ----------------------------------------------------------------
    # INSIGHT 6 — Budget caps
    # ----------------------------------------------------------------
    def _budget_recommendations(self) -> list:
        cat_spend = self.analyzer.category_spend()
        if len(cat_spend) == 0:
            return []

        num_months = max(self.analyzer.monthly_spend().shape[0], 1)
        lines      = ["💡 **Recommended Monthly Budget Caps:**"]

        for cat, total in cat_spend.items():
            avg = total / num_months
            if cat in ["Shopping", "Entertainment", "Others"]:
                cap = avg * 0.9
                lines.append(f"   • {cat}: avg ₹{avg:,.0f} → suggested cap **₹{cap:,.0f}**")
            else:
                lines.append(f"   • {cat}: ₹{avg:,.0f}/month (essential)")

        return ["\n".join(lines)]

    # ----------------------------------------------------------------
    # CHATBOT — keyword-intent matching
    # ----------------------------------------------------------------
    def answer_question(self, question: str, monthly_income: float = None) -> str:
        """
        Match user question to an intent and return a relevant response.
        Supports English and common Hindi/Hinglish phrases.
        """
        q = question.lower()

        INTENTS = {
            "save":     ["save","saving","savings","reduce","cut expense",
                         "bachao","bachat","bachau","bachana","paisa bacha"],
            "predict":  ["predict","prediction","next month","forecast","future",
                         "agle mahine","agle","kitna hoga"],
            "category": ["category","where","highest","most spent","kaha","kahan",
                         "sabse zyada","kidhar","kis category"],
            "total":    ["total","how much","overall","kitna","kul","poora"],
            "analyze":  ["analyze","analysis","summary","report","overview",
                         "dikhao","batao","dekho"],
        }

        intent = "unknown"
        for name, keywords in INTENTS.items():
            if any(kw in q for kw in keywords):
                intent = name
                break

        # ── Savings ───────────────────────────────────────────────
        if intent == "save":
            tips = self._savings_insight(monthly_income) if monthly_income else []
            advice = (
                "**Here are personalised tips based on your spending data:**\n\n"
                "1. 🛍️ Reduce Shopping & Entertainment by 20%\n"
                "2. 🍕 Cook more at home — cut food delivery orders\n"
                "3. 💰 Auto-transfer 20% of income to savings on payday\n"
                "4. 📱 Cancel unused subscriptions immediately\n"
                "5. 📊 Review your budget at the end of every month"
            )
            tip_text = str(tips[0]) if tips else ""
            return (tip_text + "\n\n" + advice) if tip_text else advice

        # ── Prediction ────────────────────────────────────────────
        if intent == "predict":
            if self.predictor:
                try:
                    pred = self.predictor.predict_next_month()
                    return (
                        f"🔮 **Next Month Expense Forecast**\n\n"
                        f"**Best Estimate: ₹{pred['blended_estimate']:,}**\n\n"
                        f"• Linear Regression: ₹{pred['linear_regression']:,}\n"
                        f"• Random Forest:     ₹{pred['random_forest']:,}\n\n"
                        f"Plan your budget around this figure."
                    )
                except Exception:
                    return "Prediction model is not ready yet. Please try again."
            return "At least 3 months of data is needed to generate a prediction."

        # ── Category ──────────────────────────────────────────────
        if intent == "category":
            top_cat, amount = self.analyzer.top_category_last_month()
            cat_pct         = self.analyzer.category_percentage()
            top_pct         = cat_pct.get(top_cat, 0)
            breakdown       = "\n".join(
                [f"• {c}: {p}%" for c, p in cat_pct.head(5).items()]
            )
            return (
                f"Your highest spending category is **{top_cat}**.\n\n"
                f"Last month: ₹{amount:,.0f} | Overall share: {top_pct}%\n\n"
                f"**Top 5 Categories:**\n{breakdown}"
            )

        # ── Total ─────────────────────────────────────────────────
        if intent == "total":
            return (
                f"Your total tracked expenditure is "
                f"**₹{self.analyzer.total_spend():,.0f}**."
            )

        # ── Analyze ───────────────────────────────────────────────
        if intent == "analyze":
            report = self.generate_full_report(monthly_income)
            return "\n\n".join([str(r) for r in report[:4]])

        # ── Fallback ──────────────────────────────────────────────
        return (
            "I can help you with the following:\n\n"
            "• **How can I save money?**\n"
            "• **What is my highest spending category?**\n"
            "• **Predict next month's expenses**\n"
            "• **What is my total spending?**\n"
            "• **Analyze my expenses**\n\n"
            "Try any of the quick buttons below or type your question!"
        )