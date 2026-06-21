
import pandas as pd


class FinancialAdvisor:
    def __init__(self, analyzer, predictor=None):
        """
        analyzer: ExpenseAnalyzer instance (Step 2 se)
        predictor: ExpensePredictor instance (Step 3 se), optional
        """
        self.analyzer = analyzer
        self.predictor = predictor

    
    def generate_full_report(self, monthly_income: float = None) -> list:
        
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

    def _spending_trend_insight(self) -> list:
        mom = self.analyzer.month_over_month_change().dropna()
        if len(mom) == 0:
            return []

        latest_change = mom.iloc[-1]
        latest_month = str(mom.index[-1])

        if latest_change > 15:
            msg = (f" {latest_month} me aapka total expense pichle mahine se "
                   f"**{latest_change}% badh gaya hai**. Yeh ek significant jump hai — "
                   f"is mahine ke transactions check karna helpful rahega.")
        elif latest_change > 0:
            msg = (f"{latest_month} me expense me halka sa increase hua hai "
                   f"({latest_change}%). Abhi worry karne ki baat nahi, par trend par nazar rakho.")
        elif latest_change < -15:
            msg = (f" Great job! {latest_month} me aapka expense pichle mahine se "
                   f"**{abs(latest_change)}% kam hua hai**. Isi tarah continue karo.")
        else:
            msg = (f" {latest_month} me aapka spending roughly stable raha "
                   f"({latest_change}% change). Consistency achi baat hai.")

        return [msg]

    
    def _category_change_insights(self) -> list:
        trend_df = self.analyzer.category_trend_change()
        if trend_df.empty:
            return []

        insights = []
       
        biggest_increase = trend_df.iloc[0]
        if biggest_increase["PctChange"] > 20 and biggest_increase["Current"] > 0:
            cat = biggest_increase["Category"]
            pct = biggest_increase["PctChange"]
            current = biggest_increase["Current"]
            extra_amount = biggest_increase["Current"] - biggest_increase["Previous"]
            annual_projection = extra_amount * 12

            msg = (f" Aapka **{cat}** expense last month **{pct}% increase** hua hai "
                   f"(₹{biggest_increase['Previous']:,.0f} → ₹{current:,.0f}). "
                   f"Agar aap is category me **₹{round(extra_amount * 0.5, -2):,.0f} kam spend** "
                   f"karenge to annual saving **₹{round(annual_projection * 0.5, -2):,.0f}** "
                   f"tak ho sakti hai.")
            insights.append(msg)

       
        biggest_decrease = trend_df.iloc[-1]
        if biggest_decrease["PctChange"] < -20:
            cat = biggest_decrease["Category"]
            pct = abs(biggest_decrease["PctChange"])
            msg = f" **{cat}** expense me {pct}% ki kami aayi hai — well controlled!"
            insights.append(msg)

        return insights

   
    def _top_category_insight(self) -> list:
        cat_pct = self.analyzer.category_percentage()
        if len(cat_pct) == 0:
            return []

        top_cat = cat_pct.index[0]
        top_pct = cat_pct.iloc[0]

        if top_pct > 35:
            msg = (f"**{top_cat}** aapke total spending ka **{top_pct}%** hai — "
                   f"yeh ek single category me kaafi concentration hai. "
                   f"Experts generally recommend kisi bhi ek non-essential category "
                   f"ko 30% se kam rakhna.")
        else:
            msg = (f" Aapka spending categories me reasonably balanced hai. "
                   f"Sabse bada share **{top_cat}** ka hai ({top_pct}%).")

        return [msg]

   
    def _savings_insight(self, monthly_income: float) -> list:
        monthly = self.analyzer.monthly_spend()
        if len(monthly) == 0:
            return []

        avg_monthly_expense = monthly.mean()
        savings = monthly_income - avg_monthly_expense
        savings_rate = (savings / monthly_income * 100) if monthly_income > 0 else 0

        if savings_rate < 0:
            msg = (f"Aapka average monthly expense (₹{avg_monthly_expense:,.0f}) "
                   f"aapki income (₹{monthly_income:,.0f}) se **zyada** hai. "
                   f"Turant budget review karna zaroori hai.")
        elif savings_rate < 20:
            msg = (f" Aapki current savings rate **{savings_rate:.1f}%** hai "
                   f"(₹{savings:,.0f}/month). Financial experts 20%+ savings rate "
                   f"recommend karte hain — thoda aur improve karne ki gunjaish hai.")
        else:
            msg = (f" Excellent! Aapki savings rate **{savings_rate:.1f}%** hai "
                   f"(₹{savings:,.0f}/month saved). Yeh ek healthy financial habit hai.")

        return [msg]

    
    def _prediction_insight(self) -> list:
        try:
            pred = self.predictor.predict_next_month()
        except Exception:
            return []

        blended = pred["blended_estimate"]
        msg = (f" ML model ke hisaab se, **agle mahine ka expected expense "
               f"₹{blended:,}** hone ka anumaan hai "
               f"(Linear Regression: ₹{pred['linear_regression']:,}, "
               f"Random Forest: ₹{pred['random_forest']:,}). "
               f"Is budget ke hisaab se planning kar sakte ho.")
        return [msg]

   
    def _budget_recommendations(self) -> list:
        cat_spend = self.analyzer.category_spend()
        if len(cat_spend) == 0:
            return []

        recommendations = [" **Recommended monthly budget caps:**"]
        avg_monthly_months = self.analyzer.monthly_spend().shape[0]

        for cat, total in cat_spend.items():
            avg_per_month = total / max(avg_monthly_months, 1)
            # Non-essential categories ke liye 10% reduction suggest karo
            if cat in ["Shopping", "Entertainment", "Others"]:
                suggested_cap = avg_per_month * 0.9
                recommendations.append(
                    f"   • {cat}: ₹{avg_per_month:,.0f}/month avg → "
                    f"suggested cap **₹{suggested_cap:,.0f}**"
                )
            else:
                recommendations.append(f"   • {cat}: ₹{avg_per_month:,.0f}/month avg (essential)")

        return ["\n".join(recommendations)]

    
    def answer_question(self, question: str, monthly_income: float = None) -> str:
        """
        Simple keyword-matching based chatbot. User ka question lekar
        relevant insight return karta hai. Yeh ek RULE-BASED chatbot hai -
        asli NLU nahi, par demo ke liye effective hai.
        """
        q = question.lower()

        if any(word in q for word in ["save", "saving", "bachat", "kaise bachau"]):
            tips = self._savings_insight(monthly_income) if monthly_income else []
            general_tips = (
               
            )
            return (tips[0] + "\n\n" + general_tips) if tips else general_tips

        elif any(word in q for word in ["predict", "next month", "agle mahine", "future"]):
            if self.predictor:
                insight = self._prediction_insight()
                return insight[0] if insight else "Prediction abhi available nahi hai."
            return "Prediction ke liye ML model load nahi hua hai."

        elif any(word in q for word in ["category", "kaha", "where", "kis", "highest"]):
            top_cat, amount = self.analyzer.top_category_last_month()
            cat_pct = self.analyzer.category_percentage()
            return (f"Aapka sabse zyada kharcha **{top_cat}** category me hota hai "
                    f"(last month: ₹{amount:,.0f}). Overall, {top_cat} aapke total "
                    f"spending ka {cat_pct.get(top_cat, 0)}% hai.")

        elif any(word in q for word in ["total", "kitna", "how much"]):
            return f"Aapka total tracked expense **₹{self.analyzer.total_spend():,.0f}** hai."

        elif any(word in q for word in ["analyze", "analysis", "summary", "report"]):
            report = self.generate_full_report(monthly_income)
            return "\n\n".join(report[:3])  # top 3 insights summary ke liye

        else:
            return (
                "Main aapke expenses ke baare me yeh questions answer kar sakta hoon:\n"
                "• 'How can I save money?'\n"
                "• 'What is my highest spending category?'\n"
                "• 'Predict next month expense'\n"
                "• 'Analyze my expenses'\n\n"
                "Koi bhi ek pucho!"
            )


if __name__ == "__main__":
    from data_analysis import ExpenseAnalyzer
    from ml_predictor import ExpensePredictor

    analyzer = ExpenseAnalyzer(csv_path="sample_expenses.csv")
    predictor = ExpensePredictor(analyzer.monthly_spend())
    predictor.train()

    advisor = FinancialAdvisor(analyzer, predictor)

    print("===== FULL REPORT =====\n")
    for insight in advisor.generate_full_report(monthly_income=50000):
        print(insight)
        print()

    print("===== CHATBOT TEST =====\n")
    print(advisor.answer_question("How can I save money?", monthly_income=50000))