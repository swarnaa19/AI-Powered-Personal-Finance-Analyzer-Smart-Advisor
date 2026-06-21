

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class ExpenseAnalyzer:
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None):
        
        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.read_csv(csv_path)

        self._clean_data()

    # ---------------------------------------------------------
    # DATA CLEANING
    # ---------------------------------------------------------
    def _clean_data(self):
        
        if self.df["Amount"].dtype == object:
            self.df["Amount"] = (
                self.df["Amount"]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
            )
        self.df["Amount"] = pd.to_numeric(self.df["Amount"], errors="coerce")

        self.df = self.df.dropna(subset=["Date", "Amount", "Category"])

        
        self.df["YearMonth"] = self.df["Date"].dt.to_period("M")
        self.df["MonthName"] = self.df["Date"].dt.strftime("%b %Y")
        self.df = self.df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # CORE METRICS
    # ---------------------------------------------------------
    def total_spend(self) -> float:
        return round(self.df["Amount"].sum(), 2)

    def monthly_spend(self) -> pd.Series:
        """Har mahine ka total expense"""
        return self.df.groupby("YearMonth")["Amount"].sum().sort_index()

    def category_spend(self) -> pd.Series:
        """Har category ka total expense, sorted descending"""
        return self.df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    def category_percentage(self) -> pd.Series:
        
        cat_spend = self.category_spend()
        return (cat_spend / cat_spend.sum() * 100).round(1)

    def month_over_month_change(self) -> pd.Series:
        
        monthly = self.monthly_spend()
        pct_change = monthly.pct_change() * 100
        return pct_change.round(1)

    def category_month_pivot(self) -> pd.DataFrame:
        """Category x Month pivot table - heatmap aur trend ke liye"""
        return self.df.pivot_table(
            index="YearMonth", columns="Category", values="Amount", aggfunc="sum"
        ).fillna(0)

    def top_category_last_month(self) -> tuple:
        
        latest_month = self.df["YearMonth"].max()
        last_month_df = self.df[self.df["YearMonth"] == latest_month]
        cat_totals = last_month_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        if len(cat_totals) == 0:
            return None, 0
        return cat_totals.index[0], cat_totals.iloc[0]

    def category_trend_change(self) -> pd.DataFrame:
        
        pivot = self._get_complete_pivot()
        if len(pivot) < 2:
            return pd.DataFrame(columns=["Category", "Previous", "Current", "PctChange"])

        prev_month = pivot.iloc[-2]
        curr_month = pivot.iloc[-1]

        result = pd.DataFrame({
            "Category": pivot.columns,
            "Previous": prev_month.values,
            "Current": curr_month.values,
        })
        result["PctChange"] = np.where(
            result["Previous"] > 0,
            ((result["Current"] - result["Previous"]) / result["Previous"] * 100).round(1),
            0
        )
        return result.sort_values("PctChange", ascending=False).reset_index(drop=True)

    def _get_complete_pivot(self) -> pd.DataFrame:
       
        pivot = self.category_month_pivot()
        if len(pivot) == 0:
            return pivot

        latest_period = pivot.index.max()
        today = pd.Timestamp.now().normalize()
       
        last_data_date = self.df[self.df["YearMonth"] == latest_period]["Date"].max()
        month_end = latest_period.end_time.normalize()

        if last_data_date < month_end and latest_period.end_time >= today:
            pivot = pivot.iloc[:-1]

        return pivot

    # ---------------------------------------------------------
    # CHARTS (Matplotlib + Seaborn)
    # ---------------------------------------------------------
    def plot_monthly_bar(self):
        """Bar chart: Month-wise total spend"""
        monthly = self.monthly_spend()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        monthly.index = monthly.index.astype(str)
        sns.barplot(x=monthly.index, y=monthly.values, ax=ax, hue=monthly.index,
                    palette="viridis", legend=False)
        ax.set_title("Monthly Spending Trend", fontsize=13, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (₹)")
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig

    def plot_category_pie(self):
        """Pie chart: Category-wise % distribution"""
        cat_spend = self.category_spend()
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = sns.color_palette("Set2", len(cat_spend))
        ax.pie(
            cat_spend.values,
            labels=cat_spend.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 10}
        )
        ax.set_title("Category-wise Spending Share", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_category_trend_line(self):
        """Line graph: Har category ka month-over-month trend"""
        pivot = self.category_month_pivot()
        pivot.index = pivot.index.astype(str)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", label=col)
        ax.set_title("Category-wise Monthly Trend", fontsize=13, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (₹)")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Quick test
    analyzer = ExpenseAnalyzer(csv_path="sample_expenses.csv")
    print("Total Spend:", analyzer.total_spend())
    print("\nMonthly Spend:\n", analyzer.monthly_spend())
    print("\nCategory %:\n", analyzer.category_percentage())
    print("\nMoM Change:\n", analyzer.month_over_month_change())
    print("\nCategory Trend Change:\n", analyzer.category_trend_change())