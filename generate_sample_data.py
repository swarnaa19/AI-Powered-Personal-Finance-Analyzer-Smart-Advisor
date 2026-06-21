
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


np.random.seed(42)

CATEGORIES = {
    "Food":      {"min": 100, "max": 600,  "freq_days": 1, "monthly_trend": 1.01},
    "Travel":    {"min": 50,  "max": 500,  "freq_days": 2, "monthly_trend": 1.00},
    "Shopping":  {"min": 300, "max": 4000, "freq_days": 5, "monthly_trend": 1.06},  # trending up
    "Bills":     {"min": 1500,"max": 3000, "freq_days": 30,"monthly_trend": 1.02},  # monthly fixed-ish
    "Entertainment": {"min": 150, "max": 1200, "freq_days": 4, "monthly_trend": 1.00},
    "Health":    {"min": 100, "max": 2500, "freq_days": 10, "monthly_trend": 1.00},
    "Others":    {"min": 50,  "max": 800,  "freq_days": 6, "monthly_trend": 1.00},
}

MONTHLY_SALARY = 50000

def generate_expenses(start_date: str, num_months: int = 6) -> pd.DataFrame:
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(days=30 * num_months)

    records = []
    current_date = start
    day_index = 0

    while current_date < end:
        month_number = (current_date.year - start.year) * 12 + (current_date.month - start.month)

        for cat, props in CATEGORIES.items():
            
            if day_index % props["freq_days"] == 0:
                
                trend_multiplier = props["monthly_trend"] ** month_number
                amount = np.random.uniform(props["min"], props["max"]) * trend_multiplier
                amount = round(amount, -1)  # nearest 10 par round (real expense jaisa lagega)

                records.append({
                    "Date": current_date.strftime("%d-%m-%y"),
                    "Category": cat,
                    "Amount": int(amount)
                })

        current_date += timedelta(days=1)
        day_index += 1

    df = pd.DataFrame(records)
    return df


def add_income_column(df: pd.DataFrame) -> pd.DataFrame:

    return df


if __name__ == "__main__":
    df = generate_expenses(start_date="2026-01-01", num_months=6)
    df = df.sort_values("Date").reset_index(drop=True)

    output_path = "sample_expenses.csv"
    df.to_csv(output_path, index=False)

    print(f"Sample data generated: {len(df)} transactions")
    print(f" Saved to: {output_path}")
    print("\nPreview:")
    print(df.head(10))
    print(f"\nTotal Spend: ₹{df['Amount'].sum():,}")