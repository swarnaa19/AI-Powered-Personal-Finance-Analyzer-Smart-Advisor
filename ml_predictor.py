import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class ExpensePredictor:
    def __init__(self, monthly_spend: pd.Series):
        self.monthly_spend = monthly_spend.copy()
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=4)
        self._prepare_features()

    def _prepare_features(self):
        self.X = np.arange(len(self.monthly_spend)).reshape(-1, 1)
        self.y = self.monthly_spend.values
        self.month_labels = [str(m) for m in self.monthly_spend.index]

    def train(self):
        if len(self.X) < 3:
            raise ValueError("Prediction ke liye kam se kam 3 mahine ka data chahiye.")
        self.lr_model.fit(self.X, self.y)
        self.rf_model.fit(self.X, self.y)

    def evaluate(self) -> dict:
        lr_pred = self.lr_model.predict(self.X)
        rf_pred = self.rf_model.predict(self.X)
        return {
            "Linear Regression MAE": round(mean_absolute_error(self.y, lr_pred), 2),
            "Random Forest MAE": round(mean_absolute_error(self.y, rf_pred), 2),
        }

    def predict_next_month(self) -> dict:
        next_month_index = np.array([[len(self.X)]])
        lr_pred = float(self.lr_model.predict(next_month_index)[0])
        rf_pred = float(self.rf_model.predict(next_month_index)[0])
        lr_pred = max(0, round(lr_pred))
        rf_pred = max(0, round(rf_pred))
        blended = round((lr_pred + rf_pred) / 2)
        return {
            "linear_regression": lr_pred,
            "random_forest": rf_pred,
            "blended_estimate": blended,
        }

    def predict_category_next_month(self, category_pivot: pd.DataFrame) -> pd.Series:
        predictions = {}
        X = np.arange(len(category_pivot)).reshape(-1, 1)
        next_idx = np.array([[len(category_pivot)]])
        for col in category_pivot.columns:
            y = category_pivot[col].values
            if len(set(y)) == 1:
                predictions[col] = max(0, round(y[-1]))
                continue
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(next_idx)[0]
            predictions[col] = max(0, round(pred))
        return pd.Series(predictions).sort_values(ascending=False)