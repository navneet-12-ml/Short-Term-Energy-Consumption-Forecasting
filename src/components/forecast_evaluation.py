import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ForecastEvaluation:

    def __init__(
        self,
        model_path,
        data_path
    ):

        self.model_path = model_path
        self.data_path = data_path

    def evaluate(self):

        print(
            "Starting Forecast Model Evaluation..."
        )

        # =========================================
        # 1. Load data
        # =========================================

        df = pd.read_csv(
            self.data_path
        )

        # =========================================
        # 2. Create next 10-minute target
        # =========================================

        df["target_next_10min"] = (
            df["Appliances"].shift(-1)
        )

        df = df.dropna().reset_index(
            drop=True
        )

        # =========================================
        # 3. Create X and y
        # =========================================

        X = df.drop(
            columns=[
                "Appliances",
                "target_next_10min",
                "date"
            ]
        )

        y = df["target_next_10min"]

        # =========================================
        # 4. Chronological split
        # =========================================

        split_index = int(
            len(df) * 0.8
        )

        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        # =========================================
        # 5. Load model
        # =========================================

        model = joblib.load(
            self.model_path
        )

        print(
            "Model loaded successfully."
        )

        # =========================================
        # 6. Predictions
        # =========================================

        y_pred = model.predict(
            X_test
        )

        # =========================================
        # 7. Metrics
        # =========================================

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        print("\n")
        print("Next 10-Minute Forecast Results")
        print("-" * 40)

        print("MAE :", mae)
        print("MSE :", mse)
        print("RMSE:", rmse)
        print("R2  :", r2)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }


if __name__ == "__main__":

    model_path = (
        "models/"
        "next_10min_energy_model.pkl"
    )

    data_path = (
        "data/processed/"
        "transformed_energy_data.csv"
    )

    evaluator = ForecastEvaluation(
        model_path,
        data_path
    )

    evaluator.evaluate()