import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelEvaluation:

    def __init__(
        self,
        model_path,
        data_path
    ):

        self.model_path = model_path

        self.data_path = data_path

    def initiate_model_evaluation(self):

        print("Starting Model Evaluation...")

        # =========================================
        # 1. Load transformed data
        # =========================================

        df = pd.read_csv(
            self.data_path
        )

        # =========================================
        # 2. Separate X and y
        # =========================================

        X = df.drop(
            columns=[
                "Appliances",
                "date"
            ]
        )

        y = df["Appliances"]

        # =========================================
        # 3. Same chronological split
        # =========================================

        split_index = int(len(df) * 0.8)

        X_train = X.iloc[:split_index]

        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]

        y_test = y.iloc[split_index:]

        print(
            "Test data shape:",
            X_test.shape
        )

        # =========================================
        # 4. Load trained model
        # =========================================

        model = joblib.load(
            self.model_path
        )

        print(
            "Model loaded successfully."
        )

        # =========================================
        # 5. Generate predictions
        # =========================================

        y_pred = model.predict(
            X_test
        )

        # =========================================
        # 6. Calculate metrics
        # =========================================

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        # =========================================
        # 7. Display results
        # =========================================

        print("\nModel Evaluation Results")

        print(
            "MAE :",
            mae
        )

        print(
            "MSE :",
            mse
        )

        print(
            "RMSE:",
            rmse
        )

        print(
            "R2  :",
            r2
        )

        # =========================================
        # 8. Save evaluation results
        # =========================================

        evaluation_results = pd.DataFrame({

            "Metric": [
                "MAE",
                "MSE",
                "RMSE",
                "R2"
            ],

            "Value": [
                mae,
                mse,
                rmse,
                r2
            ]
        })

        evaluation_results.to_csv(
            "models/evaluation_results.csv",
            index=False
        )

        print(
            "\nEvaluation results saved to:"
        )

        print(
            "models/evaluation_results.csv"
        )

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }


if __name__ == "__main__":

    model_path = (
        "models/"
        "gradient_boosting_energy_model.pkl"
    )

    data_path = (
        "data/processed/"
        "transformed_energy_data.csv"
    )

    evaluator = ModelEvaluation(
        model_path,
        data_path
    )

    evaluator.initiate_model_evaluation()