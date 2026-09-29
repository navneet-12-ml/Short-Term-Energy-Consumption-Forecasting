import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


class ModelVisualization:

    def __init__(self, model_path, data_path):

        self.model_path = model_path
        self.data_path = data_path

    def plot_actual_vs_predicted(self):

        print("Starting Model Visualization...")

        # =========================================
        # 1. Load processed data
        # =========================================

        df = pd.read_csv(
            self.data_path
        )

        # =========================================
        # 2. Separate X and y
        # =========================================

        X = df.drop(
            columns=["Appliances", "date"]
        )

        y = df["Appliances"]

        # =========================================
        # 3. Same chronological train/test split
        # =========================================

        split_index = int(len(df) * 0.8)

        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        # =========================================
        # 4. Load trained model
        # =========================================

        model = joblib.load(
            self.model_path
        )

        print("Model loaded successfully.")

        # =========================================
        # 5. Generate predictions
        # =========================================

        y_pred = model.predict(
            X_test
        )

        # =========================================
        # 6. Create comparison DataFrame
        # =========================================

        results = pd.DataFrame({

            "Actual": y_test.values,

            "Predicted": y_pred
        })

        print(
            "\nPrediction comparison:"
        )

        print(
            results.head(10)
        )

        # =========================================
        # 7. Create visualization directory
        # =========================================

        os.makedirs(
            "artifacts/visualizations",
            exist_ok=True
        )

        # =========================================
        # 8. Plot Actual vs Predicted
        # =========================================

        plt.figure(figsize=(14, 6))

        plt.plot(
            results["Actual"],
            label="Actual Energy"
        )

        plt.plot(
            results["Predicted"],
            label="Predicted Energy"
        )

        plt.title(
            "Actual vs Predicted Energy Consumption"
        )

        plt.xlabel(
            "Test Set Observation"
        )

        plt.ylabel(
            "Energy Consumption (Wh)"
        )

        plt.legend()

        plt.tight_layout()

        # =========================================
        # 9. Save plot
        # =========================================

        output_path = (
            "artifacts/visualizations/"
            "actual_vs_predicted.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.show()

        print(
            "\nVisualization saved to:"
        )

        print(
            output_path
        )

        # =========================================
        # 10. Save prediction data
        # =========================================

        results.to_csv(
            "artifacts/"
            "prediction_results.csv",
            index=False
        )

        print(
            "Prediction results saved to:"
        )

        print(
            "artifacts/prediction_results.csv"
        )


if __name__ == "__main__":

    model_path = (
        "models/"
        "gradient_boosting_energy_model.pkl"
    )

    data_path = (
        "data/processed/"
        "transformed_energy_data.csv"
    )

    visualization = ModelVisualization(
        model_path,
        data_path
    )

    visualization.plot_actual_vs_predicted()