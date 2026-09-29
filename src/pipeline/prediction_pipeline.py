import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(
        self,
        model_path,
        data_path
    ):

        self.model_path = model_path
        self.data_path = data_path

    def predict(self):

        print("=" * 60)
        print("       ENERGY CONSUMPTION PREDICTION")
        print("=" * 60)

        # =========================================
        # 1. Load trained model
        # =========================================

        print("\nLoading trained model...")

        model = joblib.load(
            self.model_path
        )

        print("Model loaded successfully.")

        # =========================================
        # 2. Load processed data
        # =========================================

        print("\nLoading processed data...")

        df = pd.read_csv(
            self.data_path
        )

        print(
            "Processed data shape:",
            df.shape
        )

        # =========================================
        # 3. Select the latest observation
        # =========================================

        latest_row = df.iloc[[-1]].copy()

        # Keep timestamp for displaying
        timestamp = latest_row["date"].iloc[0]

        # =========================================
        # 4. Create prediction input
        # =========================================

        X = latest_row.drop(
            columns=[
                "Appliances",
                "date"
            ]
        )

        print(
            "\nPrediction input shape:",
            X.shape
        )

        print(
            "Number of features:",
            X.shape[1]
        )

        # =========================================
        # 5. Generate prediction
        # =========================================

        prediction = model.predict(X)

        predicted_energy = prediction[0]

        # =========================================
        # 6. Display prediction
        # =========================================

        print("\nPrediction Result")
        print("-" * 40)

        print(
            "Timestamp:",
            timestamp
        )

        print(
            "Predicted energy consumption:",
            round(predicted_energy, 2),
            "Wh"
        )

        # =========================================
        # 7. Display actual value
        # =========================================

        actual_energy = (
            latest_row["Appliances"].iloc[0]
        )

        print(
            "Actual energy consumption:",
            actual_energy,
            "Wh"
        )

        print(
            "Prediction error:",
            round(
                abs(
                    predicted_energy
                    - actual_energy
                ),
                2
            ),
            "Wh"
        )

        return predicted_energy


if __name__ == "__main__":

    model_path = (
        "models/"
        "gradient_boosting_energy_model.pkl"
    )

    data_path = (
        "data/processed/"
        "transformed_energy_data.csv"
    )

    prediction_pipeline = PredictionPipeline(
        model_path,
        data_path
    )

    prediction_pipeline.predict()