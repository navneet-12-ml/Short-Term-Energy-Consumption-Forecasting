import joblib
import pandas as pd


class ForecastPredictionPipeline:

    def __init__(self, model_path, data_path):

        self.model_path = model_path
        self.data_path = data_path

    def predict_next_10_minutes(self):

        print("=" * 60)
        print("       NEXT 10-MINUTE ENERGY FORECAST")
        print("=" * 60)

        # =========================================
        # 1. Load trained forecasting model
        # =========================================

        print("\nLoading forecasting model...")

        model = joblib.load(
            self.model_path
        )

        print("Model loaded successfully.")

        # =========================================
        # 2. Load processed data
        # =========================================

        df = pd.read_csv(
            self.data_path
        )

        # =========================================
        # 3. Convert date
        # =========================================

        df["date"] = pd.to_datetime(
            df["date"]
        )

        # =========================================
        # 4. Get latest available observation
        # =========================================

        latest_row = df.iloc[[-1]].copy()

        current_time = latest_row[
            "date"
        ].iloc[0]

        # =========================================
        # 5. Calculate forecast timestamp
        # =========================================

        forecast_time = (
            current_time
            + pd.Timedelta(minutes=10)
        )

        # =========================================
        # 6. Create model input
        # =========================================

        X = latest_row.drop(
            columns=[
                "Appliances",
                "date"
            ]
        )

        print(
            "\nCurrent observation:",
            current_time
        )

        print(
            "Forecast time:",
            forecast_time
        )

        print(
            "Input shape:",
            X.shape
        )

        print(
            "Number of features:",
            X.shape[1]
        )

        # =========================================
        # 7. Generate forecast
        # =========================================

        prediction = model.predict(X)

        predicted_energy = prediction[0]

        # =========================================
        # 8. Display result
        # =========================================

        print("\nForecast Result")
        print("-" * 40)

        print(
            "Current time:",
            current_time
        )

        print(
            "Forecast time:",
            forecast_time
        )

        print(
            "Forecast horizon: 10 minutes"
        )

        print(
            "Predicted energy consumption:",
            round(predicted_energy, 2),
            "Wh"
        )

        return {
            "current_time": str(current_time),
            "forecast_time": str(forecast_time),
            "forecast_horizon": "10 minutes",
            "predicted_energy_wh": round(
                float(predicted_energy),
                2
            )
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

    pipeline = ForecastPredictionPipeline(
        model_path,
        data_path
    )

    result = (
        pipeline.predict_next_10_minutes()
    )

    print("\nFinal Forecast:")
    print(result)