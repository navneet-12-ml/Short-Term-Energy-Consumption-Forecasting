from flask import Flask, jsonify, render_template

from src.pipeline.forecast_prediction_pipeline import (
    ForecastPredictionPipeline
)


app = Flask(__name__)


MODEL_PATH = (
    "models/"
    "next_10min_energy_model.pkl"
)

DATA_PATH = (
    "data/processed/"
    "transformed_energy_data.csv"
)


# =========================================
# Home API
# =========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================
# Prediction API
# =========================================

@app.route("/predict", methods=["GET"])
def predict():

    try:

        pipeline = ForecastPredictionPipeline(
            MODEL_PATH,
            DATA_PATH
        )

        result = (
            pipeline.predict_next_10_minutes()
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================================
# Recent energy data
# =========================================

@app.route("/history", methods=["GET"])
def history():

    try:

        import pandas as pd

        df = pd.read_csv(
            DATA_PATH
        )

        df["date"] = pd.to_datetime(
            df["date"]
        )

        # Last 30 observations
        recent = df.tail(30)

        data = []

        for _, row in recent.iterrows():

            data.append({
                "time": row["date"].strftime(
                    "%H:%M"
                ),
                "energy": float(
                    row["Appliances"]
                )
            })

        return jsonify(data)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )