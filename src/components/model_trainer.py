import os
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor


class ModelTrainer:

    def __init__(self, data_path):

        self.data_path = data_path

        self.model_path = (
            "models/"
            "next_10min_energy_model.pkl"
        )

    def initiate_model_training(self):

        print("Starting Next 10-Minute Forecast Model Training...")

        # =========================================
        # 1. Load transformed data
        # =========================================

        df = pd.read_csv(
            self.data_path
        )

        print(
            "Processed data shape:",
            df.shape
        )

        # =========================================
        # 2. Create next-step target
        # =========================================

        df["target_next_10min"] = (
            df["Appliances"].shift(-1)
        )

        # Last row has no future target
        df = df.dropna().reset_index(
            drop=True
        )

        # =========================================
        # 3. Separate X and y
        # =========================================

        X = df.drop(
            columns=[
                "Appliances",
                "target_next_10min",
                "date"
            ]
        )

        y = df["target_next_10min"]

        print(
            "Feature shape:",
            X.shape
        )

        print(
            "Target shape:",
            y.shape
        )

        # =========================================
        # 4. Chronological train/test split
        # =========================================

        split_index = int(
            len(df) * 0.8
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        print(
            "X_train shape:",
            X_train.shape
        )

        print(
            "X_test shape:",
            X_test.shape
        )

        # =========================================
        # 5. Create Gradient Boosting model
        # =========================================

        model = GradientBoostingRegressor(

            learning_rate=0.05,

            max_depth=2,

            min_samples_leaf=2,

            min_samples_split=2,

            n_estimators=100,

            random_state=42
        )

        # =========================================
        # 6. Train
        # =========================================

        print(
            "\nTraining model..."
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "Model training completed."
        )

        # =========================================
        # 7. Create models directory
        # =========================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        # =========================================
        # 8. Save model
        # =========================================

        joblib.dump(
            model,
            self.model_path
        )

        print(
            "\nModel saved successfully:"
        )

        print(
            self.model_path
        )

        return (
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )


if __name__ == "__main__":

    data_path = (
        "data/processed/"
        "transformed_energy_data.csv"
    )

    trainer = ModelTrainer(
        data_path
    )

    trainer.initiate_model_training()