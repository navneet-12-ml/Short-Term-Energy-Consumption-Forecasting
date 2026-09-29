import os
import pandas as pd


class DataTransformation:

    def __init__(self, input_path):

        self.input_path = input_path

        self.output_path = (
            "data/processed/"
            "transformed_energy_data.csv"
        )

    def initiate_data_transformation(self):

        print("Starting Data Transformation...")

        # -----------------------------------------
        # 1. Load raw dataset
        # -----------------------------------------

        df = pd.read_csv(
            self.input_path
        )

        print(
            "Original data shape:",
            df.shape
        )

        # -----------------------------------------
        # 2. Convert date column
        # -----------------------------------------

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        # Remove rows with invalid dates
        df = df.dropna(
            subset=["date"]
        )

        # -----------------------------------------
        # 3. Sort chronologically
        # -----------------------------------------

        df = df.sort_values(
            by="date"
        ).reset_index(
            drop=True
        )

        print("Data sorted chronologically.")

        # -----------------------------------------
        # 4. Create time features
        # -----------------------------------------

        df["hour"] = df["date"].dt.hour

        df["day"] = df["date"].dt.day

        df["minute"] = df["date"].dt.minute

        df["month"] = df["date"].dt.month

        df["day_of_week"] = (
            df["date"].dt.dayofweek
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        # -----------------------------------------
        # 5. Create lag features
        # -----------------------------------------

        df["Appliances_lag_1"] = (
            df["Appliances"].shift(1)
        )

        df["Appliances_lag_2"] = (
            df["Appliances"].shift(2)
        )

        df["Appliances_lag_3"] = (
            df["Appliances"].shift(3)
        )

        df["Appliances_lag_6"] = (
            df["Appliances"].shift(6)
        )

        df["Appliances_lag_12"] = (
            df["Appliances"].shift(12)
        )

        df["Appliances_lag_24"] = (
            df["Appliances"].shift(24)
        )

        df["Appliances_lag_36"] = (
            df["Appliances"].shift(36)
        )

        df["Appliances_lag_48"] = (
            df["Appliances"].shift(48)
        )

        # 24 hours
        # 24 hours × 6 observations/hour = 144
        df["Appliances_lag_144"] = (
            df["Appliances"].shift(144)
        )

        # 7 days
        # 7 × 24 × 6 = 1008
        df["Appliances_lag_1008"] = (
            df["Appliances"].shift(1008)
        )

        # -----------------------------------------
        # 6. Create rolling features
        # -----------------------------------------

        # 6 observations = 1 hour
        df["rolling_mean_6"] = (
            df["Appliances"]
            .shift(1)
            .rolling(6)
            .mean()
        )

        # 12 observations = 2 hours
        df["rolling_mean_12"] = (
            df["Appliances"]
            .shift(1)
            .rolling(12)
            .mean()
        )

        df["rolling_std_6"] = (
            df["Appliances"]
            .shift(1)
            .rolling(6)
            .std()
        )

        # 24 observations = 4 hours
        df["rolling_mean_24"] = (
            df["Appliances"]
            .shift(1)
            .rolling(24)
            .mean()
        )

        df["rolling_std_24"] = (
            df["Appliances"]
            .shift(1)
            .rolling(24)
            .std()
        )

        # 144 observations = 24 hours
        df["rolling_mean_144"] = (
            df["Appliances"]
            .shift(1)
            .rolling(144)
            .mean()
        )

        # -----------------------------------------
        # 7. Remove NaN values
        # -----------------------------------------

        before_drop = len(df)

        df = df.dropna().reset_index(
            drop=True
        )

        after_drop = len(df)

        print(
            "Rows before removing NaN:",
            before_drop
        )

        print(
            "Rows after removing NaN:",
            after_drop
        )

        print(
            "Rows removed:",
            before_drop - after_drop
        )

        # -----------------------------------------
        # 8. Create processed directory
        # -----------------------------------------

        os.makedirs(
            "data/processed",
            exist_ok=True
        )

        # -----------------------------------------
        # 9. Save transformed dataset
        # -----------------------------------------

        df.to_csv(
            self.output_path,
            index=False
        )

        print(
            "\nTransformed data saved to:"
        )

        print(
            self.output_path
        )

        print(
            "Final transformed shape:",
            df.shape
        )

        return df


if __name__ == "__main__":

    input_path = (
        "data/raw/"
        "energydata_complete.csv"
    )

    transformer = DataTransformation(
        input_path
    )

    transformer.initiate_data_transformation()