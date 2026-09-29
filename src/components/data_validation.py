import os
import pandas as pd


class DataValidation:

    def __init__(self, data_path):

        self.data_path = data_path

        self.required_columns = [
            "date",
            "Appliances",
            "lights",
            "T1",
            "RH_1",
            "T2",
            "RH_2",
            "T3",
            "RH_3",
            "T4",
            "RH_4",
            "T5",
            "RH_5",
            "T6",
            "RH_6",
            "T7",
            "RH_7",
            "T8",
            "RH_8",
            "T9",
            "RH_9",
            "T_out",
            "Press_mm_hg",
            "RH_out",
            "Windspeed",
            "Visibility",
            "Tdewpoint",
            "rv1",
            "rv2"
        ]

    def validate_data(self):

        print("Starting Data Validation...")

        validation_status = True

        # -----------------------------------------
        # 1. Check whether file exists
        # -----------------------------------------

        if not os.path.exists(self.data_path):

            print("ERROR: Dataset file does not exist.")

            return False

        print("Dataset file exists.")

        # -----------------------------------------
        # 2. Load dataset
        # -----------------------------------------

        try:

            df = pd.read_csv(self.data_path)

        except Exception as e:

            print("ERROR while reading dataset:")
            print(e)

            return False

        print("Dataset loaded successfully.")

        # -----------------------------------------
        # 3. Check number of rows
        # -----------------------------------------

        if len(df) == 0:

            print("ERROR: Dataset is empty.")

            validation_status = False

        else:

            print("Number of rows:", len(df))

        # -----------------------------------------
        # 4. Check required columns
        # -----------------------------------------

        missing_columns = [
            column
            for column in self.required_columns
            if column not in df.columns
        ]

        if missing_columns:

            print(
                "ERROR: Missing columns:",
                missing_columns
            )

            validation_status = False

        else:

            print("All required columns are present.")

        # -----------------------------------------
        # 5. Check unexpected columns
        # -----------------------------------------

        unexpected_columns = [
            column
            for column in df.columns
            if column not in self.required_columns
        ]

        if unexpected_columns:

            print(
                "WARNING: Unexpected columns:",
                unexpected_columns
            )

        # -----------------------------------------
        # 6. Check missing values
        # -----------------------------------------

        missing_values = df.isnull().sum().sum()

        print(
            "Total missing values:",
            missing_values
        )

        if missing_values > 0:

            print(
                "WARNING: Dataset contains missing values."
            )

        # -----------------------------------------
        # 7. Check date column
        # -----------------------------------------

        if "date" in df.columns:

            converted_dates = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            invalid_dates = converted_dates.isnull().sum()

            print(
                "Invalid date values:",
                invalid_dates
            )

            if invalid_dates > 0:

                validation_status = False

                print(
                    "ERROR: Invalid date values found."
                )

        # -----------------------------------------
        # 8. Check target column
        # -----------------------------------------

        if "Appliances" in df.columns:

            if not pd.api.types.is_numeric_dtype(
                df["Appliances"]
            ):

                print(
                    "ERROR: Appliances column is not numeric."
                )

                validation_status = False

            else:

                print(
                    "Appliances column is numeric."
                )

        # -----------------------------------------
        # 9. Check duplicate rows
        # -----------------------------------------

        duplicate_rows = df.duplicated().sum()

        print(
            "Duplicate rows:",
            duplicate_rows
        )

        # -----------------------------------------
        # Final result
        # -----------------------------------------

        print("\nValidation Result:", validation_status)

        return validation_status


if __name__ == "__main__":

    data_path = (
        "data/raw/"
        "energydata_complete.csv"
    )

    validator = DataValidation(
        data_path
    )

    result = validator.validate_data()

    print("\nFinal Validation Result:", result)