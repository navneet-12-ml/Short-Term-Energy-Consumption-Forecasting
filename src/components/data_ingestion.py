import pandas as pd


class DataIngestion:

    def __init__(self, data_path):
        self.data_path = data_path

    def initiate_data_ingestion(self):

        df = pd.read_csv(self.data_path)

        print("Data loaded successfully")
        print("Shape:", df.shape)

        return df


if __name__ == "__main__":

    data_path = "data/raw/energydata_complete.csv"

    ingestion = DataIngestion(data_path)

    df = ingestion.initiate_data_ingestion()