import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")

        try:
            # 3 artifacts .csv data paths
            raw_data_path = self.data_ingestion_config.raw_data_path
            train_data_path = self.data_ingestion_config.train_data_path
            test_data_path = self.data_ingestion_config.test_data_path

            # retrieve dataframe
            logging.info("Read dataset as dataframe")
            df = pd.read_csv("notebook/data/stud.csv")

            # make artifacts dir(for storing data)
            artifacts_dir_path = os.path.dirname(train_data_path)
            os.makedirs(artifacts_dir_path, exist_ok=True)

            # put raw data to artifacts\data.csv
            df.to_csv(raw_data_path, index=False, header=True)

            # train test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            # put train data and test data to "artifacts\train.csv" and "artifacts\test.csv"
            train_set.to_csv(train_data_path, index=False, header=True)
            train_set.to_csv(test_data_path, index=False, header=True)
            logging.info("Data ingestion completed")

            # directly pass to DataTransformation in future ...
            return (train_data_path, test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # putting external data to artifacts\train.csv and test.csv
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # making or updating artifacts\preprocessor.pkl
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )
