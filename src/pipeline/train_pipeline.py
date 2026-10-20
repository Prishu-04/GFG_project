import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            train_path, test_path = DataIngestion().initiate_data_ingestion()

            train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
                train_path, test_path
            )

            best_model_name, best_score = ModelTrainer().initiate_model_trainer(
                train_arr, test_arr
            )

            logging.info(f"Training pipeline finished: {best_model_name} ({best_score:.4f})")
            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    name, score = TrainPipeline().run_pipeline()
    print(f"Best model: {name} | accuracy: {score:.4f}")