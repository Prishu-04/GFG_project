import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    min_accuracy: float = 0.6


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
            }

            params = {
                "Logistic Regression": {"C": [0.1, 1, 10]},
                "Decision Tree": {"max_depth": [5, 10, 15]},
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 15]},
                "HistGradientBoosting": {"learning_rate": [0.05, 0.1], "max_iter": [200, 300]},
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < self.model_trainer_config.min_accuracy:
                raise Exception(f"No good model found (best accuracy = {best_model_score:.3f})")

            logging.info(f"Best model: {best_model_name} (accuracy = {best_model_score:.4f})")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            y_pred = best_model.predict(X_test)
            print(classification_report(
                y_test, y_pred, target_names=["Low", "Moderate", "High", "Severe"]
            ))

            return best_model_name, best_model_score
        except Exception as e:
            raise CustomException(e, sys)