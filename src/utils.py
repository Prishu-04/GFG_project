import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException


def save_object(file_path, obj):

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "wb") as file_obj:

            pickle.dump(obj, file_obj)

    except Exception as e:

        raise CustomException(e, sys)


def load_object(file_path):

    try:

        with open(file_path, "rb") as file_obj:

            return pickle.load(file_obj)

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):

    try:

        report = {}

        for model_name in models:

            model = models[model_name]

            parameters = param[model_name]

            if parameters:

                gs = GridSearchCV(
                    model,
                    parameters,
                    cv=3,
                    scoring="accuracy",
                    n_jobs=-1
                )

                gs.fit(
                    X_train,
                    y_train
                )

                model.set_params(
                    **gs.best_params_
                )

            model.fit(
                X_train,
                y_train
            )

            y_test_pred = model.predict(
                X_test
            )

            test_model_score = accuracy_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score

            print(
                f"{model_name}: "
                f"{test_model_score:.4f}"
            )

        return report

    except Exception as e:

        raise CustomException(e, sys)