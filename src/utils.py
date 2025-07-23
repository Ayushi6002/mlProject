import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException  # Make sure this is implemented properly

def save_object(file_path, obj):
    """
    Save any Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple ML models using GridSearchCV and return a report.
    """
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            params = param[model_name]

            # Perform Grid Search with 3-fold cross-validation
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)  # ✅ Fixed typo: .fil → .fit

            # Set model to best parameters found by grid search
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R^2 score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test R^2 score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
