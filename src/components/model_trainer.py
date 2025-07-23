import sys
import os
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data...")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K Neighbors Regressor": {
                    "n_neighbors": [5, 7, 9, 11]
                },
                "XGBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.001],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            model_report = {}
            best_score = -1
            best_model_name = None
            best_model = None

            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                param_grid = params.get(model_name, {})
                if param_grid:
                    gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
                    gs.fit(X_train, y_train)
                    final_model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    final_model = model

                y_pred = final_model.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_report[model_name] = score

                logging.info(f"{model_name} R2 Score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = final_model

            if best_score < 0.6:
                raise CustomException("No suitable model found with R2 score above 0.6.")

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_score

        except Exception as e:
            raise CustomException(e, sys)
