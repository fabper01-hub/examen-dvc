from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
PARAMS_PATH = ROOT_DIR / "params.yaml"


def main() -> None:
    with PARAMS_PATH.open(encoding="utf-8") as params_file:
        search_params = yaml.safe_load(params_file)["grid_search"]

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze("columns")

    estimator = GradientBoostingRegressor(random_state=search_params["random_state"])
    parameter_grid = {
        "n_estimators": search_params["n_estimators"],
        "learning_rate": search_params["learning_rate"],
        "max_depth": search_params["max_depth"],
        "min_samples_leaf": search_params["min_samples_leaf"],
    }
    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        cv=search_params["cv"],
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_params_, MODELS_DIR / "best_params.pkl")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV MSE: {-search.best_score_:.6f}")


if __name__ == "__main__":
    main()