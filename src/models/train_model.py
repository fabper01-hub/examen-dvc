from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
MODELS_DIR = ROOT_DIR / "models"


def main() -> None:
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze("columns")
    best_params = joblib.load(MODELS_DIR / "best_params.pkl")

    model = GradientBoostingRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "gradient_boosting_model.pkl")
    print("Saved trained model.")


if __name__ == "__main__":
    main()