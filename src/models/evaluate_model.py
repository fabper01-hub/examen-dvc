import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"


def main() -> None:
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze("columns")
    model = joblib.load(MODELS_DIR / "gradient_boosting_model.pkl")

    predictions = model.predict(X_test)
    scores = {
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    prediction_data = pd.DataFrame(
        {"actual_silica_concentrate": y_test, "predicted_silica_concentrate": predictions}
    )
    prediction_data.to_csv(PROCESSED_DATA_DIR / "predictions.csv", index=False)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "scores.json").open("w", encoding="utf-8") as scores_file:
        json.dump(scores, scores_file, indent=2)

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()