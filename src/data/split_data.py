from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw_data" / "raw.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
TARGET_COLUMN = "silica_concentrate"
PARAMS_PATH = ROOT_DIR / "params.yaml"


def main() -> None:
    with PARAMS_PATH.open(encoding="utf-8") as params_file:
        split_params = yaml.safe_load(params_file)["split"]

    data = pd.read_csv(RAW_DATA_PATH)
    if data.columns[-1] != TARGET_COLUMN:
        raise ValueError(f"The last column must be {TARGET_COLUMN!r}.")

    feature_columns = [column for column in data.columns[:-1] if column != "date"]
    features = data[feature_columns]
    target = data[TARGET_COLUMN]

    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in feature_columns):
        raise TypeError("All feature columns must be numeric after removing date.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False, header=True)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False, header=True)

    print(f"Saved {len(X_train)} training rows and {len(X_test)} test rows.")


if __name__ == "__main__":
    main()