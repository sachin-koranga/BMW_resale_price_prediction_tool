# main.py

# ==============================
# Imports
# ==============================

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Data Processing
from data_loader import loader
from data_cleaning import remove_outliers
from data_preprocessing import pre_processor
from feature_engineering import scaler

# Feature Engineering
from Feature_extraction import (
    num_features_extracter,
    cat_features_extracter
)

# Models
from ML_models.Linear_regression import Linear_model
from ML_models.KNN import KNN_model
from ML_models.Decision_tree import Decision_tree_model
from ML_models.Random_forest import (
    Random_forest_model,
    best_model
)

# Evaluation
from evaluate import (
    model_evaluate,
    train_test_accuracy
)

from compare_metrics_of_diff_models import compare_metrics


# ==============================
# Configurations
# ==============================

warnings.filterwarnings("ignore")

DATA_PATH = (
    r"C:\Users\Vikash\Desktop\bmw_price_prediction_model"
    r"\BMW_resale_price_prediction_tool\data\bmw.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.25

SCALE_COLUMNS = [
    "age",
    "mileage",
    "mpg",
    "engineSize",
    "tax"
]

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# ==============================
# Main Pipeline
# ==============================

def main():

    print("\nLoading Dataset...")

    # Load data
    data = loader(DATA_PATH)

    # Remove outliers
    data = remove_outliers(data)

    print("Dataset Loaded Successfully")

    # ==============================
    # Split Features & Target
    # ==============================

    x = data.drop("price", axis=1)
    y = data["price"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # ==============================
    # Preprocessing
    # ==============================

    print("\nPerforming One-Hot Encoding...")

    x_train_oh, x_test_oh = pre_processor(
        x_train,
        x_test
    )

    print("Applying StandardScaler...")

    x_train_scaled, x_test_scaled = scaler(
        x_train,
        x_test,
        SCALE_COLUMNS
    )

    # ==============================
    # Combine Features
    # ==============================

    x_train_final = pd.concat(
        [
            x_train_scaled.reset_index(drop=True),
            x_train_oh.reset_index(drop=True)
        ],
        axis=1
    )

    x_test_final = pd.concat(
        [
            x_test_scaled.reset_index(drop=True),
            x_test_oh.reset_index(drop=True)
        ],
        axis=1
    )

    # ==============================
    # Feature Selection
    # ==============================

    print("Selecting Important Features...")

    x_train_final, x_test_final = cat_features_extracter(
        x_train_final,
        x_test_final,
        y_train
    )

    # ==============================
    # Log Transformation
    # ==============================

    y_train_log = np.log1p(y_train)

    y_train_original = np.expm1(y_train_log)

    # ==============================
    # Model Training
    # ==============================

    print("\nTraining Models...\n")

    # Linear Regression
    y_pred_linear, linear_model = Linear_model(
        x_train_final,
        y_train_log,
        x_test_final
    )

    # KNN
    y_pred_knn, knn_model = KNN_model(
        x_train_final,
        y_train_log,
        x_test_final
    )

    # Decision Tree
    y_pred_dt, dt_model = Decision_tree_model(
        x_train_final,
        y_train_log,
        x_test_final
    )

    # Random Forest
    y_pred_rf, rf_model = Random_forest_model(
        x_train_final,
        y_train_log,
        x_test_final
    )

    # Tuned Random Forest
    y_pred_best, tuned_rf_model = best_model(
        rf_model,
        x_train_final,
        y_train_log,
        x_test_final,
        PARAM_GRID
    )

    # ==============================
    # Evaluation
    # ==============================

    print("\nEvaluating Best Model...\n")

    metrics = model_evaluate(
        y_test,
        y_pred_best,
        y_train_original
    )

    train_acc, test_acc = train_test_accuracy(
        tuned_rf_model,
        x_train_final,
        x_test_final,
        y_train_log,
        y_test,
        y_train_original
    )

    results_df = compare_metrics(
        x_train_final,
        x_test_final,
        y_train_log,
        y_test,
        y_train_original,
        linear_model,
        knn_model,
        dt_model,
        rf_model,
        tuned_rf_model
    )

    # ==============================
    # Results
    # ==============================

    print("=" * 50)
    print("BEST MODEL PERFORMANCE")
    print("=" * 50)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    print("\nTrain Accuracy :", train_acc)
    print("Test Accuracy  :", test_acc)

    print("\nComparison Table")
    print("-" * 50)

    print(results_df)

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    RESULT_PATH = ARTIFACTS_DIR / "model_results.csv"

    results_df.to_csv(
        RESULT_PATH,
        index=True
    )

# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    main()