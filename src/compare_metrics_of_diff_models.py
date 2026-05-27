import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)



def compare_metrics(x_train,x_test,y_train,y_test,y_train_original,Linear_model,KNN_model,Decision_tree_model,Random_forest_model,best_model):
    results = []
    models = {
        "Linear Regression": Linear_model,
        "KNN": KNN_model,
        "Decision Tree": Decision_tree_model,
        "Random Forest": Random_forest_model,
        "random_search": best_model
    }

    for name, model in models.items():

        # Train
        model.fit(x_train, y_train)

        # Predict
        train_pred_log = model.predict(x_train)
        test_pred_log = model.predict(x_test)

        # Convert back from log scale
        train_pred = np.expm1(train_pred_log)
        test_pred = np.expm1(test_pred_log)

        y_train_original = y_train_original
        y_test_original = y_test

        # Metrics
        train_r2 = r2_score(y_train_original, train_pred)

        test_r2 = r2_score(y_test_original, test_pred)

        mae = mean_absolute_error(
            y_test_original,
            test_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test_original,
                test_pred
            )
        )
        cv_scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=5,
        scoring="r2"
    )

        cv_mean = cv_scores.mean()

        gap = train_r2 - test_r2

        # Save results
        results.append({
            "Model": name,
            "Train R2": round(train_r2, 3),
            "Test R2": round(test_r2, 3),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "RMSE/y_train_mean": round(rmse / y_train_original.mean(), 2),
            "CV Mean": round(cv_mean, 3),
            "Overfit Gap": round(gap, 3),

        })


    # Create the results DataFrame
    results_df = pd.DataFrame(results)

    # Rename columns for clarity
    results_df = results_df.rename(columns={
        "Train R2": "Train R",
        "Test R2": "Test R",
        "CV Mean": "CV R",
        "Overfit Gap": "Gap",
        "RMSE/y_test_mean": "RMSE %"
    })

    # Sort by performance and reset index for Ranking
    results_df = results_df.sort_values(by="Test R", ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1
    results_df.index.name = "Rank"

    # Round numeric values
    results_df = results_df.round({
        "Train R": 3,
        "Test R": 3,
        "CV R": 3,
        "Gap": 3,
        "MAE": 3,
        "RMSE": 3,
        "RMSE %": 2
    })
    return results_df
    # Force the table to stay horizontal with a scrollbar using CSS
    #display(results_df.style.set_table_attributes('style="display: inline-block; white-space: nowrap; overflow-x: auto; max-width: 100%;"'))

