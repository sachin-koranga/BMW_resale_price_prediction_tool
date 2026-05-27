from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
import pandas as pd


#for nuerical features
def num_features_extracter(x_train,y_train,num_features):
    correlations = {
        feature:pearsonr(x_train[feature],
        y_train)[0]
        for feature in num_features
    }
    correlations_df = pd.DataFrame(list(correlations.items()),columns=["feature","person_correlation"])
    return correlations_df.sort_values(by ="person_correlation",ascending=False)
    

#for catigoral columns


import pandas as pd
from scipy.stats import chi2_contingency


def cat_features_extracter(
    x_train,
    x_test,
    y_train,
    alpha=0.001,
    min_frequency=0.01
):

    x_train = x_train.copy()
    x_test = x_test.copy()

    # -----------------------------------
    # Create target bins
    # -----------------------------------

    x_train["price_bin"] = pd.qcut(
        y_train,
        q=4,
        labels=False
    )

    # -----------------------------------
    # Detect dummy columns automatically
    # -----------------------------------

    cat_cols = [
        col for col in x_train.columns
        if (
            x_train[col].dropna().isin([0, 1]).all()
            and col != "price_bin"
        )
    ]

    removed_cols = []
    kept_cols = []

    # -----------------------------------
    # Chi-Square Selection
    # -----------------------------------

    for col in cat_cols:

        # frequency of positive class
        freq = x_train[col].mean()

        table = pd.crosstab(
            x_train[col],
            x_train["price_bin"]
        )

        stat, p, dof, exp = chi2_contingency(table)

        # Remove only VERY weak + VERY sparse columns
        if p > alpha and freq < min_frequency:

            removed_cols.append(col)

        else:
            kept_cols.append(col)

    # -----------------------------------
    # Drop weak columns
    # -----------------------------------

    x_train.drop(
        columns=removed_cols,
        inplace=True,
        errors="ignore"
    )

    x_test.drop(
        columns=removed_cols,
        inplace=True,
        errors="ignore"
    )

    # remove helper column
    x_train.drop(
        columns=["price_bin"],
        inplace=True
    )

    # -----------------------------------
    # Debug Info
    # -----------------------------------

    print("\n========== Feature Selection ==========")

    print(f"\nTotal dummy columns: {len(cat_cols)}")

    print(f"\nRemoved columns: {len(removed_cols)}")

    print(f"\nKept columns: {len(kept_cols)}")

    print("\nRemoved Features:")
    print(removed_cols)

    return x_train, x_test