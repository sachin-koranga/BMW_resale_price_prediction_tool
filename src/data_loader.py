import pandas as pd


def loader(path="C:BMW_resale_price_prediction_tool\\data\\bmw.csv"):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    return df
