import pandas as pd

#remove outliers
def remove_outliers(new_data):
    Q1 = new_data['price'].quantile(0.25)
    Q3 = new_data['price'].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 -1.5*IQR
    upper = Q3 +1.5*IQR
    new_data = new_data[(new_data['price']>=lower)&(new_data['price'] < upper)]
    
    # creating new age columns for cars
    new_data["age"]= 2025 - new_data["year"]
    new_data.drop(columns=["year"],axis=1,inplace=True)
    return new_data

