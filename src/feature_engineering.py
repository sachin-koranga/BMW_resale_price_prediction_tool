from sklearn.preprocessing import StandardScaler
import pandas as pd

def scaler(x_train,x_test,cols = ["age","mileage","mpg","engineSize","tax"]):
    scaler = StandardScaler()
    x_train_st = scaler.fit_transform(x_train[cols])
    x_test_st = scaler.transform(x_test[cols])
    

    x_train_st = pd.DataFrame(x_train_st,columns=cols,index=x_train.index)
    x_test_st = pd.DataFrame(x_test_st,columns=cols,index=x_test.index)

    return x_train_st , x_test_st