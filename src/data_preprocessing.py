from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def pre_processor(x_train,x_test):
    one_hot = OneHotEncoder(drop="first",sparse_output=False,)
    x_train_oh = one_hot.fit_transform(x_train[["transmission","fuelType","model"]])
    x_test_oh = one_hot.transform(x_test[["transmission","fuelType","model"]])

    x_train_index = x_train.index

    x_train_oh = pd.DataFrame(x_train_oh,columns=one_hot.get_feature_names_out(),index=x_train.index)
    x_test_oh = pd.DataFrame(x_test_oh,columns=one_hot.get_feature_names_out(),index=x_test.index)

    return x_train_oh , x_test_oh