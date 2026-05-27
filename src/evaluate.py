import numpy as np
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def model_evaluate(y_test,y_pred,y_train_original):
    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print("r2_score:",r2)
    print("mae: ",mae)
    print("mse: ", mse)
    print("rmse: ",rmse)
    print(rmse/(y_train_original.mean()))
    return {
        "r2_score": r2,
        "mae": mae,
        "mse": mse,
        "rmse":rmse
    }



def train_test_accuracy(best_model,x_train,x_test,y_train,y_test,y_train_original):
    train_pred_log = best_model.predict(x_train)

    train_pred = np.expm1(train_pred_log)

    test_pred_log = best_model.predict(x_test)

    test_pred = np.expm1(test_pred_log)

    train_acc = r2_score(y_train_original,train_pred)
    test_acc = r2_score(y_test,test_pred)
    #print("train accuracy ",train_acc)
    #print("test accuracy ",test_acc)
    return train_acc,test_acc