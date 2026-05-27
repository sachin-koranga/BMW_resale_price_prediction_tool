from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score

def Linear_model(x_train,y_train,x_test):
    linear_model = LinearRegression()

    # Cross validation
    cv_scores = cross_val_score(
        linear_model,
        x_train,
        y_train,
        cv=5,
        scoring="r2"
    )
    print("Linear Regression CV Mean:",
          cv_scores.mean())
    #final training
    linear_model.fit(x_train,y_train)


    y_pred_log_linear = linear_model.predict(x_test)
    #convert log to original
    y_pred_linear = np.expm1(y_pred_log_linear)
    return y_pred_linear,linear_model

