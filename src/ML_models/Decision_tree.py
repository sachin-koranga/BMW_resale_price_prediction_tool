from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import cross_val_score

def Decision_tree_model(x_train,y_train,x_test):
    model_dt = DecisionTreeRegressor(random_state=42)
    # Cross validation
    cv_scores = cross_val_score(
        model_dt,
        x_train,
        y_train,
        cv=5,
        scoring="r2"
    )
    print("Decision Tree  CV Mean:",
          cv_scores.mean())
    
    #final training
    model_dt.fit(x_train,y_train)

    # make prediction
    y_pred_log_dt = model_dt.predict(x_test)
    # convert log prediction back to original price
    y_pred_dt = np.expm1(y_pred_log_dt)
    return y_pred_dt,model_dt



