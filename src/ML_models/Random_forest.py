from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


def Random_forest_model(x_train,y_train,x_test):
    model_rf = RandomForestRegressor(random_state=42)
    # Cross validation
    cv_scores = cross_val_score(
        model_rf,
        x_train,
        y_train,
        cv=5,
        scoring="r2"
    )
    print("Random Forest  CV Mean:",
          cv_scores.mean())
    
    #final training
    model_rf.fit(x_train,y_train)

    #make prediction
    y_pred_log_rf = model_rf.predict(x_test)
    #convert back log prediction back to original price
    y_pred_rf = np.expm1(y_pred_log_rf)
    return y_pred_rf,model_rf




def best_model(model_rf,x_train,y_train,x_test,param_grid):

    random_search = RandomizedSearchCV(
        estimator=model_rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    random_search.fit(x_train, y_train)

    best_model = random_search.best_estimator_
    #make prediction
    y_pred_log = best_model.predict(x_test)
    # convert back log prediction back to original price
    y_pred = np.expm1(y_pred_log)
    return y_pred,best_model


