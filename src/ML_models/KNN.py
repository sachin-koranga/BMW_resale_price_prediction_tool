from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.model_selection import cross_val_score

def KNN_model(x_train,y_train,x_test):
    model_knn = KNeighborsRegressor(n_neighbors=5)
    # Cross validation
    cv_scores = cross_val_score(
        model_knn,
        x_train,
        y_train,
        cv=5,
        scoring="r2"
    )
    print("KNN  CV Mean:",
          cv_scores.mean())
    
    #final training
    model_knn.fit(x_train,y_train)

    #make prediction
    y_pred_log_knn = model_knn.predict(x_test)
    #convert log prediction back to original price
    y_pred_knn = np.expm1(y_pred_log_knn)
    return y_pred_knn,model_knn
