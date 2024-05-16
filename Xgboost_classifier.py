def predict(X_train,y_train,X_test,y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import GridSearchCV
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    # accuracy_score, confusion_matrix and classification_report
    xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
    xgb_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of XgBoost is : {xgb_train_acc}")
    print(f"Test accuracy of XgBoost is : {xgb_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 10, 1)}

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5,  verbose=3,n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # best estimator 
    xgb = grid_search.best_estimator_
    y_pred = xgb.predict(X_test)
    # accuracy_score, confusion_matrix and classification_report
    xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
    xgb_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of XgBoost is : {xgb_train_acc}")
    print(f"Test accuracy of XgBoost is : {xgb_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return xgb,xgb_test_acc
