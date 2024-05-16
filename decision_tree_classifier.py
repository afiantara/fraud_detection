def predict(X_train,y_train,X_test,y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    # accuracy_score, confusion_matrix and classification_report
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
    dtc_test_acc = accuracy_score(y_test, y_pred)
    print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
    print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # hyper parameter tuning
    from sklearn.model_selection import GridSearchCV
    grid_params = {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [3, 5, 7, 10],
        'min_samples_split' : range(2, 10, 1),
        'min_samples_leaf' : range(2, 10, 1)
    }
    grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
    grid_search.fit(X_train, y_train)
    # best parameters and best score
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    # best estimator 
    dtc = grid_search.best_estimator_
    y_pred = dtc.predict(X_test)

    # accuracy_score, confusion_matrix and classification_report
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
    dtc_test_acc = accuracy_score(y_test, y_pred)
    print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
    print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return dtc,dtc_test_acc
