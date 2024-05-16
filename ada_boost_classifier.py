def predict(X_train,y_train,X_test,y_test):
    import decision_tree_classifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.ensemble import AdaBoostClassifier
    
    dtc,dtc_test_acc = decision_tree_classifier.predict(X_train,y_train,X_test,y_test)

    ada = AdaBoostClassifier(estimator = dtc)
    parameters = {
        'n_estimators' : [50, 70, 90, 120, 180, 200],
        'learning_rate' : [0.001, 0.01, 0.1, 1, 10],
        'algorithm' : ['SAMME', 'SAMME.R']
    }

    grid_search = GridSearchCV(ada, parameters, n_jobs = -1, cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)
    # best parameter and best score
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    # best estimator 
    ada = grid_search.best_estimator_
    y_pred = ada.predict(X_test)
    # accuracy_score, confusion_matrix and classification_report
    ada_train_acc = accuracy_score(y_train, ada.predict(X_train))
    ada_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of Ada Boost is : {ada_train_acc}")
    print(f"Test accuracy of Ada Boost is : {ada_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return ada,ada_test_acc