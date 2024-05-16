def predict(svc,knn,dtc,rand_clf,ada,xgb,gb,sgb,cat,etc,lgbm,X_train,y_train,X_test,y_test):
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    classifiers = [('Support Vector Classifier', svc), ('KNN', knn),  ('Decision Tree', dtc), ('Random Forest', rand_clf),
               ('Ada Boost', ada), ('XGboost', xgb), ('Gradient Boosting Classifier', gb), ('SGB', sgb),
               ('Cat Boost', cat), ('Extra Trees Classifier', etc), ('LGBM', lgbm)]

    vc = VotingClassifier(estimators = classifiers)
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    # accuracy_score, confusion_matrix and classification_report
    vc_train_acc = accuracy_score(y_train, vc.predict(X_train))
    vc_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of Voting Classifier is : {vc_train_acc}")
    print(f"Test accuracy of Voting Classifier is : {vc_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return vc,vc_test_acc