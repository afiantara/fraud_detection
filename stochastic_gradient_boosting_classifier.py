def predict(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    sgb = GradientBoostingClassifier(subsample = 0.90, max_features = 0.70)
    sgb.fit(X_train, y_train)
    # accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier
    sgb_acc = accuracy_score(y_test, sgb.predict(X_test))
    print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
    print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")

    return sgb,sgb_acc