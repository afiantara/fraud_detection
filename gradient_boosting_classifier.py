def predict(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    # accuracy score, confusion matrix and classification report of gradient boosting classifier
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
    print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")

    return gb,gb_acc