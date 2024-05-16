def predict(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    etc = ExtraTreesClassifier()
    etc.fit(X_train, y_train)

    # accuracy score, confusion matrix and classification report of extra trees classifier

    etc_acc = accuracy_score(y_test, etc.predict(X_test))

    print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
    print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")

    return etc,etc_acc