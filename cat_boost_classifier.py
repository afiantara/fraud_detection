def predict(X_train,y_train,X_test,y_test):
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    cat = CatBoostClassifier(iterations=10)
    cat.fit(X_train, y_train)
    # accuracy score, confusion matrix and classification report of cat boost
    cat_acc = accuracy_score(y_test, cat.predict(X_test))

    print(f"Training Accuracy of Cat Boost Classifier is {accuracy_score(y_train, cat.predict(X_train))}")
    print(f"Test Accuracy of Cat Boost Classifier is {cat_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, cat.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, cat.predict(X_test))}")

    return cat,cat_acc