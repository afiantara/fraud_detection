def predict(X_train,y_train,X_test,y_test):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    lgbm = LGBMClassifier(learning_rate = 1)
    lgbm.fit(X_train, y_train)

    # accuracy score, confusion matrix and classification report of lgbm classifier

    lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

    print(f"Training Accuracy of LGBM Classifier is {accuracy_score(y_train, lgbm.predict(X_train))}")
    print(f"Test Accuracy of LGBM Classifier is {lgbm_acc} \n")

    print(f"{confusion_matrix(y_test, lgbm.predict(X_test))}\n")
    print(classification_report(y_test, lgbm.predict(X_test)))

    return lgbm,lgbm_acc