def predict(X_train,y_train,X_test,y_test):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier

    rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
    rand_clf.fit(X_train, y_train)

    y_pred = rand_clf.predict(X_test)

    # accuracy_score, confusion_matrix and classification_report

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
    rand_clf_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
    print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return rand_clf,rand_clf_test_acc

    