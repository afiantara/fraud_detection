def predict(X_train,y_train,X_test,y_test):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    # accuracy_score, confusion_matrix and classification_report
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
    svc_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
    print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return svc,svc_test_acc