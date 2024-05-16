def predict(X_train,y_train,X_test,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    knn = KNeighborsClassifier(n_neighbors = 30)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # accuracy_score, confusion_matrix and classification_report
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
    knn_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of KNN is : {knn_train_acc}")
    print(f"Test accuracy of KNN is : {knn_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return knn,knn_test_acc
