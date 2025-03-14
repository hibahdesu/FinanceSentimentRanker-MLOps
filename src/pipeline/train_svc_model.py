from sklearn.svm import SVR

def train_svc_model(X_train_tf, y_train):
    model = SVR(C=0.1)  # Use SVR for regression task
    model.fit(X_train_tf, y_train)
    return model