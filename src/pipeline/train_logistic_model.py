from sklearn.linear_model import LinearRegression


def train_logistic_model(X_train_tf, y_train):
    model = LinearRegression()  
    model.fit(X_train_tf, y_train)
    return model


