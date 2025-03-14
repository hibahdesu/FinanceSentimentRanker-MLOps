from sklearn.ensemble import RandomForestRegressor

s = 101

def train_rf_model(X_train_tf, y_train):
    model = RandomForestRegressor(random_state=s)  # Use SVR for regression task
    model.fit(X_train_tf, y_train)
    return model
