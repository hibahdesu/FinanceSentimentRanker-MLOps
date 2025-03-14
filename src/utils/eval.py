from sklearn.metrics import r2_score, mean_squared_error

def eval(model, X_train, X_test, y_train, y_test, scaler):
    # Generate predictions for both train and test sets (scaled)
    y_pred_scaled = model.predict(X_test)
    y_pred_train_scaled = model.predict(X_train)

    # Reverse the scaling to get the original target values
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_pred_train = scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1))

    # Reverse the scaling for the actual target values as well
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))

    # Print R2 score and Mean Squared Error (MSE) for the test set
    print("Test Set Evaluation:")
    print(f"R2 Score (Test Set): {r2_score(y_test_original, y_pred)}")
    print(f"Mean Squared Error (Test Set): {mean_squared_error(y_test_original, y_pred)}")
    
    # Print R2 score and MSE for the training set
    print("\nTrain Set Evaluation:")
    print(f"R2 Score (Train Set): {r2_score(y_train_original, y_pred_train)}")
    print(f"Mean Squared Error (Train Set): {mean_squared_error(y_train_original, y_pred_train)}")
