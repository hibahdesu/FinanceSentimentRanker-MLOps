import sys
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Add both src and utils to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from src.pipeline.load_data import load_data
from src.pipeline.clean_data import clean_data
from src.pipeline.process_text import process_text
from src.pipeline.encode_sentiment import encode_sentiment
from src.pipeline.split_data import split_data
from src.pipeline.vectorize_data import vectorize_data
from src.pipeline.train_logistic_model import train_logistic_model
from src.pipeline.train_svc_model import train_svc_model
from src.pipeline.train_rf_model import train_rf_model
from src.pipeline.save_models import save_models
from src.utils.eval import eval

def evaluate_model(model, X_train_tf, X_test_tf, y_train_scaled, y_test_scaled, scaler_y):
    """
    Helper function to evaluate model performance using R2 and Mean Squared Error.
    """
    y_pred = model.predict(X_test_tf)
    r2 = r2_score(y_test_scaled, y_pred)
    mse = mean_squared_error(y_test_scaled, y_pred)
    print(f"Model Evaluation: R2 = {r2:.4f}, MSE = {mse:.4f}")
    return r2, mse

def train_and_evaluate_models(X_train_tf, X_test_tf, y_train_scaled, y_test_scaled, scaler_y):
    """
    Function to train and evaluate multiple models and return the best one based on R2 score.
    """
    models = {
        "Linear Regression": train_logistic_model(X_train_tf, y_train_scaled),
        # "Support Vector Regressor": train_svc_model(X_train_tf, y_train_scaled),
        # "Random Forest Regressor": train_rf_model(X_train_tf, y_train_scaled)
    }

    model_scores = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        r2, mse = evaluate_model(model, X_train_tf, X_test_tf, y_train_scaled, y_test_scaled, scaler_y)
        model_scores[model_name] = {"r2_score": r2, "mse": mse}

    best_model_name = max(model_scores, key=lambda model: model_scores[model]['r2_score'])
    print(f"The best model is {best_model_name} with R2 = {model_scores[best_model_name]['r2_score']:.4f}")
    
    return models[best_model_name], best_model_name, model_scores

def main():
    try:
        # Load and clean data
        df = load_data()
        df = clean_data(df)
        df = process_text(df)

        # Split data into train and test
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Scale the target variable (y) using MinMaxScaler
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

        # Vectorize the data
        X_train_tf, X_test_tf, vectorizer = vectorize_data(X_train, X_test)
        
        # Train and evaluate models, selecting the best one
        best_model, best_model_name, model_scores = train_and_evaluate_models(
            X_train_tf, X_test_tf, y_train_scaled, y_test_scaled, scaler_y
        )

        # Save the best model with a consistent name ('best_model') so that it will always be updated
        print(f"Saving the best model as 'best_model', {best_model_name}")
        save_models(best_model, vectorizer, scaler_y, 'best_model')

        # Output the final results
        print(f"Best model, vectorizer, and scaler saved as 'best_model'.")
        print(f"Model Evaluation Summary: {model_scores}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
