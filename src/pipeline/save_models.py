import joblib

def save_models(model, vectorizer, scaler, model_name):
    joblib.dump(model, f'models/{model_name}.pkl')  # Save model
    joblib.dump(vectorizer, 'models/vectorizer.pkl')  # Save vectorizer
    joblib.dump(scaler, 'models/scaler.pkl')  # Save scaler
