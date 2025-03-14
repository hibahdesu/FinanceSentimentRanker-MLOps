import joblib

def save_models(model, vectorizer, scaler, model_name):
    joblib.dump(model, f'saved_models/{model_name}.pkl')  # Save model
    joblib.dump(vectorizer, 'saved_models/vectorizer.pkl')  # Save vectorizer
    joblib.dump(scaler, 'saved_models/scaler.pkl')  # Save scaler
