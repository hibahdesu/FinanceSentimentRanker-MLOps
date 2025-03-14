import joblib

def load_saved_model(model_name):
    model = joblib.load(f'saved_models/{model_name}.pkl')  # Load model
    vectorizer = joblib.load('saved_models/vectorizer.pkl')  # Load vectorizer
    scaler = joblib.load('saved_models/scaler.pkl')  # Load scaler
    return model, vectorizer, scaler
