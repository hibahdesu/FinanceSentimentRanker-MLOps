import joblib

def load_saved_model(model_name):
    model = joblib.load(f'models/{model_name}.pkl')  # Load model
    vectorizer = joblib.load('models/vectorizer.pkl')  # Load vectorizer
    scaler = joblib.load('models/scaler.pkl')  # Load scaler
    return model, vectorizer, scaler
