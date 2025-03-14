from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.text_cleaning import cleanText

def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(min_df=3, stop_words='english', max_features=10000, preprocessor=cleanText)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)
    return X_train_tf, X_test_tf, vectorizer
