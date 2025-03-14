from sklearn.preprocessing import LabelEncoder

def encode_sentiment(df):
    encoder = LabelEncoder()
    df['sentiment'] = encoder.fit_transform(df['sentiment'])
    return df, encoder
