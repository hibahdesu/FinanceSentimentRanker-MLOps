from sklearn.model_selection import train_test_split

s = 101
def split_data(df):
    X = df['news']
    y = df['compound']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
    return X_train, X_test, y_train, y_test
