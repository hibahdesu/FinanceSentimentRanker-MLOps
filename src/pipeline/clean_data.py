def clean_data(df):
    # Drop rows with null values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    return df  