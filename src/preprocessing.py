def preprocess_data(df):
    df = df.drop_duplicates()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return df
