import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def feature_engineering(df: pd.DataFrame):
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    return df


def get_preprocessor():
    categorical_cols = ['Gender', 'Product Category']
    numerical_cols = ['Age', 'Quantity', 'Price per Unit', 'Month', 'Year']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    return preprocessor
