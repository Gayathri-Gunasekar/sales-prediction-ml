import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import feature_engineering, get_preprocessor

# Load data
data_path = "data/retail_sales_dataset.csv"
df = pd.read_csv(data_path)

# Feature Engineering
df = feature_engineering(df)

# Features & Target
X = df[['Gender', 'Age', 'Product Category',
        'Quantity', 'Price per Unit', 'Month', 'Year']]
y = df['Total Amount']

# Get preprocessor
preprocessor = get_preprocessor()

# Build pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model_pipeline, "model/sales_model.pkl")

print("Model saved successfully!")
