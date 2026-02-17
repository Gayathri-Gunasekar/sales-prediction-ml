import joblib
import pandas as pd



model = joblib.load("model/sales_model.pkl")

def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return float(prediction[0])


if __name__ == "__main__":
    sample_input = {
        "Gender": "Male",
        "Age": 30,
        "Product Category": "Beauty",
        "Quantity": 2,
        "Price per Unit": 50,
        "Month": 5,
        "Year": 2023
    }

    result = predict(sample_input)
    print("Predicted Sales:", result)
