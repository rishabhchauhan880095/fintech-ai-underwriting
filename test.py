import xgboost
import sklearn
import joblib
import pandas as pd

print("XGBoost:", xgboost.__version__)
print("Sklearn:", sklearn.__version__)

model = joblib.load("model/underwriting_model.pkl")

input_data = pd.DataFrame([{
    "age": 32,
    "monthly_income": 50000,
    "employment_type": "salaried",
    "credit_score": 720,
    "existing_emi": 10000,
    "loan_amount": 300000,
    "loan_tenure": 24,
    "previous_default": 0,
    "loan_to_income": 0.5,
    "emi_to_income": 0.2
}])

probability = model.predict_proba(input_data)[0, 1]

print("Probability:", probability)
print("Prediction:", int(probability >= 0.379))