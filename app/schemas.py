from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    age: float = Field(..., ge=18, le=100)
    monthly_income: float = Field(..., gt=0)
    employment_type: str
    credit_score: float = Field(..., ge=300, le=900)
    existing_emi: float = Field(..., ge=0)
    loan_amount: float = Field(..., gt=0)
    loan_tenure: float = Field(..., gt=0)
    previous_default: int = Field(..., ge=0, le=1)
    loan_to_income: float
    emi_to_income: float


class PredictionResponse(BaseModel):
    default_probability: float
    risk_class: str
    prediction: int