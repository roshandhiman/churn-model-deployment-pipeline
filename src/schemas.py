from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

YesNo = str
Gender = str
Internet = str
MultipleLines = str
AddOn = str
Contract = str
Payment = str


class Customer(BaseModel):
    gender: Gender
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: Internet
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: Payment
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_matches_tenure(cls, value: float, info):
        tenure = info.data.get("tenure", 0)
        if tenure == 0 and value > 0:
            raise ValueError("TotalCharges must be 0 when tenure is 0")
        return value


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class DriftRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=5000)
