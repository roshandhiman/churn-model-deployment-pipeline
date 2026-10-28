# Churn Model Card

## Purpose

This model predicts the probability that a telecom customer will churn. It supports CRM retention decisions and is not intended to make automatic high-impact decisions.

## Data and features

The training data is the public IBM Telco Customer Churn dataset with approximately 7,000 rows. It includes demographic fields, service subscriptions, contract details, tenure, monthly charges, total charges, and churn labels. TotalCharges is cleaned from text into numeric form.

## Pipeline

The artifact is a scikit-learn Pipeline with numeric median imputation and scaling, categorical most-frequent imputation and one-hot encoding, and class-balanced logistic regression. The encoder ignores unseen categories safely.

## Validation

Training uses a stratified 80/20 split and five-fold GridSearchCV using ROC-AUC. The artifact metadata records the data SHA-256 hash, metrics, selected parameters, training timestamp, library versions, and drift reference distribution.

## Threshold policy

A false negative costs 500 because a churner is missed. A false positive costs 100 because an unnecessary retention offer is sent. The system tests thresholds from 0.05 to 0.95 and stores the one with the lowest validation cost. This favors recall because missed churners are more expensive.

## Safe retraining

Retrain using versioned data in staging, compare metrics and expected cost with the active model, run tests, register a new artifact version, deploy as a canary, monitor errors and drift, then promote or roll back.
