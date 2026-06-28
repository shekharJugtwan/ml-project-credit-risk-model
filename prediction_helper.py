import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

MODEL_PATH  = 'artifacts/model_data.joblib'

#Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

def prepare_df(age, income, loan_amount, loan_tenure_months, avg_dpd_per_deliquency,
           delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):

    input_data = {
            'age': age,
            'loan_tenure_months': loan_tenure_months,
            'number_of_open_accounts': num_open_accounts,
            'credit_utilization_ratio': credit_utilization_ratio,
            'loan_to_income': loan_amount/income if income >0 else 0,
            'delinquency_ratio': delinquency_ratio,
            'avg_dpd_per_deliquency': avg_dpd_per_deliquency,
            'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
            'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
            'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
            'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
            'loan_purpose_Personal' : 1 if loan_purpose == 'Personal' else 0,
            'loan_type_Unsecured' : 1 if loan_type == 'Unsecured' else 0,

        # add aditional fields

        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding':1,
        'bank_balance_at_application':1,
        'number_of_closed_accounts':1,
        'enquiry_count':1
            }
    df = pd.DataFrame([input_data])

    print("Columns in df:")
    print(df.columns.tolist())

    print("\nColumns expected by scaler:")
    print(cols_to_scale)

    missing = [col for col in cols_to_scale if col not in df.columns]
    print("\nMissing columns:")
    print(missing)

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]
    return df

def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_deliquency,
                   delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                   residence_type, loan_purpose, loan_type):


    input_df = prepare_df(age, income, loan_amount, loan_tenure_months, avg_dpd_per_deliquency,
                   delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                   residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score( input_df)

    return probability, credit_score, rating

def calculate_credit_score(input_df, base_score=300, scale_length=600):
    # Predict probability of default
    default_probability = model.predict_proba(input_df)[0][1]

    # Probability of not defaulting
    non_default_probability = 1 - default_probability

    # Convert probability into a credit score (300-900)
    credit_score = base_score + (non_default_probability * scale_length)

    # Ensure score stays within range
    credit_score = max(300, min(900, credit_score))

    # Rating based on score
    def get_rating(score):
        if 300 <= score < 500:
            return "Poor"
        elif 500 <= score < 650:
            return "Average"
        elif 650 <= score < 750:
            return "Good"
        else:
            return "Excellent"

    rating = get_rating(credit_score)

    return (
        float(default_probability),
        int(round(credit_score)),
        rating
    )