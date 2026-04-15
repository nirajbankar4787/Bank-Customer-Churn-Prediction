from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='templates')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
if not hasattr(model, 'predict'):
    raise AttributeError('Loaded model does not implement predict()')


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['balance_per_product'] = df['balance'] / df['products_number'].replace(0, np.nan)
    df['balance_per_product'] = df['balance_per_product'].fillna(0)

    df['salary_balance_ratio'] = df['estimated_salary'] / df['balance'].replace(0, np.nan)
    df['salary_balance_ratio'] = df['salary_balance_ratio'].replace([np.inf, -np.inf], np.nan)
    if df['salary_balance_ratio'].isna().all():
        df['salary_balance_ratio'] = df['salary_balance_ratio'].fillna(0.0)
    else:
        df['salary_balance_ratio'] = df['salary_balance_ratio'].fillna(df['salary_balance_ratio'].median())

    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    )

    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 0, 2, 5, 10, 100],
        labels=['0', '1-2', '3-5', '6-10', '10+']
    )

    df['high_balance'] = (df['balance'] > 50000.0).astype(int)

    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'credit_score': float(request.form['credit_score']),
            'country': request.form['country'],
            'gender': request.form['gender'],
            'age': int(request.form['age']),
            'tenure': int(request.form['tenure']),
            'balance': float(request.form['balance']),
            'products_number': int(request.form['products_number']),
            'credit_card': int(request.form['credit_card']),
            'active_member': int(request.form['active_member']),
            'estimated_salary': float(request.form['estimated_salary'])
        }

        df = pd.DataFrame([input_data])
        df = apply_feature_engineering(df)

        predictions = model.predict(df)
        if len(predictions) == 0:
            raise ValueError('Model returned no predictions')

        pred = int(predictions[0])
        probability = None

        if hasattr(model, 'predict_proba'):
            try:
                proba = np.asarray(model.predict_proba(df))
                if proba.ndim == 2 and proba.shape[1] > 1:
                    probability = float(proba[0, 1])
                elif proba.ndim == 1:
                    probability = float(proba[0])
            except Exception:
                probability = None

        prediction_text = f"Churn Prediction: {'YES' if pred == 1 else 'NO'}"
        probability_text = (
            f"Churn Probability: {probability:.1%}"
            if probability is not None else 'Churn Probability: unavailable'
        )
        risk_text = 'High Risk' if probability is not None and probability > 0.5 else 'Low Risk'

    except Exception as e:
        prediction_text = 'Error in prediction'
        probability_text = str(e)
        risk_text = 'N/A'

    return render_template('index.html',
                           prediction_text=prediction_text,
                           probability_text=probability_text,
                           risk_text=risk_text)

if __name__ == '__main__':
    app.run(debug=True)