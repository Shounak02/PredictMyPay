import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import matplotlib.pyplot as plt

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df[~df.isin(['?']).any(axis=1)]
    df['income'] = df['income'].map({'<=50K': 400000, '>50K': 800000})
    
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('income', axis=1)
    y = df['income']
    return train_test_split(X, y, test_size=0.2, random_state=42), le_dict

def train_and_select_best_model(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('-inf')
    best_model_name = ""
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"RMSE": rmse, "R2": r2}
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    print("🔍 Model Comparison (on test set):")
    for name, scores in results.items():
        print(f"{name}: RMSE = {scores['RMSE']:.2f}, R² = {scores['R2']:.4f}")
    print(f"\n✅ Best model selected: {best_model_name} (R² = {best_score:.4f})")
    
    return best_model, results

def forecast_salary(model, user_data, years=3, annual_growth=0.07):
    current_salary = model.predict([user_data])[0]
    return [current_salary * ((1 + annual_growth) ** i) for i in range(years + 1)]



def train_and_select_best_model(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = float('-inf')
    best_model_name = ""
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"RMSE": rmse, "R2": r2}

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    # 🔍 Console view
    print("🔍 Model Comparison (on test set):")
    for name, scores in results.items():
        print(f"{name}: RMSE = {scores['RMSE']:.2f}, R² = {scores['R2']:.4f}")
    print(f"\n✅ Best model selected: {best_model_name} (R² = {best_score:.4f})")

    # 📊 Plot comparison graph
    fig, ax1 = plt.subplots(figsize=(8, 4))
    model_names = list(results.keys())
    r2_scores = [results[m]["R2"] for m in model_names]
    rmse_scores = [results[m]["RMSE"] for m in model_names]

    ax1.bar(model_names, r2_scores, label="R² Score", color='skyblue')
    ax1.set_ylabel("R² Score")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(model_names, rmse_scores, label="RMSE", color='tomato', marker='o')
    ax2.set_ylabel("RMSE")

    fig.suptitle("📊 Model Comparison (R² and RMSE)")
    fig.tight_layout()
    plt.savefig("model_comparison.png")  # Saves to project folder
    plt.close()

    return best_model, results
