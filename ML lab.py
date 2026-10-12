import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

print("Starting training process...")

# ---------------------------------------------------------
# Dataset 1: Credit Scoring Sample (Predicting Income)
# ---------------------------------------------------------
try:
    # Using the exact zip path provided
    df_credit = pd.read_csv(r"C:\Users\Haider Ali\Downloads\credit_scoring_sample.csv.zip")
    
    # Attempting to use standard credit scoring columns
    try:
        X_credit = df_credit[['age', 'DebtRatio', 'NumberOfOpenCreditLinesAndLoans']].fillna(0)
        y_credit = df_credit['MonthlyIncome'].fillna(df_credit['MonthlyIncome'].mean())
    except KeyError:
        # Fallback: Use the first 3 numeric columns to predict the 4th if names don't match
        numeric_cols = df_credit.select_dtypes(include='number').columns
        X_credit = df_credit[numeric_cols[0:3]].fillna(0)
        y_credit = df_credit[numeric_cols[3]].fillna(0)
        
    model_credit = LinearRegression()
    model_credit.fit(X_credit, y_credit)
    joblib.dump(model_credit, 'credit_model.pkl')
    print("Credit model saved successfully as 'credit_model.pkl'")
except FileNotFoundError:
    print("Error: Credit scoring dataset not found at the specified path.")

# ---------------------------------------------------------
# Dataset 2: Beauty (Predicting Wage)
# ---------------------------------------------------------
try:
    # Using the exact path provided
    df_beauty = pd.read_csv(r"C:\Users\Haider Ali\Downloads\beauty.csv")
    
    # Attempting to use standard beauty dataset columns
    try:
        X_beauty = df_beauty[['exper', 'educ', 'looks']]
        y_beauty = df_beauty['wage']
    except KeyError:
        # Fallback: Use the first 3 numeric columns to predict the 4th
        numeric_cols = df_beauty.select_dtypes(include='number').columns
        X_beauty = df_beauty[numeric_cols[0:3]].fillna(0)
        y_beauty = df_beauty[numeric_cols[3]].fillna(0)
        
    model_beauty = LinearRegression()
    model_beauty.fit(X_beauty, y_beauty)
    joblib.dump(model_beauty, 'beauty_model.pkl')
    print("Beauty model saved successfully as 'beauty_model.pkl'")
except FileNotFoundError:
    print("Error: Beauty dataset not found at the specified path.")

print("All training complete.")