import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges and drop 11 bad rows
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
print("Shape after cleaning:", df.shape)

# Drop customerID - not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# --- FEATURE ENGINEERING (the 3 resume features) ---

# 1. Tenure Buckets - group customers by how long they've stayed
def tenure_bucket(t):
    if t <= 12:
        return 'New'
    elif t <= 36:
        return 'Mid'
    else:
        return 'Loyal'

df['tenure_bucket'] = df['tenure'].apply(tenure_bucket)
print("Tenure buckets:\n", df['tenure_bucket'].value_counts())

# 2. Charge per month ratio - how much they pay relative to total
df['charge_per_month_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
print("\nCharge per month ratio sample:\n", df['charge_per_month_ratio'].head())

# 3. Service bundling score - how many services the customer has
service_cols = ['PhoneService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

df['service_bundle_score'] = df[service_cols].apply(
    lambda row: sum(1 for v in row if v == 'Yes'), axis=1
)
print("\nService bundle score distribution:\n", df['service_bundle_score'].value_counts().sort_index())

# --- ENCODING ---
# Convert Yes/No columns to 1/0
binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Convert gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# One-hot encode remaining categorical columns
cat_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
            'Contract','PaymentMethod','tenure_bucket']

df = pd.get_dummies(df, columns=cat_cols)
print("\nFinal shape after encoding:", df.shape)

# --- SCALING ---
X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and column names for use in the app later
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Save processed data
np.save('X_processed.npy', X_scaled)
np.save('y_processed.npy', y.values)

print("\n✅ Preprocessing complete! Files saved.")