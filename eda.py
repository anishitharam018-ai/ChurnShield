import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges hidden issue
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Missing after fix:", df['TotalCharges'].isnull().sum())

# Plot 1 - Churn distribution
plt.figure(figsize=(5,4))
df['Churn'].value_counts().plot(kind='bar', color=['steelblue','tomato'])
plt.title("Churn Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot_churn_dist.png")
plt.close()

# Plot 2 - Tenure vs Churn
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, palette=['steelblue','tomato'])
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.savefig("plot_tenure.png")
plt.close()

# Plot 3 - Monthly Charges vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=['steelblue','tomato'])
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig("plot_monthly.png")
plt.close()

# Plot 4 - Contract type vs Churn
plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Contract', hue='Churn', palette=['steelblue','tomato'])
plt.title("Contract Type vs Churn")
plt.tight_layout()
plt.savefig("plot_contract.png")
plt.close()

print("All plots saved!")