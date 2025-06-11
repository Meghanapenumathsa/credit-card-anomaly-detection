import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("creditcard.csv")

# Check class imbalance
print("Class distribution:\n", df['Class'].value_counts())

# Separate features and labels
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# Scale all features, not just Amount
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different contamination rates and show results
print("\n--- Tuning Contamination Rate ---\n")
best_score = 0
best_preds = None
best_rate = 0

for rate in [0.0017, 0.002, 0.003, 0.004, 0.005]:
    print(f"Trying contamination: {rate}")
    model = IsolationForest(n_estimators=200, contamination=rate, max_samples=0.5, random_state=42)
    preds = model.fit_predict(X_scaled)
    mapped_preds = pd.Series(preds).map({1: 0, -1: 1})  # 1 = normal, -1 = fraud

    # Basic evaluation
    cm = confusion_matrix(y, mapped_preds)
    recall = cm[1,1] / (cm[1,1] + cm[1,0])  # TP / (TP + FN)
    print("Confusion Matrix:\n", cm)
    print(f"Recall: {recall:.2f}")
    print("-" * 40)

    # Save best predictions by recall
    if recall > best_score:
        best_score = recall
        best_preds = mapped_preds
        best_rate = rate

# Final prediction from best model
df['anomaly'] = best_preds

# Show summary
print(f"\n✅ Best contamination rate: {best_rate}")
print("Predicted anomaly distribution:\n", df['anomaly'].value_counts())
print("\nConfusion Matrix:")
print(confusion_matrix(y, df['anomaly']))
print("\nClassification Report:")
print(classification_report(y, df['anomaly']))

# Visualization
plt.figure(figsize=(8, 5))
sns.countplot(x='anomaly', data=df)
plt.title("Detected Anomalies (Frauds)")
plt.xlabel("0 = Normal, 1 = Fraud Detected")
plt.ylabel("Transaction Count")
plt.ylim(0, 6000) 
plt.show()
