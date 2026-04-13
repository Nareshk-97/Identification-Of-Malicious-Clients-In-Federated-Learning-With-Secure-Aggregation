import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# Load dataset
df = pd.read_csv('malicious_clients.csv')

# Split features and labels
X = df.drop('Label', axis=1)
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'model.pkl')

# Save metrics for dashboard
metrics = {
    'accuracy': accuracy,
    'total_samples': len(df),
    'label_counts': df['Label'].value_counts().to_dict(),
    'feature_importance': dict(zip(X.columns, model.feature_importances_))
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Model and metrics saved.")
