import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv('Salary_Data.csv')
X = data[['YearsExperience']]
y = data['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Save the metrics to a JSON file for the Flask app to use
metrics = {
    "mae": round(mae, 2),
    "mse": round(mse, 2),
    "rmse": round(rmse, 2),
    "r2": round(r2, 4)
}

import json
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save the model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
