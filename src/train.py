import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("../data/Student_Performance.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert Yes/No to 1/0
if "Extracurricular_Activities" in df.columns:
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({
        "Yes": 1,
        "No": 0
    })

# Define features X and target y
X = df.drop("Performance_Index", axis=1)
y = df["Performance_Index"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {error:.2f}")

# Save model
joblib.dump(model, "../models/student_model.pkl")

print("Model saved to models/student_model.pkl")