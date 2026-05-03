import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv("../data/Student_Performance.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({
    "Yes": 1,
    "No": 0
})

# Load model
model = joblib.load("../models/student_model.pkl")

X = df.drop("Performance_Index", axis=1)
y = df["Performance_Index"]

predictions = model.predict(X)

# Actual vs Predicted
plt.figure()
plt.scatter(y, predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Performance")

plt.savefig("../images/actual_vs_predicted.png")  # save first
plt.show()
plt.close()

# Hours studied vs performance
plt.figure()
plt.scatter(df["Hours_Studied"], y)
plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.title("Study Time vs Performance")

plt.savefig("../images/study_vs_performance.png")  # save second graph
plt.show()
plt.close()