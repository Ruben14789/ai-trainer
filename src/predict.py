import argparse
import joblib
import pandas as pd

model = joblib.load("../models/student_model.pkl")

parser = argparse.ArgumentParser(description="Predict student performance index")

parser.add_argument("--hours", type=float, required=True)
parser.add_argument("--previous", type=float, required=True)
parser.add_argument("--activities", type=int, required=True)
parser.add_argument("--sleep", type=float, required=True)
parser.add_argument("--papers", type=float, required=True)

args = parser.parse_args()

input_data = pd.DataFrame([{
    "Hours_Studied": args.hours,
    "Previous_Scores": args.previous,
    "Extracurricular_Activities": args.activities,
    "Sleep_Hours": args.sleep,
    "Sample_Question_Papers_Practiced": args.papers
}])

prediction = model.predict(input_data)[0]

print(f"Predicted Performance Index: {prediction:.2f}")