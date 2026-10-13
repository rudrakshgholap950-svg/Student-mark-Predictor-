"""
Step 1: Make the dataset.
This creates a table of 500 students with study habits and their final marks.
Run:  python data/generate_dataset.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)          # same random numbers every time
n = 500                     # number of students

study_hours = np.round(np.random.uniform(0.5, 9, n), 1)          # hours per day
attendance = np.round(np.random.uniform(45, 100, n), 1)          # percent
previous_marks = np.round(np.random.normal(65, 12, n).clip(30, 100), 1)
sleep_hours = np.round(np.random.normal(7, 1.1, n).clip(4, 10), 1)
assignments_done = np.random.randint(0, 11, n)                   # out of 10

# Final marks depend on the habits above, plus some random "luck"
final_marks = (
    4.0 * study_hours
    + 0.25 * attendance
    + 0.35 * previous_marks
    + 1.2 * assignments_done
    - 1.5 * np.abs(sleep_hours - 7.5)      # too little or too much sleep hurts
    + np.random.normal(0, 4, n)
)
final_marks = np.round(final_marks.clip(0, 100), 1)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks,
    "sleep_hours": sleep_hours,
    "assignments_done": assignments_done,
    "final_marks": final_marks,
})

# Add a few missing values on purpose so students can practise cleaning data
for col in ["study_hours", "attendance", "sleep_hours"]:
    df.loc[df.sample(8, random_state=len(col)).index, col] = np.nan

df.to_csv("data/student_data.csv", index=False)
print("Saved data/student_data.csv with", len(df), "rows")
print(df.head())
