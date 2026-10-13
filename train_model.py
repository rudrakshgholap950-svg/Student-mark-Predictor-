"""
Student Marks Predictor - Training Script
-----------------------------------------
What this file does (in simple steps):
  1. Load the data
  2. Clean the data (fill missing values)
  3. Draw charts to understand the data
  4. Split data: 80% for training, 20% for testing
  5. Train 3 AI models and compare them
  6. Save the best model to "model.pkl" (the app uses this file)

Run:  python train_model.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save charts as images, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["study_hours", "attendance", "previous_marks", "sleep_hours", "assignments_done"]
TARGET = "final_marks"

sns.set_theme(style="whitegrid")
BLUE, ORANGE = "#4C72B0", "#DD8452"

# ---------------- 1. LOAD ----------------
df = pd.read_csv("data/student_data.csv")
print("Rows and columns:", df.shape)
print("\nMissing values in each column:\n", df.isnull().sum())

# ---------------- 2. CLEAN ----------------
# Fill empty cells with the middle value (median) of that column
for col in FEATURES:
    df[col] = df[col].fillna(df[col].median())
df = df.drop_duplicates()
print("\nAfter cleaning, missing values:", int(df.isnull().sum().sum()))

# ---------------- 3. CHARTS ----------------
# Chart 1: how are final marks spread out?
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df[TARGET], bins=25, color=BLUE, edgecolor="white")
ax.axvline(df[TARGET].mean(), color=ORANGE, linestyle="--", linewidth=2,
           label=f"Average: {df[TARGET].mean():.1f}")
ax.set_title("Most students score between 55 and 75 marks")
ax.set_xlabel("Final marks (out of 100)")
ax.set_ylabel("Number of students")
ax.legend()
plt.tight_layout(); plt.savefig("charts/1_marks_distribution.png", dpi=150); plt.close()

# Chart 2: more study hours -> more marks?
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["study_hours"], df[TARGET], color=BLUE, alpha=0.6)
m, b = np.polyfit(df["study_hours"], df[TARGET], 1)
xs = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 50)
ax.plot(xs, m * xs + b, color=ORANGE, linewidth=2.5, label="Trend line")
ax.set_title("More study hours → higher final marks")
ax.set_xlabel("Study hours per day")
ax.set_ylabel("Final marks")
ax.legend()
plt.tight_layout(); plt.savefig("charts/2_study_hours_vs_marks.png", dpi=150); plt.close()

# Chart 3: attendance vs marks
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["attendance"], df[TARGET], color=BLUE, alpha=0.6)
m, b = np.polyfit(df["attendance"], df[TARGET], 1)
xs = np.linspace(df["attendance"].min(), df["attendance"].max(), 50)
ax.plot(xs, m * xs + b, color=ORANGE, linewidth=2.5, label="Trend line")
ax.set_title("Better attendance → higher final marks")
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Final marks")
ax.legend()
plt.tight_layout(); plt.savefig("charts/3_attendance_vs_marks.png", dpi=150); plt.close()

# Chart 4: correlation heatmap (which columns move together?)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            linewidths=0.5, ax=ax)
ax.set_title("How strongly each factor relates to marks")
plt.tight_layout(); plt.savefig("charts/4_correlation_heatmap.png", dpi=150); plt.close()

# ---------------- 4. SPLIT ----------------
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining rows: {len(X_train)}, Testing rows: {len(X_test)}")

# ---------------- 5. TRAIN & COMPARE ----------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "R2 score": round(r2_score(y_test, pred), 3),       # closer to 1 = better
        "MAE": round(mean_absolute_error(y_test, pred), 2),  # average error in marks
    }

results_df = pd.DataFrame(results).T.sort_values("R2 score", ascending=False)
print("\nModel comparison:\n", results_df)
results_df.to_csv("charts/model_results.csv")

# Chart 5: model comparison
fig, ax = plt.subplots(figsize=(9, 5))
order = results_df.sort_values("R2 score")
bars = ax.barh(order.index, order["R2 score"], color=BLUE)
for bar in bars:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}", va="center")
ax.set_xlim(0, 1.05)
ax.set_title(f"{results_df.index[0]} predicts marks most accurately")
ax.set_xlabel("R² score (1.0 = perfect)")
plt.tight_layout(); plt.savefig("charts/5_model_comparison.png", dpi=150); plt.close()

# ---------------- 6. SAVE BEST ----------------
best_name = results_df.index[0]
best_model = models[best_name]
best_pred = best_model.predict(X_test)

# Chart 6: actual vs predicted
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, best_pred, color=BLUE, alpha=0.7)
lims = [min(y_test.min(), best_pred.min()) - 2, max(y_test.max(), best_pred.max()) + 2]
ax.plot(lims, lims, color=ORANGE, linestyle="--", linewidth=2, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"{best_name}: predicted vs actual marks")
ax.set_xlabel("Actual marks"); ax.set_ylabel("Predicted marks")
ax.legend()
plt.tight_layout(); plt.savefig("charts/6_actual_vs_predicted.png", dpi=150); plt.close()

joblib.dump(best_model, "model.pkl")
with open("model_info.json", "w") as f:
    json.dump({"best_model": best_name, "features": FEATURES,
               "results": results}, f, indent=2)

print(f"\nBest model: {best_name}  → saved as model.pkl")
print("Charts saved in the 'charts' folder.")

# Quick example prediction
sample = pd.DataFrame([[6, 90, 70, 7.5, 9]], columns=FEATURES)
print("\nExample: 6 study hrs, 90% attendance, 70 previous marks, 7.5 sleep hrs, 9 assignments")
print("Predicted final marks:", round(float(best_model.predict(sample)[0]), 1))
