"""
Student Marks Predictor - Web App
Run:  streamlit run app.py
(First run "python train_model.py" so that model.pkl exists.)
"""
import json
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Marks Predictor")
st.write("Enter a student's study habits and the AI will predict their final exam marks.")

if not os.path.exists("model.pkl"):
    st.error("model.pkl not found. Please run `python train_model.py` first.")
    st.stop()

model = joblib.load("model.pkl")
info = json.load(open("model_info.json")) if os.path.exists("model_info.json") else {}

# ----- Inputs -----
col1, col2 = st.columns(2)
with col1:
    study_hours = st.slider("📚 Study hours per day", 0.0, 10.0, 4.0, 0.5)
    attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)
    previous_marks = st.slider("📝 Previous exam marks", 0, 100, 60)
with col2:
    sleep_hours = st.slider("😴 Sleep hours per night", 3.0, 11.0, 7.0, 0.5)
    assignments_done = st.slider("✅ Assignments completed (out of 10)", 0, 10, 6)

# ----- Predict -----
if st.button("Predict my marks", type="primary", use_container_width=True):
    row = pd.DataFrame([[study_hours, attendance, previous_marks, sleep_hours, assignments_done]],
                       columns=["study_hours", "attendance", "previous_marks",
                                "sleep_hours", "assignments_done"])
    marks = float(model.predict(row)[0])
    marks = max(0.0, min(100.0, marks))

    st.metric("Predicted final marks", f"{marks:.1f} / 100")
    st.progress(marks / 100)

    if marks >= 75:
        st.success("Excellent! Keep up the good habits. 🌟")
    elif marks >= 50:
        st.info("Good, but there is room to improve. 👍")
    else:
        st.warning("At risk. Try studying more and attending more classes. 💪")

    # Simple tips
    tips = []
    if study_hours < 4:
        tips.append("Try to study at least 4 hours a day.")
    if attendance < 75:
        tips.append("Attend more classes — aim for 75% or above.")
    if sleep_hours < 6.5 or sleep_hours > 9:
        tips.append("Sleep 7–8 hours for better focus.")
    if assignments_done < 7:
        tips.append("Complete more assignments.")
    if tips:
        st.subheader("Tips to improve")
        for t in tips:
            st.write("• " + t)

# ----- Model details -----
with st.expander("About the AI model"):
    if info:
        st.write(f"**Model used:** {info['best_model']}")
        st.dataframe(pd.DataFrame(info["results"]).T)
    st.write("R² score: closer to 1 means more accurate. "
             "MAE: the average number of marks the prediction is off by.")
