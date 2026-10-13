# 🎓 Student Marks Predictor

An AI/Machine Learning project that predicts a student's **final exam marks** from:
study hours, attendance, previous marks, sleep hours, and assignments completed.

## 📁 Files

| File | What it is |
|---|---|
| `data/student_data.csv` | Dataset of 500 students |
| `data/generate_dataset.py` | Script that created the dataset |
| `train_model.py` | Cleans data, draws charts, trains 3 models, saves the best one |
| `app.py` | Website where you enter details and see predicted marks (Streamlit) |
| `Student_Marks_Predictor.ipynb` | Same project as a notebook, for Google Colab / Jupyter |
| `charts/` | Charts for the report and PPT |
| `requirements.txt` | List of Python libraries needed |

## ▶️ How to run (on a laptop)

1. Install Python 3.10+ from python.org
2. Open a terminal in this folder and install libraries:
   ```
   pip install -r requirements.txt
   ```
3. Train the model (creates `model.pkl`):
   ```
   python train_model.py
   ```
4. Start the website:
   ```
   streamlit run app.py
   ```
   It opens in the browser at http://localhost:8501

## ▶️ How to run in Google Colab (no installation)

1. Go to colab.research.google.com → Upload → choose `Student_Marks_Predictor.ipynb`
2. Click the 📁 Files icon on the left and upload `data/student_data.csv`
3. Run each cell from top to bottom (Shift + Enter)

## 🧠 Models used

| Model | R² score | MAE (avg error in marks) |
|---|---|---|
| Linear Regression ✅ best | 0.903 | 3.34 |
| Random Forest | 0.854 | 4.08 |
| Decision Tree | 0.783 | 4.86 |

- **R² score**: 1.0 means perfect prediction. 0.90 means the model explains about 90% of the change in marks.
- **MAE**: on average, the prediction is off by about 3 marks.

## 📊 Key findings

- Study hours has the strongest effect on marks.
- Attendance, previous marks, and assignments also raise marks.
- Sleeping too little or too much lowers marks slightly.

## 🚀 Future scope

- Collect real data from college students using a Google Form
- Add more factors (internet access, family support, travel time)
- Deploy the app online for free with Streamlit Community Cloud

## ⚠️ Note

The dataset is **synthetic** (made by a Python script to look realistic).
For a stronger project, replace it with real data from Kaggle ("Student Performance")
or data collected from classmates, and run `train_model.py` again.
