# 🩺 Diabetes Prediction System

An end-to-end Machine Learning web application that predicts whether a patient is **Diabetic** or **Non-Diabetic** based on clinical diagnostic metrics using a **K-Nearest Neighbors (KNN)** model and a modern **Gradio** dark-mode user interface.

---

## 📁 Repository Directory Structure

```text
Diabetes-Prediction-System/
│── diabetes.csv              # Pima Indians Diabetes Dataset
│── app.py                    # Gradio interactive web application (Dark Mode UI)
│── diabetes_model.pkl        # Exported trained KNN pipeline model
│── Diabetes_Prediction.ipynb # Step-by-step Jupyter Notebook & Model Training
│── requirements.txt          # Python dependencies list
│── .gitignore                # Git ignore rules
└── README.md                 # Complete project documentation
```

---

## 📊 Dataset Parameters

The system evaluates eight (8) clinical features:
1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
4. **Skin Thickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg / (height in m)²)
7. **Diabetes Pedigree Function**: Scoring function based on family history
8. **Age**: Age in years

---

## 🤖 Machine Learning Workflow & Model Architecture

- **Algorithm**: K-Nearest Neighbors (KNN Classifier, $k=15$, distance-weighted)
- **Data Preprocessing**: Biological zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI are converted to `NaN` and imputed using column medians via `SimpleImputer`.
- **Feature Scaling**: `StandardScaler` standardizes feature metrics.
- **Serialization**: Model pipeline saved via `joblib` into `diabetes_model.pkl`.

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Gradio Web Interface
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:7860`

---

## 📓 Running the Jupyter Notebook Line-by-Line

Launch Jupyter Notebook:
```bash
jupyter notebook Diabetes_Prediction.ipynb
```
Execute cells sequentially to inspect data loading, exploratory analysis, model training metrics, artifact serialization, and inline web interface preview.

---

## 🌐 Deploying to Hugging Face Spaces / Render

### Hugging Face Spaces Deployment:
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) using **Gradio** SDK.
2. Push `app.py`, `diabetes_model.pkl`, and `requirements.txt`.
3. Your app will automatically build and launch online!
