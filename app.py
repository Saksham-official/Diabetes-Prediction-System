import warnings
warnings.filterwarnings('ignore')

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# Model path
MODEL_PATH = 'diabetes_model.pkl'

def load_model():
    """Load model artifact trained in Diabetes_Prediction.ipynb"""
    if not os.path.exists(MODEL_PATH):
        return None, 0.7338, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    try:
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            pipeline = artifact.get('pipeline')
            accuracy = artifact.get('accuracy', 0.7338)
            feature_names = artifact.get('feature_names', ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            return pipeline, accuracy, feature_names
        else:
            # Fallback if raw pipeline was saved
            return artifact, 0.7338, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    except Exception as e:
        print(f"Error loading model artifact: {e}")
        return None, 0.7338, []

model_pipeline, model_acc, feature_names = load_model()

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    """
    Predicts diabetes status matching preprocessing from Diabetes_Prediction.ipynb:
    - Zero values in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] converted to NaN
    - Pipeline imputes NaNs using Median Imputer -> StandardScaler -> KNN(k=15, weights='distance')
    """
    if model_pipeline is None:
        return (
            "Model Error",
            "<div style='padding: 15px; border-radius: 8px; background-color: #2D1517; border: 1px solid #7F1D1D; color: #FCA5A5;'>Model artifact (diabetes_model.pkl) not found or failed to load.</div>",
            None
        )

    # Preprocessing matching notebook cell 8: replace 0 with NaN for relevant clinical features
    input_data = pd.DataFrame([{
        'Pregnancies': float(pregnancies),
        'Glucose': float(glucose) if glucose > 0 else np.nan,
        'BloodPressure': float(blood_pressure) if blood_pressure > 0 else np.nan,
        'SkinThickness': float(skin_thickness) if skin_thickness > 0 else np.nan,
        'Insulin': float(insulin) if insulin > 0 else np.nan,
        'BMI': float(bmi) if bmi > 0 else np.nan,
        'DiabetesPedigreeFunction': float(dpf),
        'Age': float(age)
    }])

    # Ensure DataFrame column order matches notebook training features
    if feature_names:
        input_data = input_data[feature_names]

    # Model inference
    prediction = int(model_pipeline.predict(input_data)[0])
    
    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(input_data)[0]
        non_diabetic_prob = float(probabilities[0])
        diabetic_prob = float(probabilities[1])
    else:
        non_diabetic_prob = 1.0 if prediction == 0 else 0.0
        diabetic_prob = 1.0 if prediction == 1 else 0.0

    prob_dict = {
        "Non-Diabetic": round(non_diabetic_prob, 4),
        "Diabetic": round(diabetic_prob, 4)
    }

    if prediction == 1:
        result_title = "Diabetic (High Risk)"
        result_msg = f"""
        <div style="background: #181012; border: 1px solid #7F1D1D; border-left: 4px solid #EF4444; padding: 18px; border-radius: 10px; margin-top: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #FCA5A5;">Prediction: DIABETIC</span>
                <span style="background: #7F1D1D; color: #FECACA; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">Confidence: {diabetic_prob * 100:.1f}%</span>
            </div>
            <p style="color: #D1D5DB; margin: 6px 0; font-size: 0.95rem;">KNN model indicates a high probability of Diabetes based on patient clinical indicators.</p>
            <p style="color: #9CA3AF; margin-top: 8px; font-size: 0.85rem;"><em>Recommendation: Consult a medical professional for formal laboratory evaluation.</em></p>
        </div>
        """
    else:
        result_title = "Non-Diabetic (Normal)"
        result_msg = f"""
        <div style="background: #0D1914; border: 1px solid #064E3B; border-left: 4px solid #10B981; padding: 18px; border-radius: 10px; margin-top: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #6EE7B7;">Prediction: NON-DIABETIC</span>
                <span style="background: #064E3B; color: #D1FAE5; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">Confidence: {non_diabetic_prob * 100:.1f}%</span>
            </div>
            <p style="color: #D1D5DB; margin: 6px 0; font-size: 0.95rem;">KNN model indicates a low probability of Diabetes based on current parameters.</p>
            <p style="color: #9CA3AF; margin-top: 8px; font-size: 0.85rem;"><em>Recommendation: Maintain a balanced diet, exercise, and regular health checkups.</em></p>
        </div>
        """

    return result_title, result_msg, prob_dict

# Dark Theme CSS
custom_css = """
body, .gradio-container {
    background-color: #0B0F17 !important;
    color: #F3F4F6 !important;
}
.header-box {
    text-align: center;
    padding: 20px 0 10px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #1F2937;
}
.header-box h1 {
    color: #F9FAFB !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 6px !important;
}
.header-box p {
    color: #9CA3AF !important;
    font-size: 0.95rem !important;
}
.panel-box {
    background-color: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
button.primary {
    background-color: #2563EB !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: background-color 0.2s ease !important;
}
button.primary:hover {
    background-color: #1D4ED8 !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui"]
).set(
    body_background_fill="#0B0F17",
    body_text_color="#F3F4F6",
    block_background_fill="#111827",
    block_border_color="#1F2937",
    block_title_text_color="#F9FAFB",
    input_background_fill="#1F2937",
    slider_color="#3B82F6"
)

with gr.Blocks(title="Diabetes Prediction System") as demo:
    
    # Header
    gr.HTML(
        f"""
        <div class="header-box">
            <h1>Diabetes Prediction System</h1>
            <p>K-Nearest Neighbors (KNN) Model | Accuracy: {model_acc * 100:.1f}%</p>
        </div>
        """
    )
    
    with gr.Row():
        # Inputs Column
        with gr.Column(scale=1, elem_classes=["panel-box"]):
            gr.Markdown("#### Patient Metrics")
            
            with gr.Row():
                pregnancies = gr.Slider(minimum=0, maximum=20, step=1, value=1, label="Pregnancies")
                age = gr.Slider(minimum=1, maximum=100, step=1, value=33, label="Age (Years)")

            with gr.Row():
                glucose = gr.Slider(minimum=0, maximum=250, step=1, value=120, label="Glucose (mg/dL)")
                blood_pressure = gr.Slider(minimum=0, maximum=140, step=1, value=70, label="Blood Pressure (mm Hg)")

            with gr.Row():
                bmi = gr.Slider(minimum=0.0, maximum=70.0, step=0.1, value=25.4, label="BMI (kg/m²)")
                insulin = gr.Slider(minimum=0, maximum=850, step=1, value=79, label="Insulin (mu U/ml)")

            with gr.Row():
                skin_thickness = gr.Slider(minimum=0, maximum=100, step=1, value=20, label="Skin Thickness (mm)")
                dpf = gr.Slider(minimum=0.0, maximum=3.0, step=0.01, value=0.372, label="Diabetes Pedigree Function")

            predict_btn = gr.Button("Predict Risk", variant="primary", size="lg")

        # Outputs Column
        with gr.Column(scale=1, elem_classes=["panel-box"]):
            gr.Markdown("#### Prediction Output")
            
            output_status = gr.Textbox(label="Status Summary", interactive=False)
            output_details = gr.HTML(value="<div style='padding: 16px; background: #1F2937; border-radius: 8px; color: #9CA3AF;'>Enter patient values and click <b>Predict Risk</b>.</div>")
            output_probs = gr.Label(label="Confidence Breakdown", num_top_classes=2)

    predict_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
        outputs=[output_status, output_details, output_probs]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=theme,
        css=custom_css
    )
