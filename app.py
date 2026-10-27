import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from src.data_preprocessing import load_and_preprocess_data

st.set_page_config(page_title="SleepWise AI", layout="wide")

try:
    df = load_and_preprocess_data('data/Sleep_health_and_lifestyle_dataset.csv')
    with open('reports/model_metrics.json', 'r') as f:
        metrics = json.load(f)
except Exception:
    st.error("Model missing. Run `python src/train_models.py` first.")
    st.stop()

page = st.sidebar.radio("Navigation", ["EDA", "Model Metrics", "Predict Disorder", "Clusters"])

if page == "EDA":
    st.title("Data Analysis")
    st.plotly_chart(px.box(df, x='Sleep Disorder', y='Sleep Duration', color='Sleep Disorder'))

elif page == "Model Metrics":
    st.title("Performance & Overfitting Analysis")
    model_names = [m for m in metrics.keys() if m != 'best_model']
    fig = go.Figure(data=[
        go.Bar(name='Train F1', x=model_names, y=[metrics[m]['train_score'] for m in model_names]),
        go.Bar(name='CV F1', x=model_names, y=[metrics[m]['cv_macro_f1'] for m in model_names])
    ])
    st.plotly_chart(fig)
    st.success(f"Best Model: {metrics['best_model']}")

elif page == "Predict Disorder":
    st.title("Predict Sleep Disorder")
    with st.form("pred"):
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", df['Gender'].unique())
        occ = st.selectbox("Occupation", df['Occupation'].unique())
        sleep = st.number_input("Sleep (hrs)", 0.0, 24.0, 7.0)
        act = st.number_input("Activity (mins)", 0, 1440, 60)
        stress = st.slider("Stress (1-10)", 1, 10, 5)
        bmi = st.selectbox("BMI", df['BMI Category'].unique())
        hr = st.number_input("Heart Rate", 40, 150, 70)
        steps = st.number_input("Steps", 0, 50000, 5000)
        sys = st.number_input("Systolic BP", 80, 200, 120)
        dia = st.number_input("Diastolic BP", 40, 130, 80)
        if st.form_submit_button("Predict"):
            model = joblib.load(f"models/{metrics['best_model'].replace(' ', '_')}_model.pkl")
            input_df = pd.DataFrame([{'Gender': gender, 'Age': age, 'Occupation': occ, 'Sleep Duration': sleep, 'Physical Activity Level': act, 'Stress Level': stress, 'BMI Category': bmi, 'Heart Rate': hr, 'Daily Steps': steps, 'Systolic': sys, 'Diastolic': dia}])
            pred = model.predict(input_df)[0]
            # st.subheader(f"Prediction: {pred}")
            if pred == "None":
                st.success("### Result: No Sleep Disorder (Healthy)")
            elif pred == "Insomnia":
                st.warning("### Result: Risk of Insomnia")
            elif pred == "Sleep Apnea":
                st.error("### Result: Risk of Sleep Apnea")

elif page == "Clusters":
    st.title("Lifestyle Clusters")
    cdf = pd.read_csv('data/clustered_data.csv')
    st.plotly_chart(px.scatter(cdf, x='PCA1', y='PCA2', color=cdf['Cluster'].astype(str)))