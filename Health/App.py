import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

model = pickle.load(open('health_predict.pkl', 'rb'))

encoders = pickle.load(open('LabelEncoders.pkl', 'rb'))

st.title("Health Predictor")

age = st.slider("Age", 12, 100, 25)
diet = st.selectbox("Diet quality", encoders["diet"].classes_)
exercise = st.slider("Exercise day per week", 0, 7, 3)
sleep = st.slider("Sleep hours per day", 2, 12, 6)
stress = st.selectbox("Stress level", encoders["stress"].classes_)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
smoking = st.selectbox(
    "Smoking habit", encoders["smoking"].classes_, format_func=str.title
)
alcohol = st.selectbox(
    "Alcohol consumption", encoders["alcohol"].classes_, format_func=str.title
)
family_history = st.selectbox(
    "Family history of health issues", encoders["family_history"].classes_
)

if st.button("Diagnose Risk Factors"):
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "diet": diet,
                "exercise_days": exercise,
                "sleep_hours": sleep,
                "stress": stress,
                "bmi": bmi,
                "smoking": smoking,
                "alcohol": alcohol,
                "family_history": family_history,
            }
        ],
        columns=model.feature_names_in_,
    )

    for column in input_data.columns:
        if column in encoders:
            input_data[column] = encoders[column].transform(input_data[column])

    prediction = model.predict(input_data)
    risk_label = encoders["risk_level"].inverse_transform(prediction)[0]

    st.subheader(f"Predicted Risk Level: {risk_label}")

    # Bar Chart for lifestyle factors

    factors={
        "Diet":encoders["diet"].transform([diet])[0],
        "Exercise":exercise,
        "Sleep":sleep,
        "Stress":encoders["stress"].transform([stress])[0],
        "BMI":bmi
    }

    pie_fig = px.pie(
        names=list(factors.keys()),
        values=list(factors.values()),
        title="Lifestyle Factors Overview",
    )
    st.plotly_chart(pie_fig)
