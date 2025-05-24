import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open("models/svc.pkl", "rb"))

# Dictionaries (you can place these in a separate file if they get too long)
symptoms_dict = { ... }  # Same dictionary you provided
diseases_list = { ... }  # Same dictionary you provided

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Streamlit UI
st.title("🩺 Disease Prediction System")
st.write("Enter your symptoms below, separated by commas (e.g. `headache, nausea`)")

# User input
symptoms_input = st.text_input("Symptoms:")

if st.button("Predict"):
    if symptoms_input.strip().lower() in ["", "symptoms"]:
        st.error("❌ Please provide valid symptoms.")
    else:
        user_symptoms = [s.strip("[]' ").lower() for s in symptoms_input.split(',')]

        try:
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precaution_list, medication_list, diet_list, workout_list = helper(predicted_disease)

            st.subheader(f"🦠 Predicted Disease: {predicted_disease}")
            st.write(f"**📘 Description:** {dis_des}")

            st.subheader("💊 Medications")
            st.write(", ".join(medication_list))

            st.subheader("🥗 Recommended Diet")
            st.write(", ".join(diet_list))

            st.subheader("🏃 Workout Suggestions")
            st.write(", ".join(workout_list))

            st.subheader("🛡️ Precautions")
            for idx, precaution in enumerate(precaution_list[0], start=1):
                st.write(f"{idx}. {precaution}")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
