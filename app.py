import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.title("Salary Prediction App")


st.write("Model R2 Score:",0.96)

#load model & encoders
model = pickle.load(open("models/model.pkl", "rb"))
le_edu = pickle.load(open("models/le_edu.pkl", "rb"))
le_role = pickle.load(open("models/le_role.pkl", "rb"))
le_loc = pickle.load(open("models/le_loc.pkl", "rb"))
le_size = pickle.load(open("models/le_size.pkl", "rb"))



# User inputs
education = st.selectbox("Education",le_edu.classes_)
job_role = st.selectbox("cleaJob Role",le_role.classes_)
location = st.selectbox("Location",le_loc.classes_)
company_size = st.selectbox("Company Size",le_size.classes_)
experience = st.slider("Years 0f Experience",0,20,1)
skills = st.slider("skills Score (1-5)",1,5,3)

#--------Prediction Button----------#

if st.button("Predict Salary"):
    #Encode Inputs
    edu_enc = le_edu.transform([education])[0]
    role_enc = le_role.transform([job_role])[0]
    loc_enc = le_loc.transform([location])[0]
    size_enc = le_size.transform([company_size])[0]

    features = np.array([[experience,edu_enc,role_enc,loc_enc,skills,size_enc]])
    prediction = model.predict(features)


    st.success(f"Predicted salary : {prediction[0]:.2f} LPA")

#------Graph Button (Separate)--------
if st.button ("Shows Experience vs Salary Trend"):
    edu_enc = le_edu.transform([education])[0]
    role_enc = le_role.transform([job_role])[0]
    loc_enc = le_loc.transform([location])[0]
    size_enc = le_size.transform([company_size])[0]

    exp_range = np.linspace(0,20,20)
    dummy_data = np.array([[e,edu_enc,role_enc,loc_enc,skills,size_enc] for e in exp_range])
    preds = model.predict(dummy_data)


    plt.plot(exp_range,preds)
    plt.xlabel("Experience")
    plt.ylabel("Predicted Salary")
    st.pyplot(plt)
