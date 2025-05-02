import os
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Insurance Charges Prediction", page_icon="💰", layout="centered")

# Apply dark theme styling
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .css-1d391kg, .css-1kyxreq, .stButton>button {
            background-color: #262730; color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# File paths
model_path = r"C:\Users\ADMIN\Desktop\Insurance_charges\data_pickle_2.pkl"
data_path = r"C:\Users\ADMIN\Desktop\Insurance_charges\insurance.csv"

# Check if files exist
if not os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}")
elif not os.path.exists(data_path):
    st.error(f"Error: Data file not found at {data_path}")
else:
    # Load trained pipeline model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load dataset
    df = pd.read_csv(data_path)

    # App UI
    st.title("💰 Insurance Charges Prediction App")
    st.markdown("Predict the insurance charges based on several factors.")
    st.sidebar.header("Enter Patient Details")

    # Input fields
    age = st.sidebar.number_input("Age", min_value=18, max_value=100)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10)
    smoker = st.sidebar.selectbox("Smoker", ("Yes", "No"))
    region = st.sidebar.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    # Predict button
    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex.lower(),
            'bmi': bmi,
            'children': children,
            'smoker': smoker.lower(),
            'region': region.lower()
        }])

        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            st.subheader("Prediction Result")
            st.success(f"💰 Predicted Insurance Charge: ${prediction:,.2f}")

            # Profile bar chart
            profile_data = pd.DataFrame({
                'Feature': ['Age', 'Sex (1=Male)', 'BMI', 'Children', 'Smoker (1=Yes)', 'Region'],
                'Value': [age, 1 if sex.lower() == "male" else 0, bmi, children, 1 if smoker.lower() == "yes" else 0, region]
            })

            fig_profile, ax_profile = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Feature', y='Value', data=profile_data, ax=ax_profile)
            ax_profile.set_title("Patient Profile", color='white')
            ax_profile.set_ylabel("Value", color='white')
            ax_profile.set_xlabel("Feature", color='white')
            ax_profile.tick_params(colors='white')
            st.pyplot(fig_profile)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Dataset preview
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.markdown(f"The dataset contains {df.shape[0]} records and {df.shape[1]} features.")
