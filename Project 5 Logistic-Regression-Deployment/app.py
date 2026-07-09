import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# 1. Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Purchase Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Insurance Purchase Prediction")
st.write("Predict if a person will buy life insurance based on their age using Logistic Regression.")

# -----------------------------------
# 2. Load Dataset
# -----------------------------------
# Make sure "insurance_data.csv" is in the same folder!
try:
    df = pd.read_csv("insurance_data.csv")
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # 3. Train Model
    # -----------------------------------
    X = df[['age']]
    y = df['bought_insurance']

    model = LogisticRegression()
    model.fit(X, y)

    # -----------------------------------
    # 4. User Input
    # -----------------------------------
    st.subheader("🔮 Make a Prediction")

    age_input = st.number_input(
        "Enter Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    # -----------------------------------
    # 5. Prediction Logic
    # -----------------------------------
    if st.button("Predict Status"):
        # Predict class (0 or 1)
        prediction = model.predict([[age_input]])
        # Predict probability percentages
        probability = model.predict_proba([[age_input]])[0][1] * 100

        if prediction[0] == 1:
            st.success(f"🎉 **Result:** Likely to buy insurance! (Probability: {probability:.2f}%)")
        else:
            st.error(f"❌ **Result:** Unlikely to buy insurance. (Probability: {probability:.2f}%)")

    # -----------------------------------
    # 6. Model Parameters
    # -----------------------------------
    st.subheader("📊 Model Details")
    st.write(f"**Coefficient (m):** {model.coef_[0][0]:.4f}")
    st.write(f"**Intercept (b):** {model.intercept_[0]:.4f}")

except FileNotFoundError:
    st.warning("⚠️ **File Not Found:** Please make sure 'insurance_data.csv' is inside the same folder as this script.")

# -----------------------------------
    # 7. Footer / Contact Info
    # -----------------------------------
st.markdown("---")
st.markdown("👨‍💻 Created by Ansh | [Connect with me on LinkedIn](https://www.linkedin.com/in/anshprogrammer/)")