import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_icon='❤️',page_title='Heart Disease Prediction')

# Load your data
data = pd.read_csv('./data.csv')

# Split into features and target
x = data.drop(columns='target', axis=1)
y = data['target']

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train your logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Function to get user inputs
def get_user_inputs():
    age = st.number_input("Enter the age:", min_value=0, max_value=150, step=1)
    sex = st.selectbox("Select sex:", options=["Female", "Male"])
    cp = st.number_input("Enter chest pain type (1-4):", min_value=1, max_value=4, step=1)
    trestbps = st.number_input("Enter resting blood pressure (mm Hg):", min_value=0.0, max_value=300.0, step=1.0)
    chol = st.number_input("Enter serum cholesterol (mg/dl):", min_value=0.0, max_value=600.0, step=1.0)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl:", options=["False", "True"])
    restecg = st.number_input("Enter resting electrocardiographic results (0-2):", min_value=0, max_value=2, step=1)
    thalach = st.number_input("Enter maximum heart rate achieved:", min_value=0, max_value=300, step=1)
    exang = st.selectbox("Exercise induced angina:", options=["No", "Yes"])
    oldpeak = st.number_input("Enter ST depression induced by exercise relative to rest:", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.number_input("Enter slope of the peak exercise ST segment (1-3):", min_value=1, max_value=3, step=1)
    ca = st.number_input("Enter number of major vessels (0-3) colored by fluoroscopy:", min_value=0, max_value=3, step=1)
    thal = st.number_input("Enter thalassemia type (0-3):", min_value=0, max_value=3, step=1)

    # Mapping inputs to numeric values
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "True" else 0
    exang = 1 if exang == "Yes" else 0

    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Main function to run the app
def main():
    st.title("Heart Disease Prediction App")
    st.write("Enter patient information to predict heart disease.")


    # Get user inputs
    user_inputs = get_user_inputs()

    # Create a button to predict
    if st.button("Predict"):
        # Prepare input data as a numpy array
        input_data = np.array(user_inputs).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)

        # Output the prediction
        if prediction[0] == 0:
            st.write("Prediction: The person has no heart disease")
        else:
            st.write("Prediction: The person has heart disease")

if __name__ == "__main__":
    main()

st.write("Made by **Anchit Das**")



