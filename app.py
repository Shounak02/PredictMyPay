import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import load_and_prepare_data, train_and_select_best_model, forecast_salary

st.set_page_config(page_title="Career Path & Salary Growth Forecaster", layout="wide")
st.title("💼 Career Path + Salary Growth Forecaster")

(X_train, X_test, y_train, y_test), label_encoders = load_and_prepare_data("data/adult 3.csv")
best_model, _ = train_and_select_best_model(X_train, X_test, y_train, y_test)

st.sidebar.header("Enter Your Current Info")
user_input = {}
for feature in X_train.columns:
    if feature in label_encoders:
        options = label_encoders[feature].classes_
        selected = st.sidebar.selectbox(feature, options)
        user_input[feature] = label_encoders[feature].transform([selected])[0]
    else:
        user_input[feature] = st.sidebar.slider(feature, int(X_train[feature].min()), int(X_train[feature].max()))

user_data = list(user_input.values())

years = [0, 1, 2, 3]
forecasted_salary = forecast_salary(best_model, user_data)
st.subheader(f"📈 Predicted Salary Growth Over 3 Years")
st.write(f"Current Estimated Salary: ₹{int(forecasted_salary[0]):,}")

fig, ax = plt.subplots()
ax.plot(years, forecasted_salary, marker='o')
ax.set_title("Salary Projection")
ax.set_xlabel("Years from now")
ax.set_ylabel("Estimated Salary (INR)")
ax.grid(True)
st.pyplot(fig)

st.markdown("### 🎓 Suggested Upskilling Steps")
st.markdown("""
- Consider certifications in Cloud, Data Science, or Full-stack Web Dev.
- Improve your communication and leadership skills.
- Contribute to open-source or build side projects.
- Regularly update your resume and LinkedIn profile.
""")
