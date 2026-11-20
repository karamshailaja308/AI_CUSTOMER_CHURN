import streamlit as st

from agent.retention_agent import predict_customer_churn

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("🤖 AI Customer Churn Prediction")
st.write(
    "Predict customer churn and generate personalized "
    "retention recommendations."
)

st.divider()

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=2
    )
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )

with col2:
    protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly = st.number_input(
        "Monthly Charges", min_value=0.0, value=85.50
    )
    total = st.number_input(
        "Total Charges", min_value=0.0, value=171.00
    )

st.divider()

if st.button("Predict Customer Churn", type="primary"):

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": billing,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        result = predict_customer_churn(customer)

        st.subheader("Customer Churn Analysis")

        st.metric(
            "Churn Probability",
            f"{result['churn_probability']}%"
        )

        st.write("**Risk Level:**", result["risk"])
        st.write("**Recommended Action:**", result["action"])

        st.subheader("Personalized Retention Message")
        st.info(result["message"])

    except Exception as error:
        st.error(f"Prediction error: {error}")

st.caption("AI Customer Churn Prediction and Retention System")