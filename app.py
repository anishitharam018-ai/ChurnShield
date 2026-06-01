import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡️",
    layout="wide"
)

# --- LOAD MODEL & ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    model = load_model('churnshield_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# --- HEADER ---
st.markdown("""
    <h1 style='text-align:center; color:#E63946;'>🛡️ ChurnShield</h1>
    <p style='text-align:center; color:gray; font-size:18px;'>
        Real-time Customer Churn Prediction powered by Deep Learning
    </p>
    <hr>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🔍 Predict Churn", "📊 EDA Insights"])

# ===================== TAB 1 - PREDICTION =====================
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 18.0, 8700.0, 1500.0)

    # --- FEATURE ENGINEERING (same as preprocess.py) ---
    def tenure_bucket(t):
        if t <= 12: return 'New'
        elif t <= 36: return 'Mid'
        else: return 'Loyal'

    service_cols_vals = [phone_service, online_security, online_backup,
                         device_protection, tech_support, streaming_tv, streaming_movies]
    service_bundle_score = sum(1 for v in service_cols_vals if v == "Yes")
    charge_per_month_ratio = monthly_charges / (total_charges + 1)
    t_bucket = tenure_bucket(tenure)

    # Build raw input dict
    input_dict = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'charge_per_month_ratio': charge_per_month_ratio,
        'service_bundle_score': service_bundle_score,
        'MultipleLines_No': 1 if multiple_lines == 'No' else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
        'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_No': 1 if online_security == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No': 1 if online_backup == 'No' else 0,
        'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_No': 1 if device_protection == 'No' else 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No': 1 if tech_support == 'No' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No': 1 if streaming_tv == 'No' else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No': 1 if streaming_movies == 'No' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
        'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'tenure_bucket_Loyal': 1 if t_bucket == 'Loyal' else 0,
        'tenure_bucket_Mid': 1 if t_bucket == 'Mid' else 0,
        'tenure_bucket_New': 1 if t_bucket == 'New' else 0,
    }

    # Align with training columns
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    # --- PREDICT ---
    if st.button("🔍 Predict Churn Risk", use_container_width=True):
        prob = model.predict(input_scaled)[0][0]
        percent = round(prob * 100, 1)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            if prob >= 0.5:
                st.error(f"⚠️ HIGH CHURN RISK: {percent}%")
                st.markdown("This customer is **likely to churn**. Consider retention actions.")
            else:
                st.success(f"✅ LOW CHURN RISK: {percent}%")
                st.markdown("This customer is **likely to stay**.")

        with col_b:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percent,
                title={'text': "Churn Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#E63946"},
                    'steps': [
                        {'range': [0, 40], 'color': "#2DC653"},
                        {'range': [40, 65], 'color': "#F4A261"},
                        {'range': [65, 100], 'color': "#E63946"},
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Risk factors
        st.markdown("### 🔎 Key Risk Factors for this customer")
        factors = []
        if contract == "Month-to-month": factors.append("📋 Month-to-month contract (high risk)")
        if tenure <= 12: factors.append("🆕 New customer — tenure under 12 months")
        if monthly_charges > 65: factors.append(f"💰 High monthly charges: ${monthly_charges}")
        if internet_service == "Fiber optic": factors.append("📡 Fiber optic users churn more")
        if online_security == "No": factors.append("🔓 No online security")
        if tech_support == "No": factors.append("🛠️ No tech support")
        if service_bundle_score <= 1: factors.append("📦 Low service bundle score")

        if factors:
            for f in factors: st.warning(f)
        else:
            st.info("✅ No major risk factors detected.")

# ===================== TAB 2 - EDA =====================
with tab2:
    st.subheader("📊 Dataset Insights")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churned", f"{df['Churn'].value_counts()['Yes']:,}")
    col3.metric("Churn Rate", f"{round(df['Churn'].value_counts(normalize=True)['Yes']*100,1)}%")
    col4.metric("Avg Monthly Charges", f"${round(df['MonthlyCharges'].mean(),2)}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(df, x='tenure', color='Churn',
                            title='Tenure Distribution by Churn',
                            color_discrete_map={'Yes':'#E63946','No':'#2DC653'})
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.box(df, x='Churn', y='MonthlyCharges',
                      title='Monthly Charges vs Churn',
                      color='Churn',
                      color_discrete_map={'Yes':'#E63946','No':'#2DC653'})
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        contract_churn = df.groupby(['Contract','Churn']).size().reset_index(name='count')
        fig3 = px.bar(contract_churn, x='Contract', y='count', color='Churn',
                      title='Contract Type vs Churn',
                      color_discrete_map={'Yes':'#E63946','No':'#2DC653'})
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.pie(df, names='Churn', title='Overall Churn Distribution',
                      color_discrete_map={'Yes':'#E63946','No':'#2DC653'})
        st.plotly_chart(fig4, use_container_width=True)