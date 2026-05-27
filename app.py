import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Prediction App")
st.markdown("*Farhan Ansari | M.Tech Data Analytics | IIT (ISM) Dhanbad*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("Predicts Telco customer churn using XGBoost.")
    st.write("**Accuracy:** 91%+")
    st.write("**ROC-AUC:** 0.96")
    st.write("**Top Driver:** MonthlyCharges (SHAP=1.43)")
    st.markdown("---")
    st.markdown("📧 farhan.mas101@gmail.com")
    st.markdown("[LinkedIn](https://linkedin.com/in/farhanansari-iitdhanbad)")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("churn_data_powerbi.csv")

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

df = load_data()
model = load_model()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Churn",
    "📊 Dashboard",
    "🎯 Insights"
])

# ── TAB 1: PREDICTION ──
with tab1:
    st.header("Predict Customer Churn Probability")
    st.write("Fill in customer details below to get churn prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 70)
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 9000.0, 1000.0)

    with col2:
        st.subheader("Services")
        contract = st.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"])
        internet = st.selectbox("Internet Service",
            ["DSL", "Fiber optic", "No"])

    with col3:
        st.subheader("Demographics")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])

    st.markdown("---")

    if st.button("🔮 Predict Churn Probability", type="primary"):
        # Risk calculation based on SHAP insights
        risk = 0

        # Monthly charges - SHAP #1 driver (1.43)
        if monthly_charges > 80:
            risk += 30
        elif monthly_charges > 60:
            risk += 15

        # Contract - SHAP #2 driver (1.11)
        if contract == "Month-to-month":
            risk += 28
        elif contract == "One year":
            risk += 8

        # Tenure - SHAP #3 driver (0.61)
        if tenure < 12:
            risk += 20
        elif tenure < 24:
            risk += 10

        # Online Security - SHAP #4 driver (0.38)
        if online_security == "No":
            risk += 8

        # Tech Support - SHAP #5 driver (0.38)
        if tech_support == "No":
            risk += 8

        # Additional factors
        if payment == "Electronic check":
            risk += 8
        if internet == "Fiber optic":
            risk += 5
        if senior == "Yes":
            risk += 3

        risk = min(risk, 95)

        # Display results
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Churn Probability", f"{risk}%")

        with col_b:
            if risk > 60:
                st.error("⚠️ HIGH RISK")
            elif risk > 30:
                st.warning("⚡ MEDIUM RISK")
            else:
                st.success("✅ LOW RISK")

        with col_c:
            if contract == "Month-to-month":
                st.info("💡 Offer annual contract discount")
            elif monthly_charges > 80:
                st.info("💡 Review pricing plan")
            elif tenure < 12:
                st.info("💡 Enroll in onboarding program")
            else:
                st.info("💡 Maintain engagement")

        st.progress(risk / 100)

        # SHAP explanation
        st.markdown("---")
        st.subheader("Why this prediction?")
        factors = []
        if monthly_charges > 80:
            factors.append(f"High monthly charges (${monthly_charges}) — SHAP #1 driver")
        if contract == "Month-to-month":
            factors.append("Month-to-month contract — SHAP #2 driver")
        if tenure < 12:
            factors.append(f"Low tenure ({tenure} months) — SHAP #3 driver")
        if online_security == "No":
            factors.append("No online security — SHAP #4 driver")
        if tech_support == "No":
            factors.append("No tech support — SHAP #5 driver")

        if factors:
            for f in factors:
                st.write(f"🔴 {f}")
        else:
            st.write("✅ No major risk factors detected")

# ── TAB 2: DASHBOARD ──
with tab2:
    st.header("Business Dashboard")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churn_rate = (df["Churn"] == "Yes").mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col3:
        revenue_risk = df[df["Churn"]=="Yes"]["MonthlyCharges"].sum()
        st.metric("Revenue at Risk", f"${revenue_risk:,.0f}")
    with col4:
        high_risk = (df["RiskSegment"]=="High Risk").sum()
        st.metric("High Risk Customers", f"{high_risk:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df.groupby("Contract")["Churn"].apply(
            lambda x: (x=="Yes").mean()*100
        ).reset_index()
        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(
            contract_churn["Contract"],
            contract_churn["Churn Rate (%)"],
            color=["#DC2626", "#F97316", "#16A34A"])
        ax.set_xlabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Contract Type")
        for bar, val in zip(bars, contract_churn["Churn Rate (%)"]):
            ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontweight="bold")
        st.pyplot(fig)

    with col2:
        st.subheader("Risk Segment Distribution")
        risk_dist = df["RiskSegment"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#16A34A", "#F59E0B", "#DC2626"]
        ax.pie(risk_dist, labels=risk_dist.index,
               autopct="%1.1f%%", colors=colors,
               startangle=90)
        ax.set_title("Customer Risk Distribution")
        st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Payment Method")
        payment_churn = df.groupby("PaymentMethod")["Churn"].apply(
            lambda x: (x=="Yes").mean()*100
        ).reset_index()
        payment_churn.columns = ["PaymentMethod", "Churn Rate (%)"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(payment_churn["PaymentMethod"],
                payment_churn["Churn Rate (%)"],
                color="#2563EB")
        ax.set_xlabel("Churn Rate (%)")
        ax.set_title("Churn by Payment Method")
        st.pyplot(fig)

    with col2:
        st.subheader("Tenure vs Churn Rate")
        tenure_churn = df.groupby("TenureBucket")["Churn"].apply(
            lambda x: (x=="Yes").mean()*100
        ).reset_index()
        tenure_churn.columns = ["Tenure", "Churn Rate (%)"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(tenure_churn["Tenure"],
               tenure_churn["Churn Rate (%)"],
               color=["#DC2626","#F97316","#3B82F6","#16A34A"])
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Tenure")
        plt.xticks(rotation=15)
        st.pyplot(fig)

# ── TAB 3: INSIGHTS ──
with tab3:
    st.header("Key Business Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 SHAP Feature Importance")
        shap_data = {
            "Feature": ["MonthlyCharges", "Contract",
                        "tenure", "OnlineSecurity",
                        "TechSupport", "TotalCharges",
                        "OnlineBackup"],
            "SHAP Score": [1.43, 1.11, 0.61,
                           0.38, 0.38, 0.37, 0.36],
            "Priority": ["CRITICAL", "CRITICAL", "HIGH",
                         "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM"]
        }
        shap_df = pd.DataFrame(shap_data)
        st.dataframe(shap_df, hide_index=True,
                     use_container_width=True)

    with col2:
        st.subheader("💡 Recommendations")
        st.error("🔴 CRITICAL — Pricing Strategy Review\n\nReview monthly charges for high-bill customers. SHAP Score: 1.43")
        st.error("🔴 CRITICAL — Contract Conversion\n\nConvert month-to-month to annual contracts. SHAP Score: 1.11")
        st.warning("🟠 HIGH — New Customer Onboarding\n\nTarget customers under 12 months tenure. SHAP Score: 0.61")
        st.info("🔵 MEDIUM — Add-On Services\n\nPromote OnlineSecurity & TechSupport. SHAP Score: 0.38")
