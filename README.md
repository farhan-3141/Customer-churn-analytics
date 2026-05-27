# 📊 Customer Churn Analysis & Retention Dashboard

🚀 **Live App:** https://churn-analytics-farhan.streamlit.app

> End-to-end churn analytics on 7,043 Telco customer records 
> with XGBoost, SHAP explainability, SQL cohort analysis, 
> and Power BI dashboard

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 81% | 0.85 |
| Random Forest | 92% | 0.97 |
| XGBoost | 93% | 0.98 |

---

## 🔑 SHAP Feature Importance

| Feature | SHAP Score | Priority |
|---------|-----------|----------|
| MonthlyCharges | 1.43 | 🔴 CRITICAL |
| Contract | 1.11 | 🔴 CRITICAL |
| Tenure | 0.61 | 🟠 HIGH |
| OnlineSecurity | 0.38 | 🟡 MEDIUM |
| TechSupport | 0.38 | 🟡 MEDIUM |

---

## 💡 Key Business Insights

- Overall churn rate: **26.5%**
- Monthly revenue at risk: **$139,130**
- Month-to-month churn rate: **42%**
- MonthlyCharges is #1 churn predictor (SHAP validated)
- New customers (0-12 months) are highest risk segment

---

## 🛠️ Tech Stack

- **Languages:** Python, SQL
- **ML:** XGBoost, Scikit-learn, SHAP, SMOTE
- **Analytics:** Pandas, NumPy, pandasql
- **Visualization:** Matplotlib, Seaborn, Power BI
- **Deployment:** Streamlit, GitHub

---

## 📁 Repository Structure

```
Customer-churn-analytics/
├── app.py                    
├── requirements.txt          
├── churn_data_powerbi.csv    
└── churn_model.pkl           
```
---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Farhan Ansari** | M.Tech Data Analytics | IIT (ISM) Dhanbad  
📧 farhan.mas101@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/farhanansari-iitdhanbad)  
🚀 [Live App](https://churn-analytics-farhan.streamlit.app)
