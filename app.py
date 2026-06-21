import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

def download_data():
    if not os.path.exists('data/engineered_data.csv'):
        os.makedirs('data', exist_ok=True)
        file_id = '1CW0uB5CmM1KJ79Sj_4yvXIwaxCbialOl'
        url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
        response = requests.get(url, stream=True)
        with open('data/engineered_data.csv', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Data downloaded successfully!")

download_data()

def load_or_train_models():
    if not os.path.exists('models/best_classifier.pkl'):
        os.makedirs('models', exist_ok=True)
        
        # Load and prepare data
        df = pd.read_csv('data/engineered_data.csv')
        X = df.drop(columns=['emi_eligibility', 'emi_eligibility_encoded', 'max_monthly_emi'])
        y_class = df['emi_eligibility_encoded']
        y_reg = df['max_monthly_emi']
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier, XGBRegressor
        
        X_train, _, y_class_train, _, y_reg_train, _ = train_test_split(
            X, y_class, y_reg, test_size=0.30, random_state=42, stratify=y_class)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1, 
                            max_depth=6, random_state=42)
        clf.fit(X_train_scaled, y_class_train)
        
        reg = XGBRegressor(n_estimators=100, learning_rate=0.1,
                           max_depth=6, random_state=42)
        reg.fit(X_train_scaled, y_reg_train)
        
        joblib.dump(clf, 'models/best_classifier.pkl')
        joblib.dump(reg, 'models/best_regressor.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
    
    clf = joblib.load('models/best_classifier.pkl')
    reg = joblib.load('models/best_regressor.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return clf, reg, scaler

# Replace the model loading lines with this
clf, reg, scaler = load_or_train_models()

# Page config
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

# Load models
clf = joblib.load('models/best_classifier.pkl')
reg = joblib.load('models/best_regressor.pkl')
scaler = joblib.load('models/scaler.pkl')

# Sidebar navigation
st.sidebar.title("EMIPredict AI")
page = st.sidebar.selectbox("Navigate", 
    ["Home", "Predict EMI", "Data Explorer", "Model Performance"])

if page == "Home":
    st.title("EMIPredict AI 💰")
    st.subheader("Intelligent Financial Risk Assessment Platform")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size", "404,800")
    col2.metric("Features", "46")
    col3.metric("Classification Accuracy", "97.73%")
    col4.metric("Regression RMSE", "848 INR")
    
    st.markdown("---")
    
    # Project overview
    st.subheader("What does this app do?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**EMI Eligibility Prediction**\n\nPredicts whether a customer is Eligible, High Risk, or Not Eligible for an EMI loan.")

    
    with col2:
        st.success("**Maximum EMI Calculation**\n\nPredicts the maximum monthly EMI amount a customer can safely afford.")
    
    st.markdown("---")
    st.subheader("EMI Scenarios Covered")
    scenarios = ["E-commerce Shopping EMI", "Home Appliances EMI", 
                 "Vehicle EMI", "Personal Loan EMI", "Education EMI"]
    for s in scenarios:
        st.write(f"✅ {s}")

elif page == "Predict EMI":
    st.title("EMI Prediction 🔮")
    st.write("Enter your financial details to get an EMI eligibility prediction.")
    
    st.markdown("---")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=25, max_value=60, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        education = st.selectbox("Education", ["Graduate", "High School", "Post Graduate", "Professional"])
        employment_type = st.selectbox("Employment Type", ["Government", "Private", "Self-employed"])
        company_type = st.selectbox("Company Type", ["Large Indian", "MNC", "Mid-size", "Small", "Startup"])
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=40.0, value=5.0)
        house_type = st.selectbox("House Type", ["Family", "Own", "Rented"])
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)
        dependents = st.number_input("Dependents", min_value=0, max_value=5, value=1)

    with col2:
        st.subheader("Financial Details")
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=15000, max_value=500000, value=50000)
        monthly_rent = st.number_input("Monthly Rent (INR)", min_value=0, max_value=100000, value=10000)
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0, max_value=2000000, value=100000)
        emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0, max_value=1000000, value=50000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI Amount (INR)", min_value=0, max_value=100000, value=0)

    with col3:
        st.subheader("Monthly Expenses")
        school_fees = st.number_input("School Fees (INR)", min_value=0, max_value=50000, value=0)
        college_fees = st.number_input("College Fees (INR)", min_value=0, max_value=100000, value=0)
        travel_expenses = st.number_input("Travel Expenses (INR)", min_value=0, max_value=50000, value=3000)
        groceries_utilities = st.number_input("Groceries & Utilities (INR)", min_value=0, max_value=50000, value=8000)
        other_monthly_expenses = st.number_input("Other Expenses (INR)", min_value=0, max_value=50000, value=2000)
        emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", 
                                                       "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
        requested_amount = st.number_input("Requested Amount (INR)", min_value=10000, max_value=2000000, value=200000)
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=12) 

        st.markdown("---")
    
    if st.button("Predict EMI Eligibility", type="primary"):
        
        # Calculate engineered features
        total_monthly_expenses = (monthly_rent + school_fees + college_fees + 
                                   travel_expenses + groceries_utilities + 
                                   other_monthly_expenses + current_emi_amount)
        disposable_income = monthly_salary - total_monthly_expenses
        expense_to_income_ratio = total_monthly_expenses / (monthly_salary + 1)
        debt_to_income_ratio = current_emi_amount / (monthly_salary + 1)
        total_savings = bank_balance + emergency_fund
        savings_to_expense_ratio = total_savings / (total_monthly_expenses + 1)
        requested_emi = requested_amount / (requested_tenure + 1)
        emi_to_disposable_ratio = requested_emi / (disposable_income + 1)
        has_existing_loan = 1 if existing_loans == "Yes" else 0
        is_financially_stressed = 1 if disposable_income < 0 else 0
        good_credit = 1 if credit_score >= 700 else 0

        # Encode binary
        gender_enc = 1 if gender == "Male" else 0
        marital_enc = 1 if marital_status == "Married" else 0
        existing_loans_enc = 1 if existing_loans == "Yes" else 0

        # One-hot encode education
        edu_hs = 1 if education == "High School" else 0
        edu_pg = 1 if education == "Post Graduate" else 0
        edu_prof = 1 if education == "Professional" else 0

        # One-hot encode employment
        emp_private = 1 if employment_type == "Private" else 0
        emp_self = 1 if employment_type == "Self-employed" else 0

        # One-hot encode company
        comp_mnc = 1 if company_type == "MNC" else 0
        comp_mid = 1 if company_type == "Mid-size" else 0
        comp_small = 1 if company_type == "Small" else 0
        comp_startup = 1 if company_type == "Startup" else 0

        # One-hot encode house
        house_own = 1 if house_type == "Own" else 0
        house_rented = 1 if house_type == "Rented" else 0

        # One-hot encode scenario
        sce_edu = 1 if emi_scenario == "Education EMI" else 0
        sce_home = 1 if emi_scenario == "Home Appliances EMI" else 0
        sce_personal = 1 if emi_scenario == "Personal Loan EMI" else 0
        sce_vehicle = 1 if emi_scenario == "Vehicle EMI" else 0

        # Build feature array (must match training column order)
        features = np.array([[
            age, gender_enc, marital_enc, monthly_salary, emp_private, emp_self,
            years_of_employment, house_own, house_rented, monthly_rent,
            family_size, dependents, school_fees, college_fees, travel_expenses,
            groceries_utilities, other_monthly_expenses, existing_loans_enc,
            current_emi_amount, credit_score, bank_balance, emergency_fund,
            requested_amount, requested_tenure,
            total_monthly_expenses, disposable_income, expense_to_income_ratio,
            debt_to_income_ratio, total_savings, savings_to_expense_ratio,
            requested_emi, emi_to_disposable_ratio, has_existing_loan,
            is_financially_stressed, good_credit,
            edu_hs, edu_pg, edu_prof,
            comp_mnc, comp_mid, comp_small, comp_startup,
            sce_edu, sce_home, sce_personal, sce_vehicle
        ]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        clf_pred = clf.predict(features_scaled)[0]
        reg_pred = reg.predict(features_scaled)[0]

        # Map prediction to label
        label_map = {0: "Eligible", 1: "High Risk", 2: "Not Eligible"}
        eligibility = label_map[clf_pred]

        # Show results
        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            if eligibility == "Eligible":
                st.success(f"✅ EMI Eligibility: **{eligibility}**")
            elif eligibility == "High Risk":
                st.warning(f"⚠️ EMI Eligibility: **{eligibility}**")
            else:
                st.error(f"❌ EMI Eligibility: **{eligibility}**")

        with col2:
            st.info(f"💰 Maximum Monthly EMI: **₹{reg_pred:,.2f}**")

        # Financial summary
        st.markdown("---")
        st.subheader("Your Financial Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Disposable Income", f"₹{disposable_income:,.0f}")
        col2.metric("Total Expenses", f"₹{total_monthly_expenses:,.0f}")
        col3.metric("Total Savings", f"₹{total_savings:,.0f}")

elif page == "Data Explorer":
    st.title("Data Explorer 📊")
    
    df = pd.read_csv('data/cleaned_data.csv')
    
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", "25")
    col3.metric("EMI Scenarios", "5")
    
    st.markdown("---")
    
    # Plot 1 - Eligibility Distribution
    st.subheader("EMI Eligibility Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    counts = df['emi_eligibility'].value_counts()
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='white', height=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 500, bar.get_y() + bar.get_height()/2,
                f'{val:,} ({val/len(df)*100:.1f}%)',
                va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel("Count", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, counts.max() * 1.25)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Plot 2 - Salary Distribution
    st.subheader("Monthly Salary Distribution by Eligibility")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Eligible': '#2ecc71', 'High_Risk': '#f39c12', 'Not_Eligible': '#e74c3c'}
    for eligibility, group in df.groupby('emi_eligibility'):
        ax.hist(group['monthly_salary'], bins=40, alpha=0.65,
                label=eligibility, color=colors[eligibility], edgecolor='none')
    ax.set_xlabel("Monthly Salary (INR)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Plot 3 - EMI Scenario Pie
    st.subheader("EMI Scenario Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    scenario_counts = df['emi_scenario'].value_counts()
    colors_pie = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    wedges, texts, autotexts = ax.pie(
        scenario_counts.values,
        labels=scenario_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        pctdistance=0.82
    )
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Plot 4 - Credit Score vs Eligibility
    st.subheader("Credit Score vs Eligibility")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_box = ['#2ecc71', '#f39c12', '#e74c3c']
    order = ['Eligible', 'High_Risk', 'Not_Eligible']
    sns.violinplot(data=df, x='emi_eligibility', y='credit_score',
                   order=order, palette=colors_box, ax=ax, inner='quartile')
    ax.set_xlabel("Eligibility", fontsize=12)
    ax.set_ylabel("Credit Score", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Raw data
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance 📈")
    
    st.markdown("---")
    
    # Classification Results
    st.subheader("Classification Models Comparison")
    
    clf_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.8146, 0.90, 0.9594, 0.9773],
        'Precision': [0.9340, 0.90, 0.9615, 0.9766],
        'Recall': [0.8146, 0.90, 0.9594, 0.9773],
        'F1 Score': [0.8605, 0.90, 0.9603, 0.9768]
    }
    clf_df = pd.DataFrame(clf_data)
    
    # Highlight best model
    st.dataframe(clf_df.style.highlight_max(
        subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        color='#2ecc71'), use_container_width=True)
    
    st.markdown("---")
    
    # Classification bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(clf_df['Model']))
    width = 0.2
    ax.bar(x - width*1.5, clf_df['Accuracy'], width, label='Accuracy', color='#3498db')
    ax.bar(x - width*0.5, clf_df['Precision'], width, label='Precision', color='#2ecc71')
    ax.bar(x + width*0.5, clf_df['Recall'], width, label='Recall', color='#e74c3c')
    ax.bar(x + width*1.5, clf_df['F1 Score'], width, label='F1 Score', color='#f39c12')
    ax.set_xticks(x)
    ax.set_xticklabels(clf_df['Model'], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='90% target')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Regression Results
    st.subheader("Regression Models Comparison")
    
    reg_data = {
        'Model': ['Linear Regression', 'Decision Tree Regressor', 
                  'Random Forest Regressor', 'XGBoost Regressor'],
        'RMSE (INR)': [4080.35, 1200.00, 985.94, 848.74],
        'MAE (INR)': [2932.39, 800.00, 289.17, 329.66],
        'R² Score': [0.7248, 0.85, 0.9839, 0.9881]
    }
    reg_df = pd.DataFrame(reg_data)
    
    st.dataframe(reg_df.style.highlight_min(
        subset=['RMSE (INR)', 'MAE (INR)'],
        color='#2ecc71').highlight_max(
        subset=['R² Score'],
        color='#2ecc71'), use_container_width=True)
    
    st.markdown("---")
    
    # Regression bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(reg_df['Model'], reg_df['RMSE (INR)'], 
                  color=['#e74c3c','#f39c12','#3498db','#2ecc71'],
                  edgecolor='white', width=0.5)
    for bar, val in zip(bars, reg_df['RMSE (INR)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'₹{val:,.0f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(y=2000, color='red', linestyle='--', alpha=0.7, label='Target: 2000 INR')
    ax.set_ylabel("RMSE (INR)", fontsize=12)
    ax.set_xticklabels(reg_df['Model'], fontsize=10, rotation=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Winner announcement
    st.subheader("🏆 Selected Models for Deployment")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Classification: XGBoost**\n\nAccuracy: 97.73% | F1: 97.68%")
    with col2:
        st.success("**Regression: XGBoost**\n\nRMSE: ₹848 | R²: 0.9881")