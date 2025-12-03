import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model and Features
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/churn_model.pkl")
        model_columns = joblib.load("models/model_columns.pkl")
        return model, model_columns
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please run train_model.py first to train and save the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
model, model_columns = load_model()

# Sidebar
with st.sidebar:
    st.title("Customer Churn Predictor")
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    This application predicts customer churn probability using machine learning.
    
    What is Churn?
    Customer churn occurs when customers stop doing business with a company.
    
    How it works:
    1. Enter customer details
    2. Click 'Predict Churn Risk'
    3. Get prediction and insights
    """)
    
    st.markdown("---")
    st.header("Model Info")
    st.write(f"Model Type: XGBoost")
    st.write(f"Features Used: {len(model_columns)}")
    
    if st.checkbox("Show all features"):
        with st.expander("Feature List"):
            for i, col in enumerate(model_columns[:50]):
                st.write(f"{i+1}. {col}")
            if len(model_columns) > 50:
                st.write(f"... and {len(model_columns)-50} more")

# Main Content
st.title("Customer Churn Prediction Dashboard")
st.markdown("Enter customer details to predict churn risk and get actionable insights.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Insights", "Guide"])

with tab1:
    # Input Form
    st.header("Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior_citizen = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True, key="senior")
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True, key="partner")
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True, key="dependents")
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long the customer has been with the company")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], help="Contract duration")
        paperless_billing = st.radio("Paperless Billing", ["No", "Yes"], horizontal=True, key="paperless")
    
    with col2:
        st.subheader("Services")
        phone_service = st.radio("Phone Service", ["No", "Yes"], horizontal=True, key="phone")
        
        if phone_service == "Yes":
            multiple_lines = st.radio("Multiple Lines", ["No", "Yes"], horizontal=True, key="lines")
        else:
            multiple_lines = "No"
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help="Type of internet service")
        
        # Additional services
        if internet_service != "No":
            col2a, col2b = st.columns(2)
            with col2a:
                online_security = st.radio("Online Security", ["No", "Yes"], horizontal=True, key="security")
                online_backup = st.radio("Online Backup", ["No", "Yes"], horizontal=True, key="backup")
                device_protection = st.radio("Device Protection", ["No", "Yes"], horizontal=True, key="protection")
            with col2b:
                tech_support = st.radio("Tech Support", ["No", "Yes"], horizontal=True, key="support")
                streaming_tv = st.radio("Streaming TV", ["No", "Yes"], horizontal=True, key="tv")
                streaming_movies = st.radio("Streaming Movies", ["No", "Yes"], horizontal=True, key="movies")
        else:
            online_security = online_backup = device_protection = "No"
            tech_support = streaming_tv = streaming_movies = "No"
        
        st.subheader("Billing")
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check",
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
        
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="Monthly service charges")
        
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(monthly_charges * tenure), step=1.0, help="Total amount charged to customer")
    
    # Prediction Button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button("Predict Churn Risk", type="primary", use_container_width=True)
    
    # Prediction Logic
    if predict_button:
        with st.spinner("Analyzing customer data..."):
            # Prepare input dictionary
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Convert to DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Map binary columns to numeric
            df_input['gender'] = df_input['gender'].map({'Male': 1, 'Female': 0})
            
            binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            for col in binary_cols:
                df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})
            
            # Handle service columns
            service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            for col in service_cols:
                df_input[col] = df_input[col].replace({'No': 'No'})
            
            # One-hot encode categorical features
            categorical_cols = [
                'InternetService', 'Contract', 'PaymentMethod', 'MultipleLines',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies'
            ]
            
            df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
            
            # Align with model columns
            for col in model_columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            
            # Reorder columns to match model
            df_input = df_input[model_columns]
            
            # Make prediction
            prediction = int(model.predict(df_input)[0])
            proba = model.predict_proba(df_input)[0]
            prob_churn = float(proba[1])
            
            # Display Results
            st.markdown("---")
            st.header("Prediction Results")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric("Churn Probability", f"{prob_churn:.1%}")
            
            with result_col2:
                if prediction == 1:
                    st.error("HIGH RISK")
                    st.write("Customer is likely to churn")
                else:
                    st.success("LOW RISK")
                    st.write("Customer is likely to stay")
            
            with result_col3:
                st.metric("Retention Probability", f"{float(proba[0]):.1%}")
            
            # Risk Level
            st.subheader("Risk Level")
            
            risk_level = float(prob_churn)
            
            if risk_level < 0.3:
                risk_label = "Low"
            elif risk_level < 0.7:
                risk_label = "Medium"
            else:
                risk_label = "High"
            
            # Progress bar with float conversion
            st.progress(float(risk_level), text=f"{risk_label} Risk ({risk_level:.1%})")
            
            # Risk indicator
            if risk_label == "Low":
                st.success(f"Risk Assessment: {risk_label}")
            elif risk_label == "Medium":
                st.warning(f"Risk Assessment: {risk_label}")
            else:
                st.error(f"Risk Assessment: {risk_label}")
            
            # Recommendations
            st.subheader("Recommendations")
            
            if prediction == 1:
                st.warning("""
                Immediate Actions Recommended:
                
                1. Proactive Outreach
                   - Schedule a personal call within 48 hours
                   - Send personalized retention offer
                
                2. Service Review
                   - Check for recent service issues
                   - Review payment history
                
                3. Retention Offer
                   - Consider loyalty discount (10-15%)
                   - Offer free month or upgrade
                
                4. Escalation
                   - Flag for manager review
                   - Add to high-risk customer list
                """)
            else:
                st.success("""
                Maintenance Actions:
                
                1. Retention Strategies
                   - Regular satisfaction surveys
                   - Loyalty program enrollment
                
                2. Growth Opportunities
                   - Consider upselling additional services
                   - Offer referral bonuses
                
                3. Monitoring
                   - Quarterly check-ins
                   - Monitor usage patterns for changes
                """)
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Top Risk Factors")
                
                importances = model.feature_importances_
                importances = [float(x) for x in importances]
                feat_imp = pd.Series(importances, index=model_columns)
                
                top_features = feat_imp.sort_values(ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                display_names = []
                for feat in top_features.index:
                    name = feat.replace('_', ' ')
                    name = name.replace('InternetService', 'Internet: ')
                    name = name.replace('Contract', 'Contract: ')
                    name = name.replace('PaymentMethod', 'Payment: ')
                    display_names.append(name)
                
                colors = plt.cm.RdYlGn_r([float(x)/max(top_features.values) for x in top_features.values])
                bars = ax.barh(range(len(top_features)), [float(x) for x in top_features.values], color=colors)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(display_names)
                ax.set_xlabel("Importance Score")
                ax.set_title("Top 10 Features Influencing Prediction")
                
                for i, v in enumerate(top_features.values):
                    ax.text(float(v) + 0.001, i, f'{float(v):.3f}', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Raw data
            with st.expander("View Input Data"):
                st.write("Customer Information:")
                st.json(input_data)
                st.write(f"Number of features sent to model: {len(df_input.columns)}")
                st.write(f"Model prediction confidence: {max([float(x) for x in proba]):.2%}")

with tab2:
    st.header("Churn Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Common Churn Factors")
        st.markdown("""
        1. Contract Type
           - Month-to-month: 43% churn rate
           - One year: 11% churn rate
           - Two year: 3% churn rate
        
        2. Internet Service
           - Fiber optic: Highest churn
           - DSL: Moderate churn
           - No internet: Lowest churn
        
        3. Payment Method
           - Electronic check: Highest churn
           - Automatic payments: Lowest churn
        """)
    
    with col2:
        st.subheader("Retention Drivers")
        st.markdown("""
        1. Tenure
           - < 12 months: High risk
           - 12-24 months: Medium risk
           - > 24 months: Low risk
        
        2. Services
           - Tech support reduces churn by 25%
           - Online security reduces churn by 30%
           - Multiple services increase loyalty
        
        3. Demographics
           - Senior citizens: More loyal
           - Dependents: Less likely to churn
        """)
    
    st.markdown("---")
    st.subheader("Churn Statistics")
    
    stats_data = {
        'Metric': ['Overall Churn Rate', 'Avg Customer Lifetime', 'Cost of Customer Acquisition', 'Value of Retention'],
        'Value': ['27%', '32 months', '$300', '5x Acquisition Cost']
    }
    
    st.table(pd.DataFrame(stats_data))

with tab3:
    st.header("User Guide")
    
    st.markdown("""
    How to Use This Application
    
    Step 1: Enter Customer Details
    - Fill in all fields in the Prediction tab
    - Be as accurate as possible for best results
    
    Step 2: Click Predict
    - Click the "Predict Churn Risk" button
    - Wait for analysis to complete
    
    Step 3: Review Results
    - Check the churn probability percentage
    - Review risk level (Low/Medium/High)
    - Read actionable recommendations
    
    Step 4: Take Action
    - High risk: Immediate intervention needed
    - Medium risk: Monitor closely
    - Low risk: Focus on retention strategies
    
    Understanding the Results
    
    Churn Probability: The likelihood (0-100%) that this customer will leave
    Risk Level: Simplified categorization of risk
    Top Factors: Most influential features in the prediction
    Recommendations: Specific actions based on risk level
    
    Tips for Best Results
    
    1. Complete Information: Ensure all fields are filled accurately
    2. Regular Updates: Update customer information quarterly
    3. Historical Data: Compare with previous predictions
    4. Team Collaboration: Share insights with customer service teams
    """)

# Footer
# Footer
st.markdown("---")
st.caption("Customer Churn Predictor v1.0 | Built with using Streamlit & XGBoost | Data: Telco Customer Churn Dataset | Developed by Mahendar reddy Lakkireddy")