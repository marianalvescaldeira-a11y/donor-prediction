import streamlit as st
import pandas as pd
import joblib

log_reg = joblib.load('logistic_model.joblib')
scaler = joblib.load('scaler.joblib')
imputer = joblib.load('knn_imputer.joblib')

st.set_page_config(page_title="Donor Predictor", layout="wide")

st.title("Machine Learning Project: Donor Outreach Interface")
st.write("This tool uses Logistic Regression to predict the probability of a donation.")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Demographics")
    DONOR_AGE = st.number_input("Donor age: ", min_value=18, max_value=100)
    CHILDREN = st.number_input("Number of children: ", min_value=0, max_value=20, value=0)
    SES = st.selectbox("Socio-Economic Status (1=Highest, 4=Lowest): ", [1, 2, 3, 4], index=1)
    
    HOME_OWNER_U = st.selectbox(
        "Is the individual a homeowner? ", 
        ["Yes", "No"],
        help="Select 'Yes' if confirmed, or 'No' if not recorded."
    )

    URBANICITY_R = st.selectbox("Do you live in a rural area? ", ["Yes", "No"],
        help="Select 'Yes' if confirmed, or 'No' if not recorded.")
    URBANICITY_S = st.selectbox("Do you live in a suburban area? ", ["Yes", "No"],
        help="Select 'Yes' if confirmed, or 'No' if not recorded.")
    URBANICITY_T = st.selectbox("Do you live in a town? ", ["Yes", "No"],
        help="Select 'Yes' if confirmed, or 'No' if not recorded.")

with col2:
    st.header("Donation History")
    LAST_GIFT_AMT = st.number_input("Amount donated in the individuals most recent donation: ", min_value=0.0)
    FILE_CARD_GIFT = st.number_input("Lifetime average donation from card solicitations: ", min_value=0.0)
    
    RECENCY_STATUS_96NK = st.selectbox("Donor status:", [
        "A (active - made first donation 12+ months ago, and donated in last 12 months)", 
        "E (inactive - made a donation 25+ months ago)", 
        "F (first time - first donation in last 6 months, exactly one donation)", 
        "L (lapsing - last donation between 13-24 months ago)", 
        "N (new - first donation in last 12 months, not a First time donor)", 
        "S (star donor)"
    ])
    
    DONOR_VELOCITY = st.number_input("Donor Velocity (FREQUENCY_STATUS_97NK / MONTHS_SINCE_LAST_GIFT):", min_value=0.0)

with col3:
    st.header("Promotion & Engagement")
    PEP_STAR = st.selectbox("Did the individual donate in 3 consecutive campaigns?", ["Yes", "No"], help="Select 'Yes' if confirmed, or 'No' if not recorded.")
    RECENT_STAR_STATUS = st.selectbox("In the last 4 years, did the individual donate in 3 consecutive campaigns?", ["Yes", "No"], help="Select 'Yes' if confirmed, or 'No' if not recorded.")
    RECENT_CARD_RESPONSE_COUNT = st.number_input("Number of times individual responded to promotion in last 4 years:", min_value=0)
    RECENT_CARD_RESPONSE_PROP = st.number_input("Proportion of responses to card solicitations in last 4 years: ", min_value=0.0, max_value=1.0)

st.header("Neighborhood Economic Data")
n_col1, n_col2, n_col3 = st.columns(3)
with n_col1:
    MEDIAN_HOME_VALUE = st.number_input("Individuals’ median home value in hundreds of units of euros: ", min_value=0, value=1500)
with n_col2:
    MEDIAN_HOUSEHOLD_INCOME = st.number_input("Individuals’ median household income in hundreds of units: ", min_value=0)
with n_col3:
    PER_CAPITA_INCOME = st.number_input("Per Capita Income of neighborhood in which individual lives: ", min_value=0)

if st.button("Calculate Donation Probability"):
    
    urb_r_val = 1 if URBANICITY_R == "Yes" else 0
    urb_s_val = 1 if URBANICITY_S == "Yes" else 0
    urb_t_val = 1 if URBANICITY_T == "Yes" else 0
    home_u_val = 1 if HOME_OWNER_U == "Yes" else 0
    pep_val = 1 if PEP_STAR == "Yes" else 0
    star_status_val = 1 if RECENT_STAR_STATUS == "Yes" else 0

    status_letter = RECENCY_STATUS_96NK[0]

    data_dict = {
        'DONOR_AGE': DONOR_AGE,
        'CHILDREN': CHILDREN,
        'SES': SES,
        'HOME_OWNER_U': home_u_val,
        'URBANICITY_R': urb_r_val,
        'URBANICITY_S': urb_s_val,
        'URBANICITY_T': urb_t_val,
        'LAST_GIFT_AMT': LAST_GIFT_AMT,
        'FILE_CARD_GIFT': FILE_CARD_GIFT,
        'DONOR_VELOCITY': DONOR_VELOCITY,
        'PEP_STAR': pep_val,
        'RECENT_STAR_STATUS': star_status_val,
        'RECENT_CARD_RESPONSE_COUNT': RECENT_CARD_RESPONSE_COUNT,
        'RECENT_CARD_RESPONSE_PROP': RECENT_CARD_RESPONSE_PROP,
        'MEDIAN_HOME_VALUE': MEDIAN_HOME_VALUE,
        'MEDIAN_HOUSEHOLD_INCOME': MEDIAN_HOUSEHOLD_INCOME,
        'PER_CAPITA_INCOME': PER_CAPITA_INCOME,
        'RECENCY_STATUS_96NK_A': 1 if status_letter == 'A' else 0,
        'RECENCY_STATUS_96NK_E': 1 if status_letter == 'E' else 0,
        'RECENCY_STATUS_96NK_F': 1 if status_letter == 'F' else 0,
        'RECENCY_STATUS_96NK_L': 1 if status_letter == 'L' else 0,
        'RECENCY_STATUS_96NK_N': 1 if status_letter == 'N' else 0,
        'RECENCY_STATUS_96NK_S': 1 if status_letter == 'S' else 0
    }
    
    full_df = pd.DataFrame([data_dict])
    
    
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
        for col in scaler_features:
            if col not in full_df.columns:
                full_df[col] = 0
        input_for_scaler = full_df[scaler_features]
        input_scaled = scaler.transform(input_for_scaler)
        processed_numeric = pd.DataFrame(input_scaled, columns=scaler_features)
    else:
        numeric_cols = ['DONOR_AGE', 'CHILDREN', 'LAST_GIFT_AMT', 'FILE_CARD_GIFT', 'DONOR_VELOCITY', 
                        'RECENT_CARD_RESPONSE_COUNT', 'RECENT_CARD_RESPONSE_PROP', 
                        'MEDIAN_HOME_VALUE', 'MEDIAN_HOUSEHOLD_INCOME', 'PER_CAPITA_INCOME']
        input_for_scaler = full_df[numeric_cols]
        input_scaled = scaler.transform(input_for_scaler.values)
        processed_numeric = pd.DataFrame(input_scaled, columns=numeric_cols)
    
    
    base_df = full_df.copy()
    for col in processed_numeric.columns:
        base_df[col] = processed_numeric[col].values

   
    if hasattr(imputer, 'feature_names_in_'):
        imputer_features = list(imputer.feature_names_in_)
        for col in imputer_features:
            if col not in base_df.columns:
                base_df[col] = 0
        input_for_imputer = base_df[imputer_features]
        input_imputed_scaled = imputer.transform(input_for_imputer)
        processed_df = pd.DataFrame(input_imputed_scaled, columns=imputer_features)
    else:
        
        n_features = imputer.n_features_in_
        if n_features == processed_numeric.shape[1]:
            input_imputed_scaled = imputer.transform(processed_numeric.values)
            processed_df = pd.DataFrame(input_imputed_scaled, columns=processed_numeric.columns)
        else:
            for col in base_df.columns:
                pass
            input_imputed_scaled = imputer.transform(base_df.values[:, :n_features])
            processed_df = pd.DataFrame(input_imputed_scaled, columns=base_df.columns[:n_features])

    
    for col in base_df.columns:
        if col not in processed_df.columns:
            processed_df[col] = base_df[col].values
            
   
    if hasattr(log_reg, 'feature_names_in_'):
        model_features = list(log_reg.feature_names_in_)
        for col in model_features:
            if col not in processed_df.columns:
                processed_df[col] = 0
        input_final = processed_df[model_features]
    else:
        expected_column_order = [
            'DONOR_AGE', 'CHILDREN', 'SES', 'HOME_OWNER_U', 'URBANICITY_R', 'URBANICITY_S', 'URBANICITY_T',
            'LAST_GIFT_AMT', 'FILE_CARD_GIFT', 'DONOR_VELOCITY', 'PEP_STAR', 'RECENT_STAR_STATUS',
            'RECENT_CARD_RESPONSE_COUNT', 'RECENT_CARD_RESPONSE_PROP', 'MEDIAN_HOME_VALUE',
            'MEDIAN_HOUSEHOLD_INCOME', 'PER_CAPITA_INCOME', 'RECENCY_STATUS_96NK_A', 'RECENCY_STATUS_96NK_E',
            'RECENCY_STATUS_96NK_F', 'RECENCY_STATUS_96NK_L', 'RECENCY_STATUS_96NK_N', 'RECENCY_STATUS_96NK_S'
        ]
        for col in expected_column_order:
            if col not in processed_df.columns:
                processed_df[col] = 0
        input_final = processed_df[expected_column_order]
    
    
    proba = log_reg.predict_proba(input_final)[0][1]
    
    st.divider()
    if proba > 0.45: #since we are dealing with donor/non donors, it's preferable to contact more potencial donors
        st.success(f"### Recommendation: CONTACT")
        st.write(f"The model predicts a **{proba:.1%}** chance of a donation.")
    else:
        st.error(f"### Recommendation: DO NOT CONTACT")
        st.write(f"The model predicts only a **{proba:.1%}** chance of a donation.")
