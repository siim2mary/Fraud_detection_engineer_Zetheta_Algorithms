#Part 7	
# Surveillance Dashboard	
# anomaly_app.py (The main Streamlit interface that integrates all the above).
#================================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import our custom logic from Parts 1-6
from data_pipeline import run_data_pipeline # Part 1
from stats_model import StatisticalDetector # Part 2
from ml_model import MLPatternDetector     # Part 3
from alert_scoring import AlertPrioritizer  # Part 4
from case_manager import CaseManager        # Part 5
from tracker import ModelTuner              # Part 6

# --- Page Configuration ---
st.set_page_config(page_title="Zetheta Anomaly Surveillance", layout="wide")

# --- Initialize Modules ---
# These act as our backend engines
if 'manager' not in st.session_state:
    st.session_state.manager = CaseManager()
    st.session_state.tuner = ModelTuner()

# --- Part 7: Dashboard UI Layout ---
st.title("🔍 Zetheta Anomaly Detection & Surveillance Platform")
st.markdown("---")

# --- Part 1 & 2: Pipeline & Statistical Check ---
# We pull the simulated data stream and run the Z-Score engine
raw_df = run_data_pipeline()
stat_engine = StatisticalDetector(z_threshold=2.5)
raw_df['stat_anomaly'] = stat_engine.calculate_z_score(raw_df['amount'])

# --- Part 3: ML Pattern Recognition ---
# We use the Isolation Forest to find hidden patterns
ml_engine = MLPatternDetector(contamination=0.10)
processed_df = ml_engine.train_and_predict(raw_df)
processed_df = ml_engine.get_anomaly_scores(processed_df)

# --- Part 4: Scoring & Prioritization ---
# Merge all signals into a 0-100 Risk Score
ranker = AlertPrioritizer()
final_df = ranker.calculate_priority(processed_df)

# --- Dashboard Visualizations (The 'Surveillance' aspect) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live Transaction Risk Map")
    # Color-coding transactions by their ML anomaly status
    fig = px.scatter(final_df, x="amount", y="final_risk_score", 
                     color="priority_level", size="amount",
                     hover_data=['priority_level'],
                     title="Transaction Value vs. Risk Score")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🚨 Priority Alert Queue")
    # Show only the high-risk items requiring immediate human eyes
    st.dataframe(final_df[['amount', 'priority_level', 'final_risk_score']].head(5))

# --- Part 5: Case Management Workflow ---
st.markdown("---")
st.subheader("📁 Investigation Workflow")
case_col1, case_col2 = st.columns(2)

with case_col1:
    # Allow analyst to select a high-risk case to review
    target_id = st.selectbox("Select Transaction for Review (ID):", final_df.index)
    selected_amt = final_df.loc[target_id, 'amount']
    st.info(f"Reviewing Transaction {target_id} for ${selected_amt}")

with case_col2:
    # Decision buttons for the analyst
    decision = st.radio("Resolution Action:", ["Confirmed Fraud", "False Positive", "Legitimate Spike"])
    notes = st.text_input("Analyst Audit Notes:")
    if st.button("Submit Final Decision"):
        # Log the decision to the case manager
        msg = st.session_state.manager.update_case_status(str(target_id), decision, notes)
        st.success("Case Resolved and Logged for Audit.")

# --- Part 6: Model Tuning & Feedback Loop ---
with st.sidebar:
    st.header("⚙️ Model Control Center")
    # Calculate performance based on recent decisions
    fpr = st.session_state.tuner.track_accuracy(pd.Series([decision]))
    new_param, advice = st.session_state.tuner.suggest_tuning(0.10, fpr)
    
    st.metric("False Positive Rate", f"{fpr}%")
    st.warning(f"Advice: {advice}")
    
    # Allow manual override of ML sensitivity
    st.slider("Adjust ML Sensitivity (Contamination)", 0.01, 0.20, new_param)