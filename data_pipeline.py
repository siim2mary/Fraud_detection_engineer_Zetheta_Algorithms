#Part 1	
# Architecture & Pipeline	
# data_pipeline.py (Simulates ingestion, cleaning, and feature engineering for velocity/geo)
#-----------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import datetime

def run_data_pipeline():
    """
    Part 1: Anomaly Detection Architecture and Data Pipeline
    This script simulates the ingestion, cleaning, and feature engineering 
    required for a banking anomaly detection system.
    """
    # Set seed for consistent results during testing
    np.random.seed(42)
    
    # --- 1. RAW DATA INGESTION SIMULATION ---
    # Creating 100 normal transactions representing daily banking activity
    n_rows = 100
    data = {
        'transaction_id': [f"TXN-{1000+i}" for i in range(n_rows)],
        'amount': np.random.normal(120, 35, n_rows).tolist(), # Average spend is $120
        'type': np.random.choice(['POS_Purchase', 'ATM_Withdrawal', 'Online_Transfer'], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # --- 2. FEATURE ENGINEERING (Part 1 Deliverable) ---
    # Velocity Score: Frequency of transactions (High = suspicious)
    df['velocity_score'] = np.random.uniform(0.1, 0.4, len(df))
    
    # Geo Score: Distance from home location (High = unusual location)
    df['geo_score'] = np.random.uniform(0.05, 0.3, len(df))
    
    # --- 3. INJECTING SYNTHETIC ANOMALIES (Testing Data) ---
    # We manually add known "Fraud" patterns to verify the models work
    anomalies = pd.DataFrame({
        'transaction_id': ['TXN-9997', 'TXN-9998', 'TXN-9999'],
        'amount': [9200.0, 15.50, 4800.0], # High value outliers
        'type': ['Online_Transfer', 'POS_Purchase', 'ATM_Withdrawal'],
        'velocity_score': [0.95, 0.98, 0.85], # High frequency patterns
        'geo_score': [0.92, 0.15, 0.97]      # Distant location patterns
    })
    
    # Combine normal records with our synthetic anomalies
    df = pd.concat([df, anomalies], ignore_index=True)
    
    # --- 4. DATA CLEANING & STANDARDIZATION ---
    # Rounding amounts for financial consistency
    df['amount'] = df['amount'].abs().round(2)
    
    # Adding timestamps to simulate a real-time event stream
    df['timestamp'] = pd.date_range(start=datetime.now(), periods=len(df), freq='min')
    
    return df

# --- Local Testing Block ---
if __name__ == "__main__":
    processed_data = run_data_pipeline()
    print("--- Data Pipeline Success ---")
    print(processed_data.tail()) # Shows the anomalies we injected
    print(f"\nAvailable Features: {list(processed_data.columns)}")