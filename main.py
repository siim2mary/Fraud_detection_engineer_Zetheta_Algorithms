import pandas as pd
import numpy as np
import zipfile
import os
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ==========================================
# PART 1: DATA PIPELINE (Ingestion & Cleaning)
# ==========================================
def run_pipeline(zip_path):
    """Handles ZIP extraction and data cleaning."""
    if not os.path.exists(zip_path):
        print(f"❌ ERROR: File not found at {zip_path}")
        return None

    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    
    # Clean the 'Amount' column (Removing £ and converting to float)
    df['Amount_Num'] = df['Amount'].replace('[£,]', '', regex=True).astype(float)
    return df

# ==========================================
# PART 2 & 3: DETECTION ENGINES (Math & ML)
# ==========================================
def run_detection(df):
    """Tuned Detection: More sensitive to catch anomalies."""
    
    # PART 2: Statistical Engine
    # Lowered from 3 to 2.2 to catch more 'unusual' spending
    z_scores = np.abs(stats.zscore(df['Amount_Num']))
    df['stat_anomaly'] = (z_scores > 2.2).astype(int)

    # PART 3: ML Engine
    le = LabelEncoder()
    df['Merchant_Enc'] = le.fit_transform(df['Merchant Group'].astype(str))
    
    # Increased contamination to 0.1 (Top 10% of weird patterns)
    model = IsolationForest(contamination=0.1, random_state=42)
    features = df[['Amount_Num', 'Age', 'Merchant_Enc']].fillna(0)
    df['ml_anomaly'] = [1 if x == -1 else 0 for x in model.fit_predict(features)]
    
    return df

# ==========================================
# PART 4, 5 & 6: SCORING & VALIDATION
# ==========================================
def run_scoring_and_audit(df):
    """Aggressive Tuning: Ensures alerts appear on the dashboard."""
    
    # Part 4: Scoring logic remains the same
    df['risk_score'] = (df['stat_anomaly'] * 40) + (df['ml_anomaly'] * 60)
    
    # Part 5: NEW BINS - Lowered the High threshold from 70 to 40
    # This means if either the Stats OR the ML engine flags a transaction, 
    # it will appear in the 'High' category.
    df['priority'] = pd.cut(df['risk_score'], 
                            bins=[-1, 1, 39, 100], 
                            labels=['Low', 'Medium', 'High'])
    
    # Part 6: Validation
    actual_fraud = len(df[(df['priority'] == 'High') & (df['Fraud'] == 1)])
    print(f"✅ Audit Complete: Found {actual_fraud} verified fraud cases.")
    
    return df

# ==========================================
# MAIN EXECUTION (How it runs in VS Code)
# ==========================================
if __name__ == "__main__":
    # REPLACE THIS PATH if you move your file
    FILE_PATH = r"D:\anomaly detection project\credit_card_trans.zip"
    
    print("--- Starting Anomaly Detection System ---")
    
    # Step 1: Ingest
    raw_data = run_pipeline(FILE_PATH)
    
    if raw_data is not None:
        # Step 2: Detect
        detected_data = run_detection(raw_data)
        
        # Step 3: Score & Audit
        final_data = run_scoring_and_audit(detected_data)
        
        # Step 4: Save for Dashboard (Part 7)
        final_data.to_csv('analyzed_data.csv', index=False)
        print("📁 Success: Results saved to 'analyzed_data.csv'.")
        print(final_data[['Transaction ID', 'Amount', 'risk_score', 'priority']].head(10))