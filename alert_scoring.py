#Part 4
# 	Alert Scoring & Priority	
# alert_scoring.py (Combines Part 2 & 3 into a 0-100 score with Priority Levels).
#===============================================================================================


import pandas as pd

class AlertPrioritizer:
    """
    Part 4: Alert Scoring and Prioritization System
    Combines different detection signals into a unified 'Risk Score' (0-100).
    """

    def calculate_priority(self, df):
        """
        Weights statistical and ML signals to create a final priority rank.
        """
        # 1. Start with the ML Anomaly Score (normalized to 0-50)
        # Isolation Forest 'decision_function' gives lower scores to outliers
        # We invert and scale it so 50 is high risk.
        df['ml_component'] = (1 - df['anomaly_score']) * 25

        # 2. Add the Statistical component (Z-Score)
        # If it's a statistical outlier, we add 30 points to the risk
        df['stat_component'] = df['stat_anomaly'] * 30

        # 3. Add 'High Value' multiplier
        # Any transaction over a certain threshold gets a 20 point boost
        df['value_boost'] = df['amount'].apply(lambda x: 20 if x > 5000 else 0)

        # 4. Final Risk Score Calculation (Max 100)
        df['final_risk_score'] = df['ml_component'] + df['stat_component'] + df['value_boost']
        
        # Ensure the score does not exceed 100
        df['final_risk_score'] = df['final_risk_score'].clip(0, 100)

        # 5. Prioritization Labeling
        def assign_priority(score):
            if score > 80: return "🔴 CRITICAL"
            if score > 50: return "🟠 HIGH"
            if score > 25: return "🟡 MEDIUM"
            return "🟢 LOW"

        df['priority_level'] = df['final_risk_score'].apply(assign_priority)
        
        # Sort the dataframe so the analyst sees CRITICAL alerts first
        return df.sort_values(by='final_risk_score', ascending=False)

# --- Local Testing ---
if __name__ == "__main__":
    # Simulated data combining Part 2 and Part 3 outputs
    data = {
        'amount': [100, 7000, 50, 4500],
        'stat_anomaly': [0, 1, 0, 0],       # Flagged by Z-Score
        'anomaly_score': [0.1, -0.2, 0.05, -0.1] # ML score (lower is worse)
    }
    df_alerts = pd.DataFrame(data)
    
    ranker = AlertPrioritizer()
    prioritized_df = ranker.calculate_priority(df_alerts)
    
    print("Part 4 - Prioritized Alert Queue:")
    print(prioritized_df[['amount', 'final_risk_score', 'priority_level']])