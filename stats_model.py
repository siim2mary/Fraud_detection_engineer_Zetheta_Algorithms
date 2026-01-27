#Part 2
# 	Statistical Algorithms
# 	stats_model.py (Uses Z-Score and Standard Deviation to catch high-value outliers).
#==============================================================================================

import pandas as pd
import numpy as np

class StatisticalDetector:
    """
    Part 2: Statistical Anomaly Detection Algorithms
    Focuses on 'Point Anomalies'—single data points that are extreme outliers.
    """
    
    def __init__(self, z_threshold=3.0):
        # The z_threshold determines sensitivity. 3.0 covers 99.7% of normal data.
        self.z_threshold = z_threshold

    def calculate_z_score(self, data_series):
        """
        Implementation of the Z-Score (Standard Score) algorithm.
        Formula: z = (x - mean) / std_dev
        """
        # Calculate mean (average) of the transaction amounts
        mean = np.mean(data_series)
        
        # Calculate standard deviation (variance from the average)
        std = np.std(data_series)
        
        # Prevent division by zero if all values are identical
        if std == 0: 
            return [0] * len(data_series)
        
        # Calculate how many standard deviations each point is from the mean
        z_scores = [(x - mean) / std for x in data_series]
        
        # Flag as 1 (Anomaly) if the absolute z-score is higher than our threshold
        return [1 if abs(z) > self.z_threshold else 0 for z in z_scores]

    def moving_average_check(self, df, window=5):
        """
        Detects 'Spike' anomalies by comparing current amount to a rolling window.
        Good for detecting sudden account draining.
        """
        # Create a rolling average of the last 'n' transactions
        df['rolling_mean'] = df['amount'].rolling(window=window, min_periods=1).mean()
        
        # Flag transactions that are significantly higher (e.g., 5x) than the recent average
        df['spike_alert'] = df['amount'] > (df['rolling_mean'] * 5)
        
        return df

# --- Local Testing ---
if __name__ == "__main__":
    # Sample data: Most transactions are $50-$150, one is $10,000 (The Anomaly)
    sample_amounts = [100, 110, 95, 105, 10000, 102] 
    
    detector = StatisticalDetector()
    flags = detector.calculate_z_score(sample_amounts)
    
    print("Part 2 - Statistical Test Results:")
    for amt, is_anomaly in zip(sample_amounts, flags):
        status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
        print(f"Amount: ${amt:<6} | Status: {status}")