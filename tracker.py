#Part 6	
# FP Tracking & Tuning	
# tracker.py (Calculates False Positive Rates and suggests model parameter changes).
#==============================================================================================


import pandas as pd

class ModelTuner:
    """
    Part 6: False Positive Tracking and Model Tuning
    Analyzes analyst decisions to optimize the detection engine's accuracy.
    """

    def __init__(self):
        # Dictionary to store performance metrics
        self.performance_metrics = {
            'total_alerts': 0,
            'true_positives': 0,
            'false_positives': 0
        }

    def track_accuracy(self, resolution_series):
        """
        Calculates the False Positive Rate (FPR) based on analyst resolutions.
        """
        # Count how many alerts were generated in this batch
        self.performance_metrics['total_alerts'] = len(resolution_series)
        
        # Count cases marked as 'Confirmed Fraud' by the analyst (True Positives)
        self.performance_metrics['true_positives'] = sum(resolution_series == 'Confirmed Fraud')
        
        # Count cases marked as 'False Positive' by the analyst
        self.performance_metrics['false_positives'] = sum(resolution_series == 'False Positive')
        
        # Calculate FPR: The percentage of alerts that were actually harmless
        if self.performance_metrics['total_alerts'] > 0:
            fpr = (self.performance_metrics['false_positives'] / self.performance_metrics['total_alerts']) * 100
        else:
            fpr = 0
            
        return fpr

    def suggest_tuning(self, current_contamination, fpr):
        """
        Recommends a new 'contamination' setting for the ML model in Part 3.
        If FPR is too high (e.g., > 20%), we should lower the model's sensitivity.
        """
        # Initialize the suggested setting with the current one
        suggested_setting = current_contamination
        
        # Logic: If more than 20% of alerts are false, the model is too "paranoid"
        if fpr > 20:
            # Reduce sensitivity by 1% to decrease alert volume
            suggested_setting = max(0.01, current_contamination - 0.01)
            reason = "High False Positive Rate detected. Reducing model sensitivity."
        
        # Logic: If 0% are false, the model might be missing real fraud (too "lenient")
        elif fpr == 0 and self.performance_metrics['total_alerts'] > 0:
            # Increase sensitivity by 1% to catch more subtle patterns
            suggested_setting = min(0.20, current_contamination + 0.01)
            reason = "Zero False Positives. Increasing sensitivity to capture missed threats."
        
        else:
            reason = "Model performance is within acceptable range. No tuning required."
            
        return suggested_setting, reason

# --- Local Testing ---
if __name__ == "__main__":
    # Mock data: Analyst reviewed 10 alerts, 4 were False Positives (40% FPR)
    resolutions = pd.Series(['False Positive', 'Confirmed Fraud', 'False Positive', 
                             'Confirmed Fraud', 'False Positive', 'Confirmed Fraud', 
                             'False Positive', 'Confirmed Fraud', 'Confirmed Fraud', 'Confirmed Fraud'])
    
    tuner = ModelTuner()
    current_fpr = tuner.track_accuracy(resolutions)
    
    # Get recommendation based on a current 5% contamination setting
    new_param, msg = tuner.suggest_tuning(0.05, current_fpr)
    
    print(f"Part 6 - Performance Report:")
    print(f"Current FPR: {current_fpr}%")
    print(f"Action: {msg}")
    print(f"Suggested ML Contamination Parameter: {new_param}")