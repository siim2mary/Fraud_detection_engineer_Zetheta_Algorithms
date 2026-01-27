import pandas as pd
from sklearn.ensemble import IsolationForest

class MLPatternDetector:
    """
    Part 3: ML-based Pattern Recognition Models
    Uses Isolation Forest to detect complex fraud patterns that statistical models miss.
    """
    
    def __init__(self, contamination=0.05):
        # Contamination is the expected % of anomalies (e.g., 5% of transactions are fraud)
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def train_and_predict(self, df):
        """
        Trains the ML model on multi-dimensional data and flags pattern outliers.
        """
        # We select multiple features for pattern recognition
        # In a real bank, this would include: Amount, Time_of_Day, Distance_from_Home
        features = df[['amount', 'velocity_score', 'geo_score']]
        
        # Fit the model: It builds random trees to isolate data points
        self.model.fit(features)
        
        # Predict: 
        #  1 = Normal (inside the dense clusters)
        # -1 = Anomaly (isolated early in the trees)
        df['ml_prediction'] = self.model.predict(features)
        
        # Convert -1/1 format to a more readable 1/0 (1 for Anomaly)
        df['ml_anomaly'] = df['ml_prediction'].apply(lambda x: 1 if x == -1 else 0)
        
        return df

    def get_anomaly_scores(self, df):
        """
        Returns a 'normality score'. Lower/negative scores indicate higher fraud risk.
        """
        features = df[['amount', 'velocity_score', 'geo_score']]
        # Decision function returns the 'path length' to isolate the point
        df['anomaly_score'] = self.model.decision_function(features)
        return df

# --- Local Testing ---
if __name__ == "__main__":
    # Create sample data with patterns
    # (Amount, Velocity, Geography)
    data = {
        'amount': [50, 60, 55, 3000, 45],
        'velocity_score': [1, 1.2, 0.9, 8.5, 1.1], # 8.5 indicates high frequency
        'geo_score': [0.1, 0.1, 0.2, 0.9, 0.1]     # 0.9 indicates unusual location
    }
    df_test = pd.DataFrame(data)
    
    ml_engine = MLPatternDetector()
    results = ml_engine.train_and_predict(df_test)
    
    print("Part 3 - ML Pattern Recognition Results:")
    print(results[['amount', 'velocity_score', 'ml_anomaly']])