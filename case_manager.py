#Part 5	
# Investigation Workflow	
# case_manager.py (Handles Case IDs, status updates, and analyst audit notes).
#==============================================================================================

import pandas as pd
from datetime import datetime

class CaseManager:
    """
    Part 5: Investigation Workflow and Case Management
    Handles the transition of an anomaly from a 'Detected Alert' to a 'Resolved Case'.
    """

    def __init__(self):
        # We simulate a simple database for cases using a dictionary or DataFrame
        self.case_database = pd.DataFrame(columns=[
            'case_id', 'timestamp', 'priority', 'status', 'analyst_notes', 'resolution'
        ])

    def create_cases_from_alerts(self, prioritized_df):
        """
        Converts high-risk alerts into actionable cases for investigators.
        """
        # We only escalate CRITICAL and HIGH priority alerts to the case queue
        new_cases = prioritized_df[prioritized_df['priority_level'].isin(['🔴 CRITICAL', '🟠 HIGH'])].copy()
        
        # Assign unique Case IDs and initial 'Open' status
        new_cases['case_id'] = [f"CASE-{i}" for i in range(1001, 1001 + len(new_cases))]
        new_cases['status'] = 'Open'
        new_cases['analyst_notes'] = 'Pending Review'
        
        # Merge new cases into our permanent case database
        self.case_database = pd.concat([self.case_database, new_cases], ignore_index=True)
        return self.case_database

    def update_case_status(self, case_id, decision, notes):
        """
        Part 5 Workflow: Allows an analyst to resolve a case.
        Decisions: 'Confirmed Fraud', 'False Positive', 'Inquiry Sent'
        """
        # Locate the specific case in our database
        if case_id in self.case_database['case_id'].values:
            idx = self.case_database[self.case_database['case_id'] == case_id].index[0]
            
            # Update the status and add audit notes
            self.case_database.at[idx, 'status'] = 'Closed'
            self.case_database.at[idx, 'resolution'] = decision
            self.case_database.at[idx, 'analyst_notes'] = notes
            
            return f"Success: {case_id} has been resolved as {decision}."
        return "Error: Case ID not found."

# --- Local Testing ---
if __name__ == "__main__":
    # Mock data representing prioritized alerts from Part 4
    mock_prioritized_data = pd.DataFrame({
        'amount': [7000, 4500],
        'priority_level': ['🔴 CRITICAL', '🟠 HIGH'],
        'final_risk_score': [95, 75]
    })
    
    manager = CaseManager()
    cases = manager.create_cases_from_alerts(mock_prioritized_data)
    
    print("Part 5 - New Investigation Queue:")
    print(cases[['case_id', 'amount', 'priority_level', 'status']])
    
    # Simulate an analyst resolving a case
    result = manager.update_case_status("CASE-1001", "Confirmed Fraud", "User traveling, but amount exceeds limit.")
    print(f"\nWorkflow Action: {result}")