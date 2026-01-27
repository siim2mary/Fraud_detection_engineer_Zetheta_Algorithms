# Fraud_detection_engineer_Zetheta_Algorithms
Live Dashboard: https://frauddetectionengineerzethetaalgorithms-769dauhbdaajj7tawjvvvb.streamlit.app/
Financial Fraud & Anomaly Detection Platform
Zetheta Algorithms - Project 1a Submission
This repository contains a production-ready Real-Time Anomaly Detection System designed to identify suspicious banking transactions using a hybrid of statistical methods and Machine Learning.

🚀 Live Demo
Access the Surveillance Dashboard here: https://frauddetectionengineerzethetaalgorithms-769dauhbdaajj7tawjvvvb.streamlit.app/

🏗️ Project Architecture (7-Part Deliverables)
This project was built over a 15-day development cycle, organized into the following modular components:

Part 1: Data Pipeline (data_pipeline.py) Simulates ingestion and feature engineering for transaction velocity and geographic displacement scores.

Part 2: Statistical Detection (stats_model.py) Implements Z-Score and Standard Deviation algorithms to flag immediate high-value outliers.

Part 3: ML Pattern Recognition (ml_model.py) Utilizes an Isolation Forest model to detect complex, multi-dimensional fraud patterns that statistics alone might miss.

Part 4: Alert Scoring Engine (alert_scoring.py) A weighted system that combines Statistical and ML outputs into a unified Risk Score (0-100).

Part 5: Investigation Workflow (case_manager.py) A simulated case management system allowing analysts to transition alerts from "Pending" to "Resolved."

Part 6: Model Tuning (tracker.py) Tracks False Positive Rates (FPR) and provides automated suggestions for threshold adjustments.

Part 7: Surveillance Dashboard (anomaly_app.py) The primary Streamlit interface integrating all modules into a single pane of glass for security analysts.

🛠️ Setup & Installation
Prerequisites
Python 3.11 or 3.12 (Recommended)

Git

Installation Steps
Clone the repository:

Bash

git clone https://github.com/siim2mary/Fraud_detection_engineer_Zetheta_Algorithms.git
cd Fraud_detection_engineer_Zetheta_Algorithms
Install dependencies:

Bash

pip install -r requirements.txt
Run the Platform locally:

Bash

streamlit run anomaly_app.py
📊 Technical Stack
Language: Python

Frontend: Streamlit

Analysis: Pandas, NumPy

Machine Learning: Scikit-Learn (Isolation Forest)

Visualization: Plotly
