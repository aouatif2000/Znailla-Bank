# Znailla Bank ML Campaign Optimization

A production-ready machine learning solution to help Znailla Bank reduce unnecessary calls and maximize profit through customer conversion prediction.

---

##  Project Overview

Znailla Bank has been running weekly call center campaigns to promote term deposit products.  
Each call costs **8€**, and each successful conversion brings **80€** of profit.

This project provides a predictive model that:
- Estimates the probability that a client will subscribe
- Helps the marketing team **prioritize calls**
- Improves campaign efficiency with **data-driven decisions**

---

##  Business Problem

- **Cost per call**: 8€
- **Profit per conversion**: 80€
- **Goal**: Reduce the number of unnecessary calls while keeping most of the conversions

---

##  Solution Highlights

- Classification model using **Random Forest**
- One-hot encoding for categorical features
- Metrics used: Precision, Recall, F1-score, ROC-AUC
- Output: A ranked list of customers with their probability of conversion

---



##  Architecture

```
znailla_bank_ml/
├── data/                  # Raw and input data
│   └── bank-additional-full.csv          
├── output/                # Predictions and trained model
│   ├── model.joblib
│   └── recommended_clients                      
├── src/                   # Source code
│   ├── data_preprocessing.py
│   ├── modeling.py
│   └── evaluate.py
├── main.py                # Main training and prediction script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

##  Quick Start

```bash
# Create and activate virtual environment
python -m venv 
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

This runs the full pipeline:
- Loads and preprocesses the dataset
- Trains the model
- Evaluates performance
- Simulates a weekly prediction on 500 customers
- Generates `output/recommended_clients.csv`

---

##  Weekly Predictions

By default, the pipeline simulates a random batch of 500 clients.
You can replace this with your actual weekly file in `main.py`:

```python
new_data = pd.read_csv("data/new_weekly_cohort.csv", sep=";")
```

The model will predict conversion probabilities and export a ranked list.

---

##  Model Deployment 

This project is designed to be extensible for deployment:
- Wrap the model in a **FastAPI** service for real-time prediction
- Schedule weekly batch predictions with **cron**, **Airflow**, or **Azure Data Factory**
- Monitor prediction performance and retrain monthly if needed

---

##  Author

Aouatif Allam – June 2025  
[LinkedIn](https://www.linkedin.com/in/aouatif-allam/) 
