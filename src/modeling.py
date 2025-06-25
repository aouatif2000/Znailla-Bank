import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    joblib.dump(model, "output/model.joblib")
    return model

def predict(model, X_new):
    proba = model.predict_proba(X_new)[:, 1]
    predictions = pd.DataFrame({'probability': proba})
    return predictions.sort_values(by='probability', ascending=False)
