from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("ROC AUC Score:", roc_auc_score(y, y_proba))
    importances = model.feature_importances_
    features = X.columns
    sorted_idx = importances.argsort()[-10:]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()
