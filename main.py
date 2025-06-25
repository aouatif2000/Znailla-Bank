import pandas as pd
from src.data_preprocessing import preprocess_data
from src.modeling import train_model, predict
from src.evaluate import evaluate_model

# Load dataset
data = pd.read_csv("data/bank-additional-full.csv", sep=';')

# Preprocess
X, y, encoder = preprocess_data(data)

# Train
model = train_model(X, y)

# Evaluate
evaluate_model(model, X, y)

# Simulate new weekly cohort
data_for_new = data.copy()
new_data = data_for_new.sample(500, random_state=1).reset_index(drop=True)

X_new, _, _ = preprocess_data(new_data, encoder=encoder)
probas = predict(model, X_new)["probability"].reset_index(drop=True)
# Attach probabilities to full client info
recommended = new_data.copy()
recommended["probability"] = probas

recommended.to_csv("output/recommended_clients.csv", index=False)

print("Model training and evaluation complete.")
