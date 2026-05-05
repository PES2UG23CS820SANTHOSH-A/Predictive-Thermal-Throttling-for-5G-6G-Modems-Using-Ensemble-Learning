import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *

from preprocess import preprocess

# ---------------- CREATE OUTPUT FOLDER ----------------
os.makedirs("outputs", exist_ok=True)

# ---------------- LOAD ----------------
df = pd.read_csv("dataset.csv")

gb, gru, lstm, scaler = joblib.load("ensemble_model.pkl")

# ---------------- PREPROCESS ----------------
data = preprocess(df, scaler=scaler, fit=False)

# ---------------- SEQUENCE ----------------
seq_len = 20
X, y = [], []

for i in range(seq_len, len(data)):
    X.append(data[i-seq_len:i])
    y.append(data[i,4])

X = np.array(X)
y = np.array(y)

# ---------------- SPLIT ----------------
split = int(len(X)*0.8)

X_test = X[split:]
y_test = y[split:]

X_test_f = X_test.reshape(X_test.shape[0], -1)

# ---------------- PREDICTIONS ----------------
gru_pred = gru.predict(X_test).flatten()
lstm_pred = lstm.predict(X_test).flatten()
gb_pred = gb.predict(X_test_f)

final = 0.5*gb_pred + 0.3*gru_pred + 0.2*lstm_pred


# ---------------- METRICS ----------------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/(y_true+1e-8))) * 100

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)

    print(f"\n{name}")
    print(f"R2: {r2:.4f} ({r2*100:.2f}%)")
    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAPE: {mape_val:.2f}%")

# ---------------- PRINT METRICS ----------------
evaluate("GRU", y_test, gru_pred)
evaluate("LSTM", y_test, lstm_pred)
evaluate("Gradient Boosting", y_test, gb_pred)
evaluate("ENSEMBLE", y_test, final)


# ---------------- SUMMARY ----------------
print("\n📊 MODEL COMPARISON (%)")
models = {
    "GRU": r2_score(y_test, gru_pred),
    "LSTM": r2_score(y_test, lstm_pred),
    "GB": r2_score(y_test, gb_pred),
    "Ensemble": r2_score(y_test, final)
}

for name, score in models.items():
    print(f"{name}: {score*100:.2f}%")


# ---------------- PLOT 1 ----------------
plt.figure(figsize=(12,5))
plt.plot(y_test[:200], label="Actual", linewidth=2)
plt.plot(final[:200], label="Ensemble", linewidth=2)

m = r2_score(y_test, final)
plt.title(f"Prediction vs Actual | R2: {m*100:.2f}%")

plt.legend()
plt.savefig("outputs/prediction.png")
plt.show()


# ---------------- PLOT 2 ----------------
plt.figure(figsize=(6,4))
scores = [score*100 for score in models.values()]
bars = plt.bar(models.keys(), scores)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, h, f"{h:.2f}%", ha='center')

plt.title("Model Accuracy (%)")
plt.savefig("outputs/comparison.png")
plt.show()


# ---------------- PLOT 3 ----------------
res = y_test - final

plt.figure(figsize=(6,5))
plt.scatter(final, res)
plt.axhline(0, color='red')
plt.title("Residual Plot")

plt.savefig("outputs/residual.png")
plt.show()


# ---------------- PLOT 4 ----------------
plt.figure(figsize=(6,5))
plt.hist(res, bins=30)
plt.title("Error Distribution")

plt.savefig("outputs/error_dist.png")
plt.show()


# ---------------- PLOT 5 ----------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, final)
plt.plot([0,1],[0,1],'r--')

plt.title(f"Actual vs Predicted | R2: {m*100:.2f}%")

plt.savefig("outputs/actual_vs_pred.png")
plt.show()