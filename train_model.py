import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense

from preprocess import preprocess

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset.csv")

# ---------------- PREPROCESS (WITH VISUALS) ----------------
data, scaler = preprocess(df, fit=True, visualize=True)

# ---------------- CREATE SEQUENCES ----------------
seq_len = 20
X, y = [], []

for i in range(seq_len, len(data)):
    X.append(data[i-seq_len:i])
    y.append(data[i, 4])  # temperature target

X = np.array(X)
y = np.array(y)

print("Data Shape:", X.shape)

# ---------------- 80/20 SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Flatten for GB
X_train_f = X_train.reshape(X_train.shape[0], -1)

# ---------------- GRU MODEL ----------------
gru = Sequential([
    GRU(32, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    GRU(16),
    Dense(1)
])

gru.compile(optimizer="adam", loss="mse")

print("\n🚀 Training GRU...")
gru.fit(X_train, y_train, epochs=10, batch_size=16)

# ---------------- LSTM MODEL ----------------
lstm = Sequential([
    LSTM(16, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(8),
    Dense(1)
])

lstm.compile(optimizer="adam", loss="mse")

print("\n🚀 Training LSTM...")
lstm.fit(X_train, y_train, epochs=8, batch_size=16)

# ---------------- GRADIENT BOOSTING ----------------
print("\n🚀 Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4)
gb.fit(X_train_f, y_train)

# ---------------- SAVE EVERYTHING ----------------
joblib.dump((gb, gru, lstm, scaler), "ensemble_model.pkl")

print("\n✅ Training complete & model saved successfully!")