import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter

def preprocess(df, scaler=None, fit=False, visualize=False):

    df = df.drop(columns=["timestamp","minute","second"], errors="ignore")
    df = df.dropna()

    # ---------------- VISUAL: RAW DATA ----------------
    if visualize:
        plt.figure(figsize=(12,5))
        plt.plot(df["temperature"], label="Temperature")
        plt.plot(df["cpu_percent"], label="CPU")
        plt.plot(df["throughput_mbps"], label="Throughput")
        plt.legend()
        plt.title("Raw Data Overview")
        plt.show()

    # ---------------- VISUAL: DISTRIBUTION ----------------
    if visualize:
        df[["cpu_percent","throughput_mbps","power","temperature"]].hist(
            bins=30, figsize=(10,6)
        )
        plt.suptitle("Feature Distribution")
        plt.show()

    # ---------------- VISUAL: CORRELATION ----------------
    if visualize:
        plt.figure(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Feature Correlation")
        plt.show()

    # ---------------- KALMAN FILTER ----------------
    kf = KalmanFilter(initial_state_mean=df["temperature"].iloc[0], n_dim_obs=1)
    temp_smooth, _ = kf.smooth(df["temperature"].values)

    df["temperature_smooth"] = temp_smooth

    # ---------------- VISUAL: SMOOTHING ----------------
    if visualize:
        plt.figure(figsize=(12,5))
        plt.plot(df["temperature"], label="Original")
        plt.plot(df["temperature_smooth"], label="Smoothed", linewidth=2)
        plt.legend()
        plt.title("Kalman Filter Smoothing")
        plt.show()

    # ---------------- FEATURES ----------------
    features = [
        "cpu_percent",
        "throughput_mbps",
        "byte_rate",
        "power",
        "temperature_smooth"
    ]

    data = df[features].values

    # ---------------- SCALING ----------------
    if fit:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        data_scaled = scaler.transform(data)

    # ---------------- VISUAL: SCALED DATA ----------------
    if visualize:
        plt.figure(figsize=(10,4))
        plt.plot(data_scaled[:,4])
        plt.title("Scaled Temperature (After Processing)")
        plt.show()

    # ---------------- RETURN ----------------
    if fit:
        return data_scaled, scaler   # 
    else:
        return data_scaled