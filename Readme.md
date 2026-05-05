# 🔥 Predictive Thermal Throttling for 5G/6G Modems

## 📌 Overview

This project presents an **AI-driven predictive thermal management system** for 5G/6G modems.
Instead of reacting to overheating, the system **predicts future temperature trends** and enables **proactive thermal throttling**.

The solution integrates **real-time data simulation, signal processing, and ensemble machine learning models** to improve system stability and performance.

---

## 🚀 Key Features

* 📡 Real-time network traffic simulation using **iPerf + Docker**
* 📊 Time-series temperature prediction
* 🧠 Ensemble learning using:

  * GRU (short-term patterns)
  * LSTM (long-term dependencies)
  * Gradient Boosting (nonlinear patterns)
* 🔧 Kalman Filter for noise reduction
* ⚡ Real-time inference pipeline
* 🖥️ Interactive dashboard using Streamlit

---

## 🏗️ System Architecture

The system follows a modular pipeline:

1. **Data Generation**

   * iPerf server/client inside Docker

2. **Data Storage**

   * CSV-based dataset

3. **Preprocessing**

   * Kalman Filter for noise smoothing

4. **Model Layer**

   * GRU + LSTM + Gradient Boosting

5. **Ensemble Layer**

   * Combined prediction

6. **Output**

   * Temperature prediction + thermal control

7. **Visualization**

   * Streamlit dashboard

---

## ⚙️ Tech Stack

### 🧑‍💻 Languages

* Python

### 📚 Libraries / Frameworks

* TensorFlow / Keras
* Scikit-learn
* Pandas, NumPy

### 🛠️ Tools / Platforms

* Docker
* iPerf
* Streamlit
* VS Code

---

## 📊 Dataset Details

* Total observations: **1,263**
* Session duration: **80.6 minutes**
* Sampling interval: **~4 seconds**
* Features: **9 → 5 selected**
* Temperature range: **53.02°C – 107.40°C**
* Mean temperature: **98.83°C**
* Sequence window: **20**
* Train/Test split: **80/20**

---

## 📈 Model Performance

| Model             | MAE    | RMSE   | MAPE   | R²     |
| ----------------- | ------ | ------ | ------ | ------ |
| Persistence       | 0.0599 | 0.0997 | 0.0573 | 0.9937 |
| GRU               | 0.2947 | 0.3683 | 0.2806 | 0.9131 |
| LSTM              | 0.5851 | 0.7314 | 0.5571 | 0.6574 |
| Gradient Boosting | 0.0715 | 0.1091 | 0.0680 | 0.9924 |
| Ensemble          | 0.2206 | 0.2757 | 0.2100 | 0.9513 |

💡 **Insight:**

* Gradient Boosting shows strong accuracy
* Ensemble model provides **balanced and stable predictions**

---

## 🔬 Methodology

1. Generate network traffic using iPerf (Docker)
2. Store data in CSV format
3. Apply Kalman Filter for preprocessing
4. Convert data into time-series sequences
5. Train GRU, LSTM, and Gradient Boosting models
6. Combine predictions using ensemble learning
7. Deploy for real-time inference

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/PES2UG23CS820SANTHOSH-A/<repo-name>.git
cd <repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 🖥️ Demo

* Real-time prediction dashboard (Streamlit)
* Displays temperature trends and model outputs

*(Add Loom / YouTube demo link here)*

---

## 🎯 Applications

* 5G/6G modem thermal management
* Edge computing systems
* IoT devices
* Data center monitoring
* Predictive maintenance

---

## 🚧 Challenges

* Noisy real-world data
* Model bias in time-series trends
* Real-time pipeline integration

---

## 🔮 Future Work

* Reinforcement learning for adaptive throttling
* Edge deployment optimization
* Hardware-level integration
* Advanced anomaly detection

---

## 👨‍💻 Author

**Santhosh A**
📧 [santhucaprcb@gmail.com](mailto:santhucaprcb@gmail.com)
🎓 PES University

---

## 🧠 Acknowledgment

Guided by **Dr. Mohammad Asif A Raibag**
Department of Computer Science & Engineering (AIML)
PES University

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
