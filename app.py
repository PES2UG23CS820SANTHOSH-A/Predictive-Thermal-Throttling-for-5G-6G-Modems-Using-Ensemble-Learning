import streamlit as st
import pandas as pd
import docker
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import joblib, pickle

from preprocess import preprocess

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Thermal System", layout="wide", page_icon="🔥")

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f1a; }
    [data-testid="stAppViewContainer"] { background-color: #0f0f1a; }
    [data-testid="stHeader"] { background-color: #0f0f1a; }
    [data-testid="stSidebar"] { background-color: #16213e; }

    /* Remove default padding */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Title */
    h1 { color: #e0e0e0 !important; font-size: 1.2rem !important; font-weight: 500 !important;
         border-bottom: 0.5px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 16px !important; }
    h2, h3 { color: #aaaaaa !important; font-size: 0.85rem !important; font-weight: 400 !important;
              letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 8px !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #16213e;
        border: 0.5px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #888888 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #e0e0e0 !important; font-size: 1.4rem !important; font-weight: 500 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* Status badges */
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 4px;
        font-size: 11px; font-weight: 500; margin-right: 6px;
    }
    .badge-ok   { background: rgba(74,222,128,0.15); color: #4ade80; }
    .badge-warn { background: rgba(251,191,36,0.15);  color: #fbbf24; }
    .badge-hot  { background: rgba(248,113,113,0.15); color: #f87171; }

    /* Gauge container */
    .gauge-box {
        background: #16213e;
        border: 0.5px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .gauge-label { color: #888; font-size: 11px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .gauge-value { font-size: 26px; font-weight: 500; margin-top: -8px; }
</style>
""", unsafe_allow_html=True)

st.title("🔥  Thermal System — Live + Model Monitoring")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("ensemble_model.pkl")
    except:
        try:
            with open("ensemble_model.pkl", "rb") as f:
                return pickle.load(f)
        except:
            return None

model = load_model()

# ---------------- DOCKER ----------------
@st.cache_resource
def get_client():
    return docker.from_env()

client = get_client()

# ---------------- SESSION ----------------
if "data" not in st.session_state:
    st.session_state.data = []
    st.session_state.prev_bytes = None
    st.session_state.prev_cpu = None
    st.session_state.prev_system = None
    st.session_state.temp = 30
    st.session_state.future_temp = 33
    st.session_state.start_time = datetime.now()

# ---------------- DATA ----------------
def get_data():
    try:
        container = client.containers.get("iperf-server")
        stats = container.stats(stream=False)
    except Exception as e:
        st.error(f"❌ Docker Error: {e}")
        return None

    # CPU
    cpu_total = stats["cpu_stats"]["cpu_usage"]["total_usage"]
    system_cpu = stats["cpu_stats"].get("system_cpu_usage", 0)

    if st.session_state.prev_cpu is None:
        st.session_state.prev_cpu = cpu_total
        st.session_state.prev_system = system_cpu

    cpu_delta = cpu_total - st.session_state.prev_cpu
    system_delta = system_cpu - st.session_state.prev_system
    cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0

    st.session_state.prev_cpu = cpu_total
    st.session_state.prev_system = system_cpu

    # NETWORK
    net = stats.get("networks", {})
    total_bytes = sum(v["rx_bytes"] + v["tx_bytes"] for v in net.values())

    if st.session_state.prev_bytes is None:
        st.session_state.prev_bytes = total_bytes

    byte_rate = total_bytes - st.session_state.prev_bytes
    st.session_state.prev_bytes = total_bytes
    throughput = (byte_rate * 8) / 1e6

    # POWER
    power = 0.5 * cpu_percent + 0.05 * throughput

    # LIVE TEMP (PHYSICS MODEL)
    ambient = 30
    temp = st.session_state.temp + (0.04 * power - 0.025 * (st.session_state.temp - ambient))
    temp = max(25, min(temp, 100))
    st.session_state.temp = temp

    # FUTURE TEMP
    future = st.session_state.future_temp
    model_pred = temp
    if model is not None:
        try:
            input_df = pd.DataFrame([{
                "cpu_percent": cpu_percent,
                "throughput_mbps": throughput,
                "byte_rate": byte_rate,
                "power": power
            }])
            processed = preprocess(input_df)
            model_pred = model.predict(processed)[0]
        except:
            pass

    trend = temp - future
    gap = 5 if throughput > 600 else 4 if throughput > 400 else 3 if throughput > 200 else 2

    future = 0.7 * future + 0.3 * model_pred
    future += 0.2 * trend

    if throughput < 150:
        future -= 4
    elif throughput < 300:
        future -= 2

    if trend > 0:
        future = max(temp + gap, future)

    future = max(25, min(future, 120))
    st.session_state.future_temp = future

    elapsed = int((datetime.now() - st.session_state.start_time).total_seconds())

    return {
        "elapsed": elapsed,
        "cpu": cpu_percent,
        "throughput": throughput,
        "power": power,
        "temp": temp,
        "future_temp": future
    }

# ---------------- FETCH ----------------
data = get_data()
if data:
    st.session_state.data.append(data)

st.session_state.data = st.session_state.data[-120:]
df = pd.DataFrame(st.session_state.data)

# ---------------- HELPER: GAUGE FIGURE ----------------
def make_gauge(value, title, min_val, max_val, color, suffix="°C"):
    pct = (value - min_val) / (max_val - min_val)
    # color thresholds
    if suffix == "°C":
        bar_color = "#4ade80" if value < 50 else "#fbbf24" if value < 70 else "#f87171"
    else:
        bar_color = color

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 2),
        number={"suffix": f" {suffix}", "font": {"size": 28, "color": bar_color}},
        gauge={
            "axis": {
                "range": [min_val, max_val],
                "tickcolor": "#555",
                "tickfont": {"size": 9, "color": "#666"},
                "nticks": 6,
            },
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#16213e",
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [min_val, min_val + (max_val - min_val) * 0.5], "color": "rgba(74,222,128,0.07)"},
                {"range": [min_val + (max_val - min_val) * 0.5, min_val + (max_val - min_val) * 0.75], "color": "rgba(251,191,36,0.07)"},
                {"range": [min_val + (max_val - min_val) * 0.75, max_val], "color": "rgba(248,113,113,0.07)"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
        title={"text": title, "font": {"size": 11, "color": "#888"}},
    ))
    fig.update_layout(
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        font_color="#e0e0e0",
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig

# ---------------- HELPER: LINE CHART ----------------
# Base axis styles (used to build per-chart axis dicts)
AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)")

CHART_LAYOUT = dict(
    paper_bgcolor="#16213e",
    plot_bgcolor="#16213e",
    font=dict(color="#888", size=10),
    margin=dict(l=40, r=10, t=10, b=30),
    legend=dict(orientation="h", y=-0.25, font=dict(size=10, color="#aaa"), bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
)

# ---------------- ROW 1: GAUGES ----------------
if not df.empty:
    latest = df.iloc[-1]

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            make_gauge(latest["temp"], "Live Temperature", 25, 100, "#4ade80", "°C"),
            use_container_width=True, config={"displayModeBar": False}
        )
    with g2:
        fc = "#fb923c" if latest["future_temp"] < 80 else "#f87171"
        st.plotly_chart(
            make_gauge(latest["future_temp"], "Future Temperature", 25, 120, fc, "°C"),
            use_container_width=True, config={"displayModeBar": False}
        )

    # ---------------- ROW 2: METRIC CARDS ----------------
    st.markdown("##### Live Metrics")
    m1, m2, m3, m4 = st.columns(4)

    prev = df.iloc[-2] if len(df) > 1 else latest
    m1.metric("🌡 Live Temp",     f"{latest['temp']:.2f} °C",      f"{latest['temp'] - prev['temp']:+.2f}")
    m2.metric("🔮 Future Temp",   f"{latest['future_temp']:.2f} °C", f"{latest['future_temp'] - prev['future_temp']:+.2f}")
    m3.metric("🧠 CPU",           f"{latest['cpu']:.1f} %",         f"{latest['cpu'] - prev['cpu']:+.1f}")
    m4.metric("📡 Throughput",    f"{latest['throughput']:.1f} Mbps", f"{latest['throughput'] - prev['throughput']:+.1f}")

    # ---------------- STATUS BADGES ----------------
    cpu_v = latest["cpu"]
    temp_v = latest["temp"]
    cpu_cls = "badge-ok" if cpu_v < 50 else "badge-warn" if cpu_v < 80 else "badge-hot"
    cpu_lbl = "CPU Normal" if cpu_v < 50 else "CPU Moderate" if cpu_v < 80 else "CPU Critical"
    tmp_cls = "badge-ok" if temp_v < 50 else "badge-warn" if temp_v < 70 else "badge-hot"
    tmp_lbl = "Temp Normal" if temp_v < 50 else "Temp Elevated" if temp_v < 70 else "Temp Critical"

    st.markdown(
        f'<div style="margin:8px 0 16px">'
        f'<span class="badge badge-ok">System Online</span>'
        f'<span class="badge {cpu_cls}">{cpu_lbl}</span>'
        f'<span class="badge {tmp_cls}">{tmp_lbl}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ---------------- ROW 3: TEMP vs TIME (full width) ----------------
    st.markdown("##### Temperature vs Time")

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df["elapsed"], y=df["temp"],
        name="Live Temp", line=dict(color="#4ade80", width=2),
        fill="tozeroy", fillcolor="rgba(74,222,128,0.07)"
    ))
    fig_temp.add_trace(go.Scatter(
        x=df["elapsed"], y=df["future_temp"],
        name="Future Temp", line=dict(color="#f87171", width=2),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.05)"
    ))
    fig_temp.update_layout(**CHART_LAYOUT, height=220,
        xaxis=dict(**AXIS_STYLE, title="Elapsed (s)", tickfont=dict(size=9)),
        yaxis=dict(**AXIS_STYLE, range=[25, 120], title="°C", tickfont=dict(size=9)))
    st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})

    # ---------------- ROW 4: THREE SMALL CHARTS ----------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### CPU & Temp Correlation")
        fig_cpu = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cpu.add_trace(go.Scatter(x=df["elapsed"], y=df["cpu"],
            name="CPU %", line=dict(color="#fbbf24", width=1.5), mode="lines"), secondary_y=False)
        fig_cpu.add_trace(go.Scatter(x=df["elapsed"], y=df["temp"],
            name="Temp", line=dict(color="#f87171", width=1.5), mode="lines"), secondary_y=True)
        fig_cpu.update_layout(**CHART_LAYOUT, height=180,
            xaxis=dict(**AXIS_STYLE, tickfont=dict(size=9)),
            yaxis=dict(**AXIS_STYLE, title="CPU %", tickfont=dict(color="#fbbf24", size=9)),
            yaxis2=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#f87171", size=9), title="°C"))
        st.plotly_chart(fig_cpu, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("##### Throughput Trend")
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(x=df["elapsed"], y=df["throughput"],
            name="Mbps", line=dict(color="#60a5fa", width=1.5),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"))
        fig_tp.update_layout(**CHART_LAYOUT, height=180,
            xaxis=dict(**AXIS_STYLE, tickfont=dict(size=9)),
            yaxis=dict(**AXIS_STYLE, title="Mbps", tickfont=dict(size=9)))
        st.plotly_chart(fig_tp, use_container_width=True, config={"displayModeBar": False})

    with c3:
        st.markdown("##### Power Draw")
        fig_pwr = go.Figure()
        fig_pwr.add_trace(go.Scatter(x=df["elapsed"], y=df["power"],
            name="Power", line=dict(color="#a78bfa", width=1.5),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"))
        fig_pwr.update_layout(**CHART_LAYOUT, height=180,
            xaxis=dict(**AXIS_STYLE, tickfont=dict(size=9)),
            yaxis=dict(**AXIS_STYLE, title="Watts", tickfont=dict(size=9)))
        st.plotly_chart(fig_pwr, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("⏳ Waiting for first data point...")

# ---------------- REFRESH ----------------
time.sleep(10)
st.rerun()