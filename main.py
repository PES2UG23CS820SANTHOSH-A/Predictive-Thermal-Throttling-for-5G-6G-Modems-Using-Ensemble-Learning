import docker
import subprocess
import threading
import time
import pandas as pd
from datetime import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ------------------------
# GLOBAL VARIABLES
# ------------------------
client = docker.from_env()

data = []
ambient = 30
current_temp = 30

prev_total_bytes = 0
prev_cpu_total = 0
prev_system_cpu = 0

start_time = datetime.now()

# ------------------------
# TRAFFIC CONTROLLER
# ------------------------
def run_traffic(streams):
    subprocess.Popen([
        "docker", "run", "--rm",
        "networkstatic/iperf3",
        "-c", "host.docker.internal",
        "-P", str(streams),
        "-t", "15"
    ])


# ------------------------
# DATA COLLECTOR
# ------------------------
def collect_data():
    global current_temp, prev_total_bytes, prev_cpu_total, prev_system_cpu

    while True:
        try:
            container = client.containers.get("iperf-server")
            stats = container.stats(stream=False)

            # CPU calculation
            cpu_total = stats["cpu_stats"]["cpu_usage"]["total_usage"]
            system_cpu = stats["cpu_stats"]["system_cpu_usage"]

            cpu_delta = cpu_total - prev_cpu_total
            system_delta = system_cpu - prev_system_cpu

            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0

            prev_cpu_total = cpu_total
            prev_system_cpu = system_cpu

            # Network
            net = stats["networks"]
            total_bytes = sum(
                net[iface]["rx_bytes"] + net[iface]["tx_bytes"]
                for iface in net
            )

            byte_rate = total_bytes - prev_total_bytes
            prev_total_bytes = total_bytes

            throughput_mbps = (byte_rate * 8) / 1e6

            # Power model
            power = 0.5 * cpu_percent + 0.05 * throughput_mbps

            # Thermal model
            current_temp = (
                current_temp
                + 0.05 * power
                - 0.02 * (current_temp - ambient)
            )

            # Time calculations
            now = datetime.now()
            elapsed = (now - start_time).total_seconds()

            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            data.append({
                "timestamp": now.strftime("%H:%M:%S"),
                "elapsed_seconds": int(elapsed),
                "minute": minutes,
                "second": seconds,
                "cpu_percent": cpu_percent,
                "throughput_mbps": throughput_mbps,
                "byte_rate": byte_rate,
                "power": power,
                "temperature": current_temp
            })

            pd.DataFrame(data).to_csv("dataset.csv", index=False)

        except Exception as e:
            print("Error:", e)

        time.sleep(1)


# start background thread
threading.Thread(target=collect_data, daemon=True).start()


# ------------------------
# DASHBOARD
# ------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("5G Workload + Thermal Simulator"),

    html.Label("Simulated UE Streams"),
    dcc.Slider(id="stream-slider", min=1, max=20, step=1, value=5),

    html.Button("Start Traffic", id="start-btn"),

    dcc.Graph(id="live-graph"),

    dcc.Interval(id="interval", interval=2000)
])


@app.callback(
    Output("live-graph", "figure"),
    Input("interval", "n_intervals")
)
def update_graph(n):

    if len(data) < 2:
        return go.Figure()

    df = pd.DataFrame(data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["elapsed_seconds"],
        y=df["temperature"],
        mode="lines",
        name="Temperature"
    ))

    fig.add_trace(go.Scatter(
        x=df["elapsed_seconds"],
        y=df["throughput_mbps"],
        mode="lines",
        name="Throughput"
    ))

    fig.add_trace(go.Scatter(
        x=df["elapsed_seconds"],
        y=df["cpu_percent"],
        mode="lines",
        name="CPU"
    ))

    fig.update_layout(
        title="Live Workload & Thermal Monitoring",
        xaxis_title="Time (seconds)",
        yaxis_title="Value",
        template="plotly_dark"
    )

    return fig


@app.callback(
    Output("start-btn", "children"),
    Input("start-btn", "n_clicks"),
    Input("stream-slider", "value")
)
def start_traffic(n_clicks, streams):

    if n_clicks:
        threading.Thread(target=run_traffic, args=(streams,), daemon=True).start()
        return f"Traffic Running ({streams} streams)"

    return "Start Traffic"


if __name__ == "__main__":
    app.run(debug=True)