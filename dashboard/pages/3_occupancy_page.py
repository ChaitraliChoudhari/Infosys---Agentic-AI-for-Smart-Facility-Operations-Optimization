import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from shared import (
    configure_page, sidebar_branding, get_api, post_api, delete_api,
    metric_card, alert_card, empty_state, section_title, BACKEND_URL
)

configure_page("Occupancy Agent")
sidebar_branding()

dashboard = get_api("/occupancy/dashboard")

if not dashboard:
    st.markdown('<div class="main-title">Occupancy Agent</div>', unsafe_allow_html=True)
    empty_state(
        "Could not load data from the backend. "
        f"Confirm FastAPI is running at {BACKEND_URL}, and that "
        "dataset/occupancy_zones.csv exists (run "
        "occupancy_agent/generate_occupancy_data.py if not)."
    )
    st.stop()

st.markdown('<div class="status-badge">● SYSTEM MONITORING</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="last-updated">Last updated: '
    f'{datetime.now().strftime("%d %b %Y, %I:%M:%S %p")}</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Occupancy Agent</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="main-subtitle">Real-time occupancy heatmaps, overcrowding '
    'alerts, space utilization, and ML-based occupancy forecasting.</div>',
    unsafe_allow_html=True
)

# ==================================================
# OVERVIEW KPIs
# ==================================================

section_title("Occupancy Overview")

k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card(
        "Current Occupants",
        f'{dashboard["Current Occupants"]} / {dashboard["Total Capacity"]}',
        f'{dashboard["Total Zones"]} zones across {dashboard["Total Floors"]} floors'
    )

with k2:
    metric_card(
        "Avg Occupancy",
        f'{dashboard["Avg Occupancy %"]:.1f}%',
        "Across all zones, current snapshot"
    )

with k3:
    metric_card(
        "High Alerts",
        f'{dashboard["High Alerts"]}',
        "Zones at/above 90% capacity"
    )

with k4:
    metric_card(
        "Medium Alerts",
        f'{dashboard["Medium Alerts"]}',
        "Zones approaching capacity (75-90%)"
    )

# ==================================================
# OCCUPANCY HEATMAP
# ==================================================

section_title("Occupancy Heatmap")

heatmap_data = get_api("/occupancy/heatmap")

if heatmap_data:
    heatmap_df = pd.DataFrame(heatmap_data)

    floors = sorted(heatmap_df["floor"].unique())
    floor_tabs = st.tabs([f"Floor {int(f)}" for f in floors])

    def heatmap_color(pct):
        if pct >= 90:
            return "#ff5a6a"   # critical / very high
        if pct >= 75:
            return "#ffb020"   # high
        if pct >= 40:
            return "#ffd93c"   # moderate
        return "#00e676"       # low

    for tab, floor in zip(floor_tabs, floors):
        with tab:
            floor_df = heatmap_df[heatmap_df["floor"] == floor].sort_values("zone")

            cols = st.columns(3)

            for index, (_, zone_row) in enumerate(floor_df.iterrows()):
                color = heatmap_color(zone_row["occupancy_pct"])

                with cols[index % 3]:
                    st.markdown(
                        f'<div class="metric-card" style="border-color:{color};min-height:100px;">'
                        f'<div class="metric-title">{zone_row["zone"]}</div>'
                        f'<div class="metric-value" style="color:{color};">'
                        f'{zone_row["occupancy_pct"]:.0f}%</div>'
                        f'<div class="metric-description">'
                        f'{int(zone_row["occupancy_count"])} / {int(zone_row["capacity"])} people</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
else:
    empty_state("No occupancy data available yet.")

# ==================================================
# LIVE DATA — ADD / REMOVE OCCUPANCY READINGS
# ==================================================

OCCUPANCY_ALERT_HIGH_THRESHOLD_UI = 90
OCCUPANCY_ALERT_MEDIUM_THRESHOLD_UI = 75

@st.fragment
def render_occupancy_live_data():
    zones = get_api("/occupancy/zones")
    zone_names = [z["zone"] for z in zones] if zones else []
    zone_capacity_lookup = {z["zone"]: z["capacity"] for z in zones} if zones else {}

    section_title("Add / Remove Live Reading")

    st.markdown('<div class="data-panel-title">Add New Reading</div>', unsafe_allow_html=True)

    # Zone picker lives outside the form so picking a zone immediately
    # updates the capacity/threshold hint below, instead of waiting for
    # the whole form to be submitted.
    selected_zone = st.selectbox("Zone", zone_names, key="new_occupancy_zone")

    if selected_zone and selected_zone in zone_capacity_lookup:
        capacity = zone_capacity_lookup[selected_zone]
        high_count = int(capacity * OCCUPANCY_ALERT_HIGH_THRESHOLD_UI / 100)
        medium_count = int(capacity * OCCUPANCY_ALERT_MEDIUM_THRESHOLD_UI / 100)

        st.caption(
            f"**{selected_zone}** capacity: **{capacity}**. "
            f"Enter **≥{high_count}** for a HIGH alert, "
            f"**{medium_count}-{high_count - 1}** for MEDIUM, "
            f"below **{medium_count}** for normal. "
            f"(You can enter above capacity too — that's a valid overcrowding reading.)"
        )

    # No date/time inputs here on purpose — every reading is stamped
    # with the real current time by the backend at the moment it's
    # submitted. This also matches what "live reading" should mean,
    # and it's what fixed the NaT-timestamp bug (a bad manually-entered
    # timestamp getting written to the CSV).
    with st.form("add_occupancy_form", clear_on_submit=True):
        st.number_input(
            "People Count", min_value=0, max_value=1000, value=0, step=1,
            key="new_occupancy_count"
        )

        submitted = st.form_submit_button("Add Reading")

        if submitted:
            zone = st.session_state.get("new_occupancy_zone")
            count = st.session_state.get("new_occupancy_count")

            result = post_api(
                "/occupancy/records",
                {
                    "zone": zone,
                    "occupancy_count": count
                }
            )

            if result is not None:
                now_label = datetime.now().strftime('%d %b %Y, %I:%M %p')
                st.session_state["_occ_flash"] = ("success", f"Reading added at {now_label}.")

    flash = st.session_state.pop("_occ_flash", None)
    if flash:
        kind, message = flash
        getattr(st, kind)(message)

    st.markdown(
        '<div class="data-panel-title">Recent Readings (click to delete)</div>',
        unsafe_allow_html=True
    )

    recent = get_api("/occupancy/records", params={"limit": 10})
    recent_df = pd.DataFrame(recent) if recent else pd.DataFrame()

    if not recent_df.empty:
        recent_df = recent_df.sort_values("occupancy_id", ascending=False)

        for _, row in recent_df.iterrows():
            is_high = row.get("occupancy_pct", 0) >= OCCUPANCY_ALERT_HIGH_THRESHOLD_UI
            row_class = "record-row failure" if is_high else "record-row"

            rc1, rc2, rc3, rc4 = st.columns([1.4, 1, 1, 0.6])

            with rc1:
                st.markdown(
                    f'<div class="{row_class}">{row.get("zone", "N/A")} (Floor {int(row.get("floor", 0))})</div>',
                    unsafe_allow_html=True
                )
            with rc2:
                st.markdown(
                    f'<div class="{row_class}">'
                    f'{int(row.get("occupancy_count", 0))} / {int(row.get("capacity", 0))}</div>',
                    unsafe_allow_html=True
                )
            with rc3:
                # Defensive: str() + explicit NaT/None check so a stray
                # bad row in the CSV can never crash this page or print
                # a raw "NaT" to the user again.
                ts_value = row.get("timestamp")
                if ts_value and str(ts_value) != "NaT":
                    ts_display = str(ts_value)[:16]
                else:
                    ts_display = "—"

                st.markdown(
                    f'<div class="{row_class}">{ts_display}</div>',
                    unsafe_allow_html=True
                )
            with rc4:
                record_id = row.get("occupancy_id", None)

                if record_id is not None and st.button("Delete", key=f"delete_occ_{record_id}"):
                    if delete_api(f"/occupancy/records/{int(record_id)}"):
                        st.rerun()
    else:
        empty_state("No live readings yet. Add one above to get started.")


render_occupancy_live_data()

# ==================================================
# OVERCROWDING ALERTS
# ==================================================

section_title("Overcrowding Alerts")

occupancy_alerts = get_api("/occupancy/alerts")

if occupancy_alerts:
    for alert in occupancy_alerts[:10]:
        alert_card(
            f'{alert["Severity"]} — {alert["Zone"]} (Floor {alert["Floor"]})',
            f'{alert["Message"]} Currently at {alert["Occupancy %"]:.0f}% capacity.'
        )
else:
    st.success("No overcrowding alerts — all zones within normal range.")

# ==================================================
# SPACE UTILIZATION ANALYSIS
# ==================================================

section_title("Space Utilization Analysis")

utilization = get_api("/occupancy/utilization")

if utilization:
    utilization_df = pd.DataFrame(utilization)

    color_map = {
        "Highly Utilized": "#00e676",
        "Moderately Utilized": "#ffb020",
        "Underutilized": "#ff5a6a"
    }

    fig_util = go.Figure()

    for classification, color in color_map.items():
        subset = utilization_df[utilization_df["Classification"] == classification]

        if not subset.empty:
            fig_util.add_trace(
                go.Bar(
                    x=subset["Zone"],
                    y=subset["Avg Occupancy % (Business Hours)"],
                    name=classification,
                    marker_color=color
                )
            )

    fig_util.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Zone",
        yaxis_title="Avg Occupancy % (Business Hours)",
        legend_title="Classification",
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    st.plotly_chart(fig_util, use_container_width=True)

    st.dataframe(utilization_df, use_container_width=True, hide_index=True)
else:
    empty_state("No utilization data available yet.")

# ==================================================
# ML OCCUPANCY FORECASTING
# ==================================================

section_title("Occupancy Forecast (ML)")

zones = get_api("/occupancy/zones")
zone_names = sorted({z["zone"] for z in zones}) if zones else []

if zone_names:
    forecast_col1, forecast_col2 = st.columns([1, 3])

    with forecast_col1:
        forecast_zone = st.selectbox("Zone", zone_names, key="forecast_zone_select")

        day_options = {
            "Tomorrow": None, "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        forecast_day_label = st.selectbox("Day", list(day_options.keys()), key="forecast_day_select")

    forecast_params = {"zone": forecast_zone}

    if day_options[forecast_day_label] is not None:
        forecast_params["day_of_week"] = day_options[forecast_day_label]

    forecast = get_api("/occupancy/forecast", params=forecast_params)

    with forecast_col2:
        if forecast and "hourly_forecast" in forecast:
            metric_card(
                f'{forecast["zone"]} — {forecast["day_name"]}',
                f'{forecast["predicted_avg_occupancy_pct_business_hours"]:.1f}%',
                "Predicted avg occupancy, business hours (9am-6pm)"
            )
        elif forecast and "error" in forecast:
            empty_state(forecast["error"])

    if forecast and "hourly_forecast" in forecast:
        hourly_df = pd.DataFrame(forecast["hourly_forecast"])

        fig_forecast = go.Figure()

        fig_forecast.add_trace(
            go.Scatter(
                x=hourly_df["hour"],
                y=hourly_df["predicted_occupancy_pct"],
                mode="lines+markers",
                line=dict(color="#20a4ff", width=3, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(32, 164, 255, 0.28)",
                hovertemplate="Hour: %{x}:00<br>Predicted: %{y:.1f}%<extra></extra>"
            )
        )

        fig_forecast.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=1, range=[0, 23]),
            yaxis=dict(title="Predicted Occupancy %", range=[0, 100], gridcolor="rgba(255,255,255,0.06)")
        )

        st.plotly_chart(fig_forecast, use_container_width=True)
else:
    empty_state("No zones available for forecasting yet.")

# ==================================================
# LIVE SENSOR-BASED OCCUPANCY ESTIMATION (real ML, real sensor data)
# ==================================================

section_title("Live Sensor Estimation (Reference Room)")

st.markdown(
    '<div class="main-subtitle" style="margin-bottom:16px;">'
    'This uses a model trained on real ambient sensor data (temperature, '
    'light, sound, CO2, motion) from one instrumented room — separate from '
    'the multi-zone capacity system above. Enter sensor values to see a '
    'live occupancy estimate.</div>',
    unsafe_allow_html=True
)

with st.form("sensor_estimate_form"):
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        temp = st.number_input("Temperature (°C)", value=25.0, step=0.1, key="sensor_temp")
        light = st.number_input("Light", value=100, step=1, key="sensor_light")

    with s2:
        sound = st.number_input("Sound", value=0.2, step=0.01, key="sensor_sound")
        co2 = st.number_input("CO2", value=400, step=1, key="sensor_co2")

    with s3:
        co2_slope = st.number_input("CO2 Slope", value=0.0, step=0.01, key="sensor_co2_slope")
        pir = st.selectbox("Motion Detected (PIR)", ["No", "Yes"], key="sensor_pir")

    with s4:
        st.write("")
        st.write("")
        estimate_submitted = st.form_submit_button("Estimate Occupancy")

    if estimate_submitted:
        pir_value = 1 if pir == "Yes" else 0

        sensor_payload = {
            "S1_Temp": temp, "S2_Temp": temp, "S3_Temp": temp, "S4_Temp": temp,
            "S1_Light": light, "S2_Light": light, "S3_Light": light, "S4_Light": light,
            "S1_Sound": sound, "S2_Sound": sound, "S3_Sound": sound, "S4_Sound": sound,
            "S5_CO2": co2, "S5_CO2_Slope": co2_slope,
            "S6_PIR": pir_value, "S7_PIR": pir_value
        }

        result = post_api("/occupancy/estimate-live", sensor_payload)

        if result and "predicted_occupancy_count" in result:
            st.success(f'Estimated occupancy: **{result["predicted_occupancy_count"]} people**')
        elif result and "error" in result:
            st.warning(result["error"])