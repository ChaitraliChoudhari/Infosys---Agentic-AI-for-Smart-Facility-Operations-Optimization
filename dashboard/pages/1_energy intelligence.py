import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from shared import (
    configure_page, sidebar_branding, get_api, post_api, delete_api,
    metric_card, empty_state, section_title, alert_card, recommendation_card,
    BACKEND_URL
)

configure_page("Energy Intelligence")
sidebar_branding()

dashboard = get_api("/dashboard")
energy_data = get_api("/energy")

if dashboard is None or energy_data is None:
    st.markdown('<div class="main-title">Energy Intelligence</div>', unsafe_allow_html=True)
    empty_state(
        "Could not load data from the backend. "
        f"Confirm FastAPI is running at {BACKEND_URL}"
    )
    st.stop()

df = pd.DataFrame(energy_data)

if df.empty:
    st.error("The energy dataset is empty.")
    st.stop()

df["EnergyConsumption"] = pd.to_numeric(df["EnergyConsumption"], errors="coerce")

# A preliminary average so the "Recent Records" list further down
# (rendered before the simulated preview row is added) can flag
# high-consumption readings without a NameError.
average_energy = df["EnergyConsumption"].mean()

st.markdown('<div class="status-badge">● SYSTEM MONITORING</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="last-updated">Last updated: '
    f'{datetime.now().strftime("%d %b %Y, %I:%M:%S %p")}</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Energy Intelligence</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="main-subtitle">Monitor consumption patterns, identify '
    'anomalies and generate AI-driven energy recommendations.</div>',
    unsafe_allow_html=True
)

# ==================================================
# ENERGY OVERVIEW (KPI cards — shown above Add/Remove)
# ==================================================

section_title("Energy Overview")

preview_maximum_energy = df["EnergyConsumption"].max()
preview_minimum_energy = df["EnergyConsumption"].min()
preview_high_alert_count = int((df["EnergyConsumption"] > average_energy * 1.2).sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("Average Consumption", f"{average_energy:.2f}", "Average energy usage")

with c2:
    metric_card("Peak Consumption", f"{preview_maximum_energy:.2f}", "Highest recorded value")

with c3:
    metric_card("Minimum Consumption", f"{preview_minimum_energy:.2f}", "Lowest recorded value")

with c4:
    metric_card("High Energy Alerts", f"{preview_high_alert_count}", "Above 120% of average")


# ==================================================
# LIVE DATA — ADD / REMOVE ENERGY RECORDS
# ==================================================

@st.fragment
def render_energy_live_data(average_energy):
    """
    Wrapped in a fragment: typing in these inputs or hitting
    Add/Delete only reruns this function, not the whole page
    (charts below stay untouched until you actually add/delete).
    """
    section_title("Add / Remove Live Data")

    st.markdown(
        '<div class="data-panel-title">Enter Reading — updates the whole dashboard</div>',
        unsafe_allow_html=True
    )

    e1, e2, e3, e4, e5, e6 = st.columns([0.8, 1, 1, 1, 0.9, 0.9])

    with e1:
        hour_text = st.text_input("Hour (0-23)", value="12", key="record_hour_text")
    with e2:
        temperature_text = st.text_input("Outdoor Temp (°C)", value="24.0", key="record_temperature_text")
    with e3:
        occupancy_text = st.text_input("Occupancy", value="25", key="record_occupancy_text")
    with e4:
        humidity_text = st.text_input("Humidity (%)", value="45.0", key="record_humidity_text")
    with e5:
        record_weekend = st.selectbox("Weekend", ["No", "Yes"], key="record_weekend")
    with e6:
        st.write("")
        st.write("")
        add_energy_submitted = st.button("Add Record", key="add_energy_record_button")

    parse_errors = []

    try:
        record_hour = int(float(hour_text))
        if not (0 <= record_hour <= 23):
            raise ValueError
    except (TypeError, ValueError):
        record_hour = 12
        parse_errors.append("Hour must be a whole number from 0-23.")

    try:
        record_temperature = float(temperature_text)
    except (TypeError, ValueError):
        record_temperature = 24.0
        parse_errors.append("Outdoor Temperature must be a number.")

    try:
        record_occupancy = int(float(occupancy_text))
        if record_occupancy < 0:
            raise ValueError
    except (TypeError, ValueError):
        record_occupancy = 25
        parse_errors.append("Occupancy must be a whole number ≥ 0.")

    try:
        record_humidity = float(humidity_text)
    except (TypeError, ValueError):
        record_humidity = 45.0
        parse_errors.append("Humidity must be a number.")

    if parse_errors:
        st.warning(" ".join(parse_errors))

    if add_energy_submitted and not parse_errors:
        new_energy_record = {
            "hour": record_hour,
            "outdoor_temperature": record_temperature,
            "occupancy": record_occupancy,
            "humidity": record_humidity,
            "weekend": record_weekend
        }

        energy_record_result = post_api("/energy-records", new_energy_record)

        if energy_record_result is not None:
            st.success("Record added — dashboard updated with the new reading.")
            st.rerun()

    st.markdown(
        '<div class="data-panel-title">Recent Records (click to delete)</div>',
        unsafe_allow_html=True
    )

    if not df.empty:
        recent_energy_records = df.tail(10).iloc[::-1]

        for _, energy_row in recent_energy_records.iterrows():
            energy_value = energy_row.get("EnergyConsumption", "-")

            is_high = (
                isinstance(energy_value, (int, float))
                and energy_value > average_energy * 1.2
            )

            energy_row_class = "record-row failure" if is_high else "record-row"

            ec1, ec2, ec3, ec4, ec5 = st.columns([1.3, 1, 1, 1, 0.6])

            with ec1:
                st.markdown(
                    f'<div class="{energy_row_class}">{energy_row.get("Timestamp", "N/A")}</div>',
                    unsafe_allow_html=True
                )
            with ec2:
                st.markdown(
                    f'<div class="{energy_row_class}">Temp: {energy_row.get("Temperature", "-")}</div>',
                    unsafe_allow_html=True
                )
            with ec3:
                st.markdown(
                    f'<div class="{energy_row_class}">Occ: {energy_row.get("Occupancy", "-")}</div>',
                    unsafe_allow_html=True
                )
            with ec4:
                st.markdown(
                    f'<div class="{energy_row_class}">Energy: {energy_value}</div>',
                    unsafe_allow_html=True
                )
            with ec5:
                energy_record_id = energy_row.get("id", None)
 
                # Guard against NaN ids (e.g. a stray simulated/preview
                # row with no real backend id) so we never crash on
                # int(NaN) — we simply skip rendering a Delete button
                # for that row instead.
                valid_id = energy_record_id is not None and not (
                    isinstance(energy_record_id, float) and pd.isna(energy_record_id)
                )
 
                if valid_id and st.button(
                    "Delete", key=f"delete_energy_{int(energy_record_id)}"
                ):
                    if delete_api(f"/energy-records/{int(energy_record_id)}"):
                        st.success("Record deleted.")
                        st.rerun()
    else:
        empty_state("No live records yet. Add one above to get started.")

    return record_hour, record_temperature, record_occupancy, record_weekend


record_hour, record_temperature, record_occupancy, record_weekend = render_energy_live_data(average_energy)

# ==================================================
# SIMULATED PREVIEW POINT (uses the values above)
# ==================================================

simulation = get_api(
    "/energy/simulate",
    params={
        "hour": record_hour,
        "outdoor_temperature": record_temperature,
        "occupancy": record_occupancy,
        "weekend": record_weekend
    }
)
 
if simulation is None:
    st.stop()
 
simulated_df = pd.DataFrame([simulation])
 
# NOTE: everything from here on uses `plot_df`, a separate variable
# from `df`. Do NOT reassign `df` itself — the render_energy_live_data
# fragment above closes over the module-level `df`, and on a
# fragment-only rerun (e.g. clicking Delete) this section does not
# re-execute, so `df` must stay exactly what came back from /energy.
plot_df = df.copy()
plot_df["Simulated"] = "Historical"
plot_df = pd.concat([plot_df, simulated_df], ignore_index=True)
plot_df["Simulated"] = plot_df["Simulated"].fillna("Simulated")
 
plot_df["EnergyConsumption"] = pd.to_numeric(plot_df["EnergyConsumption"], errors="coerce")
plot_df["Temperature"] = pd.to_numeric(plot_df["Temperature"], errors="coerce")
plot_df["Occupancy"] = pd.to_numeric(plot_df["Occupancy"], errors="coerce")
plot_df["Timestamp"] = pd.to_datetime(plot_df["Timestamp"], errors="coerce")
 
plot_df = plot_df.dropna(subset=["EnergyConsumption"])
 
average_energy = plot_df["EnergyConsumption"].mean()
maximum_energy = plot_df["EnergyConsumption"].max()
minimum_energy = plot_df["EnergyConsumption"].min()
high_alert_count = int((plot_df["EnergyConsumption"] > average_energy * 1.2).sum())

# ==================================================
# CONSUMPTION TREND AREA CHART
# ==================================================

section_title("Consumption Trend")

if "Hour" not in df.columns:
    df["Hour"] = df["Timestamp"].dt.hour
else:
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")

hourly_average = (
    df.dropna(subset=["Hour", "EnergyConsumption"])
    .groupby("Hour", as_index=False)["EnergyConsumption"]
    .mean()
    .sort_values("Hour")
)

hourly_average["Hour"] = hourly_average["Hour"].astype(int)

fig_trend = go.Figure()

fig_trend.add_trace(
    go.Scatter(
        x=hourly_average["Hour"],
        y=hourly_average["EnergyConsumption"],
        mode="lines+markers",
        name="Average Consumption",
        line=dict(color="#20a4ff", width=3, shape="spline"),
        marker=dict(size=8, color="#20a4ff", line=dict(color="#ffffff", width=1)),
        fill="tozeroy",
        fillcolor="rgba(32, 164, 255, 0.28)",
        hovertemplate="Hour: %{x}:00<br>Average Consumption: %{y:.2f} kWh<extra></extra>"
    )
)

fig_trend.add_hline(
    y=average_energy,
    line_dash="dash",
    line_color="#ff9d3c",
    line_width=2,
    annotation_text="Overall average",
    annotation_position="top right",
    annotation_font_color="#ff9d3c"
)

fig_trend.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=400,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=1,
               range=[0, 23], showgrid=False, zeroline=False),
    yaxis=dict(title="Average Energy Consumption", rangemode="tozero",
               showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified"
)

st.plotly_chart(fig_trend, use_container_width=True)

# ==================================================
# TEMPERATURE VS ENERGY
# ==================================================

section_title("Temperature vs Energy")

temperature_columns = ["Temperature", "EnergyConsumption", "Simulated"]
temperature_df = df.dropna(subset=[c for c in temperature_columns if c in df.columns])

if not temperature_df.empty:
    temperature_hover = [c for c in ["Timestamp", "Occupancy", "Hour", "Weekend"] if c in temperature_df.columns]

    fig_temp = px.scatter(
        temperature_df, x="Temperature", y="EnergyConsumption", color="Simulated",
        hover_data=temperature_hover,
        color_discrete_map={"Historical": "#20a4ff", "Simulated": "#ff9d3c"}
    )

    fig_temp.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0)))

    fig_temp.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Outdoor Temperature (°C)", yaxis_title="Energy Consumption",
        legend_title="Record Type",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    st.plotly_chart(fig_temp, use_container_width=True)
else:
    empty_state("Temperature data is unavailable.")

# ==================================================
# OCCUPANCY VS ENERGY
# ==================================================

section_title("Occupancy vs Energy")

occupancy_columns = ["Occupancy", "EnergyConsumption", "Simulated"]
occupancy_df = df.dropna(subset=[c for c in occupancy_columns if c in df.columns])

if not occupancy_df.empty:
    occupancy_hover = [c for c in ["Timestamp", "Temperature", "Hour", "Weekend"] if c in occupancy_df.columns]

    fig_occ = px.scatter(
        occupancy_df, x="Occupancy", y="EnergyConsumption", color="Simulated",
        hover_data=occupancy_hover,
        color_discrete_map={"Historical": "#20a4ff", "Simulated": "#ff9d3c"}
    )

    fig_occ.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0)))

    fig_occ.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Occupancy", yaxis_title="Energy Consumption",
        legend_title="Record Type",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    st.plotly_chart(fig_occ, use_container_width=True)
else:
    empty_state("Occupancy data is unavailable.")

# ==================================================
# HIGH ENERGY ALERTS
# ==================================================

section_title("High Energy Alerts")

high_energy_df = df[df["EnergyConsumption"] > average_energy * 1.2].copy()

if not high_energy_df.empty:
    alert_columns = [
        "Timestamp", "Hour", "Temperature", "Occupancy", "Weekend", "Humidity",
        "SquareFootage", "HVACUsage", "LightingUsage", "RenewableEnergy",
        "DayOfWeek", "Holiday", "EnergyConsumption", "Simulated"
    ]
    alert_columns = [c for c in alert_columns if c in high_energy_df.columns]

    st.dataframe(high_energy_df[alert_columns], use_container_width=True, hide_index=True)
else:
    st.success("No high energy consumption alerts detected.")

# ==================================================
# ALERTS AND RECOMMENDATIONS
# ==================================================

col1, col2 = st.columns(2)

with col1:
    section_title("Energy Alerts")

    alerts = get_api("/alerts")

    if alerts:
        for alert in alerts[:8]:
            alert_card("High Energy Consumption", f'Timestamp: {alert.get("Timestamp", "N/A")}')
    else:
        st.success("No high-energy alerts detected.")

with col2:
    section_title("AI Recommendations")

    recommendations = get_api("/recommendations")

    if recommendations:
        for item in recommendations[:8]:
            recommendation_card("AI Recommendation", item.get("Recommendations", []))
    else:
        st.info("No recommendations currently available.")