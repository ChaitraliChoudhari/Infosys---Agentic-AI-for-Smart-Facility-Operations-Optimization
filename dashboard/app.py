import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


# ==========================================================
# CONFIGURATION
# ==========================================================

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Agentic FacilityOps AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.logo {
    font-size: 26px;
    font-weight: 800;
    color: white;
    letter-spacing: -0.5px;
}

.logo-blue {
    color: #20a4ff;
}

.sidebar-subtitle {
    color: #8fa3bf;
    font-size: 13px;
    margin-bottom: 30px;
    line-height: 1.4;
}

.sidebar-heading {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #5c7291;
    margin-bottom: 10px;
}

.status-badge {
    display: inline-block;
    padding: 7px 16px;
    border: 1px solid #00c853;
    border-radius: 20px;
    color: #00e676;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
    background: rgba(0, 200, 83, 0.08);
}

.last-updated {
    color: #5c7291;
    font-size: 12.5px;
    margin-bottom: 18px;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.main-subtitle {
    font-size: 16px;
    color: #8fa3bf;
    margin-bottom: 36px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: white;
    margin-top: 28px;
    margin-bottom: 16px;
}

.metric-card {
    padding: 20px 22px;
    border-radius: 14px;
    border: 1px solid #263a56;
    background: linear-gradient(
        180deg,
        #131f33 0%,
        #0f1a2b 100%
    );
    min-height: 128px;
}

.metric-card:hover {
    border-color: #3477c9;
}

.metric-title {
    color: #76a9e8;
    font-size: 13.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
    line-height: 1.1;
}

.metric-description {
    color: #7d8fa8;
    font-size: 13px;
    margin-top: 6px;
}

.sidebar-card {
    padding: 14px 16px;
    margin-top: 12px;
    border-radius: 12px;
    border: 1px solid #263a56;
    background: #111c2d;
}

.sidebar-card-title {
    color: #20a4ff;
    font-weight: 700;
    font-size: 11.5px;
    letter-spacing: 0.5px;
}

.sidebar-card-text {
    color: white;
    font-size: 14px;
    margin-top: 6px;
}

.status-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    background: #00d639;
    border-radius: 50%;
    margin-right: 7px;
    box-shadow: 0 0 6px #00d639;
}

.alert-card {
    padding: 14px 16px;
    margin-bottom: 10px;
    border-radius: 12px;
    border: 1px solid #7d3040;
    background: #24151c;
}

.alert-title {
    color: #ff6575;
    font-weight: 700;
    font-size: 15px;
}

.alert-text {
    color: #c5ccd6;
    font-size: 13px;
    margin-top: 6px;
}

.recommendation-card {
    padding: 14px 16px;
    margin-bottom: 10px;
    border-radius: 12px;
    border: 1px solid #31547d;
    background: #111f31;
}

.recommendation-title {
    color: #20a4ff;
    font-weight: 700;
    font-size: 15px;
}

.recommendation-text {
    color: #c5ccd6;
    font-size: 13px;
    margin-top: 6px;
    line-height: 1.6;
}

.healthy-card,
.warning-card,
.critical-card {
    padding: 20px 22px;
    border-radius: 14px;
    min-height: 118px;
}

.healthy-card {
    background: linear-gradient(
        180deg,
        #14301f 0%,
        #0f2418 100%
    );
    border: 1px solid #1e7545;
}

.warning-card {
    background: linear-gradient(
        180deg,
        #2e2814 0%,
        #241f0e 100%
    );
    border: 1px solid #856d22;
}

.critical-card {
    background: linear-gradient(
        180deg,
        #301820 0%,
        #241118 100%
    );
    border: 1px solid #873344;
}

.maintenance-label {
    color: #9ba9ba;
    font-size: 12.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.maintenance-number {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.empty-state {
    padding: 24px;
    border-radius: 12px;
    border: 1px dashed #2a3d5a;
    color: #7d8fa8;
    text-align: center;
    font-size: 14px;
}

.stat-panel {
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid #263a56;
    background: linear-gradient(
        180deg,
        #131f33 0%,
        #0f1a2b 100%
    );
    height: 100%;
}

.stat-panel-title {
    color: #5c7291;
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 10px;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    margin-top: 8px;
}

.stat-block-label {
    color: #6d84a3;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.stat-block-value {
    color: white;
    font-size: 24px;
    font-weight: 800;
    margin-top: 4px;
}

.stat-block-value.critical {
    color: #ff5a6a;
}

.stat-block-value.good {
    color: #00e676;
}

.live-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #1e7545;
    background: rgba(0, 200, 83, 0.1);
    color: #00e676;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

.data-panel {
    padding: 22px 24px;
    border-radius: 14px;
    border: 1px solid #263a56;
    background: #0f1a2b;
    margin-bottom: 18px;
}

.data-panel-title {
    color: #5c7291;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 14px;
}

.record-row {
    padding: 10px 14px;
    border-radius: 10px;
    border: 1px solid #22314a;
    background: #111c2d;
    margin-bottom: 8px;
}

.record-row.failure {
    border-color: #7d3040;
    background: #1e131a;
}

div[data-testid="stForm"] {
    border: none;
    padding: 0;
}

.stButton > button {
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True
)


# ==========================================================
# HTML HELPERS
# ==========================================================

def metric_card(title, value, description):
    html = (
        '<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-description">{description}</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def health_card(css_class, label, number):
    html = (
        f'<div class="{css_class}">'
        f'<div class="maintenance-label">{label}</div>'
        f'<div class="maintenance-number">{number}</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def alert_card(title, text):
    html = (
        '<div class="alert-card">'
        f'<div class="alert-title">⚠ {title}</div>'
        f'<div class="alert-text">{text}</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def recommendation_card(title, items):
    if isinstance(items, list):
        if items:
            recommendation_text = "<br>".join(
                f"• {item}"
                for item in items
            )
        else:
            recommendation_text = "No recommendation."
    else:
        recommendation_text = str(items)

    html = (
        '<div class="recommendation-card">'
        f'<div class="recommendation-title">🤖 {title}</div>'
        f'<div class="recommendation-text">'
        f'{recommendation_text}'
        '</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def empty_state(message):
    st.markdown(
        f'<div class="empty-state">{message}</div>',
        unsafe_allow_html=True
    )


def section_title(text):
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True
    )


def fleet_health_gauge(score):
    if score >= 75:
        bar_color = "#00e676"
    elif score >= 50:
        bar_color = "#ffb020"
    else:
        bar_color = "#ff5a6a"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(
                suffix="",
                font=dict(color="white", size=34)
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor="#5c7291",
                    tickfont=dict(color="#5c7291", size=10)
                ),
                bar=dict(color=bar_color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color="rgba(255,90,106,0.12)"),
                    dict(range=[50, 75], color="rgba(255,176,32,0.12)"),
                    dict(range=[75, 100], color="rgba(0,230,118,0.12)")
                ]
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=190,
        margin=dict(l=20, r=20, t=10, b=0)
    )

    return fig


def stat_panel(title, blocks):
    rows_html = '<div class="stat-row">'

    for label, value, css in blocks:
        value_class = f"stat-block-value {css}".strip()
        rows_html += (
            '<div>'
            f'<div class="stat-block-label">{label}</div>'
            f'<div class="{value_class}">{value}</div>'
            '</div>'
        )

    rows_html += '</div>'

    html = (
        '<div class="stat-panel">'
        f'<div class="stat-panel-title">{title}</div>'
        f'{rows_html}'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# ==========================================================
# API HELPER
# ==========================================================

def get_api(endpoint, params=None):
    try:
        response = requests.get(
            f"{BACKEND_URL}{endpoint}",
            params=params,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as error:
        st.error(
            f"FastAPI connection error on "
            f"{endpoint}: {error}"
        )
        return None


def post_api(endpoint, payload):
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as error:
        st.error(
            f"FastAPI connection error on "
            f"{endpoint}: {error}"
        )
        return None


def delete_api(endpoint):
    try:
        response = requests.delete(
            f"{BACKEND_URL}{endpoint}",
            timeout=30
        )

        response.raise_for_status()

        return True

    except requests.exceptions.RequestException as error:
        st.error(
            f"FastAPI connection error on "
            f"{endpoint}: {error}"
        )
        return False


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown(
        '<div class="logo">'
        ' Agentic '
        '<span class="logo-blue">FacilityOps</span>'
        '</div>'
        '<div class="sidebar-subtitle">'
        'AI-powered facility monitoring and intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Operations</div>',
        unsafe_allow_html=True
    )

    module = st.radio(
        "Module",
        [
            " Energy Intelligence",
            " Predictive Maintenance"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")


    st.markdown(
        '<div class="sidebar-card">'
        '<div class="sidebar-card-title">'
        'SYSTEM STATUS'
        '</div>'
        '<div class="sidebar-card-text">'
        '<span class="status-dot"></span>'
        'Monitoring Active'
        '</div>'
        '</div>'
        '<div class="sidebar-card">'
        '<div class="sidebar-card-title">'
        'BACKEND'
        '</div>'
        '<div class="sidebar-card-text">'
        'FastAPI'
        '</div>'
        '</div>'
        '<div class="sidebar-card">'
        '<div class="sidebar-card-title">'
        'PLATFORM'
        '</div>'
        '<div class="sidebar-card-text">'
        'Agentic FacilityOps AI'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================================
# ENERGY INTELLIGENCE
# ==========================================================

if module == " Energy Intelligence":
    dashboard = get_api("/dashboard")
    energy_data = get_api("/energy")

    if dashboard is not None and energy_data is not None:
        df = pd.DataFrame(energy_data)

        if df.empty:
            st.error("The energy dataset is empty.")
            st.stop()

        df["EnergyConsumption"] = pd.to_numeric(
            df["EnergyConsumption"],
            errors="coerce"
        )

        # A preliminary average so the "Recent Records" list further
        # down (rendered before the simulated preview row is added)
        # can flag high-consumption readings without a NameError.
        average_energy = df["EnergyConsumption"].mean()

        st.markdown(
            '<div class="status-badge">'
            '● SYSTEM MONITORING'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="last-updated">'
            f'Last updated: '
            f'{datetime.now().strftime("%d %b %Y, %I:%M:%S %p")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="main-title">'
            'Energy Intelligence'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="main-subtitle">'
            'Monitor consumption patterns, identify '
            'anomalies and generate AI-driven energy '
            'recommendations.'
            '</div>',
            unsafe_allow_html=True
        )

        # ==================================================
        # ENERGY OVERVIEW (KPI cards — shown above Add/Remove)
        # ==================================================

        section_title("Energy Overview")

        preview_maximum_energy = df["EnergyConsumption"].max()
        preview_minimum_energy = df["EnergyConsumption"].min()
        preview_high_alert_count = int(
            (df["EnergyConsumption"] > average_energy * 1.2).sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Average Consumption",
                f"{average_energy:.2f}",
                "Average energy usage"
            )

        with c2:
            metric_card(
                "Peak Consumption",
                f"{preview_maximum_energy:.2f}",
                "Highest recorded value"
            )

        with c3:
            metric_card(
                "Minimum Consumption",
                f"{preview_minimum_energy:.2f}",
                "Lowest recorded value"
            )

        with c4:
            metric_card(
                "High Energy Alerts",
                f"{preview_high_alert_count}",
                "Above 120% of average"
            )


        # ==================================================
        # LIVE DATA — ADD / REMOVE ENERGY RECORDS
        # ==================================================

        section_title("Add / Remove Live Data")

        st.markdown(
            '<div class="data-panel-title">'
            'Enter Reading — updates the whole dashboard'
            '</div>',
            unsafe_allow_html=True
        )

        e1, e2, e3, e4, e5, e6 = st.columns(
            [0.8, 1, 1, 1, 0.9, 0.9]
        )

        with e1:
            hour_text = st.text_input(
                "Hour (0-23)",
                value="12",
                key="record_hour_text"
            )

        with e2:
            temperature_text = st.text_input(
                "Outdoor Temp (°C)",
                value="24.0",
                key="record_temperature_text"
            )

        with e3:
            occupancy_text = st.text_input(
                "Occupancy",
                value="25",
                key="record_occupancy_text"
            )

        with e4:
            humidity_text = st.text_input(
                "Humidity (%)",
                value="45.0",
                key="record_humidity_text"
            )

        with e5:
            record_weekend = st.selectbox(
                "Weekend",
                ["No", "Yes"],
                key="record_weekend"
            )

        with e6:
            st.write("")
            st.write("")
            add_energy_submitted = st.button(
                "Add Record",
                key="add_energy_record_button"
            )

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

            energy_record_result = post_api(
                "/energy-records",
                new_energy_record
            )

            if energy_record_result is not None:
                st.success(
                    "Record added — dashboard updated "
                    "with the new reading."
                )
                st.rerun()

        st.markdown(
            '<div class="data-panel-title">'
            'Recent Records (click to delete)</div>',
            unsafe_allow_html=True
        )

        if not df.empty:
            recent_energy_records = df.tail(10).iloc[::-1]

            for _, energy_row in recent_energy_records.iterrows():
                energy_value = energy_row.get(
                    "EnergyConsumption", "-"
                )

                is_high = (
                    isinstance(energy_value, (int, float))
                    and energy_value > average_energy * 1.2
                )

                energy_row_class = (
                    "record-row failure"
                    if is_high
                    else "record-row"
                )

                ec1, ec2, ec3, ec4, ec5 = st.columns(
                    [1.3, 1, 1, 1, 0.6]
                )

                with ec1:
                    st.markdown(
                        f'<div class="{energy_row_class}">'
                        f'{energy_row.get("Timestamp", "N/A")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with ec2:
                    st.markdown(
                        f'<div class="{energy_row_class}">'
                        f'Temp: {energy_row.get("Temperature", "-")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with ec3:
                    st.markdown(
                        f'<div class="{energy_row_class}">'
                        f'Occ: {energy_row.get("Occupancy", "-")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with ec4:
                    st.markdown(
                        f'<div class="{energy_row_class}">'
                        f'Energy: {energy_value}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with ec5:
                    energy_record_id = energy_row.get("id", None)

                    if energy_record_id is not None and st.button(
                        "Delete",
                        key=f"delete_energy_{energy_record_id}"
                    ):
                        if delete_api(
                            f"/energy-records/{int(energy_record_id)}"
                        ):
                            st.success("Record deleted.")
                            st.rerun()
        else:
            empty_state(
                "No live records yet. Add one above to get started."
            )

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

        df["Simulated"] = "Historical"

        df = pd.concat(
            [
                df,
                simulated_df
            ],
            ignore_index=True
        )

        df["Simulated"] = df["Simulated"].fillna(
            "Simulated"
        )

        df["EnergyConsumption"] = pd.to_numeric(
            df["EnergyConsumption"],
            errors="coerce"
        )

        df["Temperature"] = pd.to_numeric(
            df["Temperature"],
            errors="coerce"
        )

        df["Occupancy"] = pd.to_numeric(
            df["Occupancy"],
            errors="coerce"
        )

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["EnergyConsumption"]
        )

        average_energy = df[
            "EnergyConsumption"
        ].mean()

        maximum_energy = df[
            "EnergyConsumption"
        ].max()

        minimum_energy = df[
            "EnergyConsumption"
        ].min()

        high_alert_count = int(
            (
                df["EnergyConsumption"]
                > average_energy * 1.2
            ).sum()
        )

        # ==================================================
        # CONSUMPTION TREND AREA CHART
        # ==================================================

        section_title("Consumption Trend")

        if "Hour" not in df.columns:
            df["Hour"] = df["Timestamp"].dt.hour
        else:
            df["Hour"] = pd.to_numeric(
                df["Hour"],
                errors="coerce"
            )

        hourly_average = (
            df.dropna(
                subset=[
                    "Hour",
                    "EnergyConsumption"
                ]
            )
            .groupby(
                "Hour",
                as_index=False
            )["EnergyConsumption"]
            .mean()
            .sort_values("Hour")
        )

        hourly_average["Hour"] = (
            hourly_average["Hour"].astype(int)
        )

        fig_trend = go.Figure()

        fig_trend.add_trace(
            go.Scatter(
                x=hourly_average["Hour"],
                y=hourly_average[
                    "EnergyConsumption"
                ],
                mode="lines+markers",
                name="Average Consumption",
                line=dict(
                    color="#20a4ff",
                    width=3,
                    shape="spline"
                ),
                marker=dict(
                    size=8,
                    color="#20a4ff",
                    line=dict(
                        color="#ffffff",
                        width=1
                    )
                ),
                fill="tozeroy",
                fillcolor="rgba(32, 164, 255, 0.28)",
                hovertemplate=(
                    "Hour: %{x}:00"
                    "<br>Average Consumption: "
                    "%{y:.2f} kWh"
                    "<extra></extra>"
                )
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
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            ),
            xaxis=dict(
                title="Hour of Day",
                tickmode="linear",
                tick0=0,
                dtick=1,
                range=[0, 23],
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title="Average Energy Consumption",
                rangemode="tozero",
                showgrid=True,
                gridcolor=(
                    "rgba(255,255,255,0.06)"
                ),
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )

        # ==================================================
        # TEMPERATURE VS ENERGY
        # ==================================================

        section_title("Temperature vs Energy")

        temperature_columns = [
            "Temperature",
            "EnergyConsumption",
            "Simulated"
        ]

        temperature_df = df.dropna(
            subset=[
                column
                for column in temperature_columns
                if column in df.columns
            ]
        )

        if not temperature_df.empty:
            temperature_hover = [
                column
                for column in [
                    "Timestamp",
                    "Occupancy",
                    "Hour",
                    "Weekend"
                ]
                if column in temperature_df.columns
            ]

            fig_temp = px.scatter(
                temperature_df,
                x="Temperature",
                y="EnergyConsumption",
                color="Simulated",
                hover_data=temperature_hover,
                color_discrete_map={
                    "Historical": "#20a4ff",
                    "Simulated": "#ff9d3c"
                }
            )

            fig_temp.update_traces(
                marker=dict(
                    size=9,
                    opacity=0.8,
                    line=dict(width=0)
                )
            )

            fig_temp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),
                xaxis_title="Outdoor Temperature (°C)",
                yaxis_title="Energy Consumption",
                legend_title="Record Type",
                xaxis=dict(
                    gridcolor=(
                        "rgba(255,255,255,0.06)"
                    )
                ),
                yaxis=dict(
                    gridcolor=(
                        "rgba(255,255,255,0.06)"
                    )
                )
            )

            st.plotly_chart(
                fig_temp,
                use_container_width=True
            )
        else:
            empty_state(
                "Temperature data is unavailable."
            )

        # ==================================================
        # OCCUPANCY VS ENERGY
        # ==================================================

        section_title("Occupancy vs Energy")

        occupancy_columns = [
            "Occupancy",
            "EnergyConsumption",
            "Simulated"
        ]

        occupancy_df = df.dropna(
            subset=[
                column
                for column in occupancy_columns
                if column in df.columns
            ]
        )

        if not occupancy_df.empty:
            occupancy_hover = [
                column
                for column in [
                    "Timestamp",
                    "Temperature",
                    "Hour",
                    "Weekend"
                ]
                if column in occupancy_df.columns
            ]

            fig_occ = px.scatter(
                occupancy_df,
                x="Occupancy",
                y="EnergyConsumption",
                color="Simulated",
                hover_data=occupancy_hover,
                color_discrete_map={
                    "Historical": "#20a4ff",
                    "Simulated": "#ff9d3c"
                }
            )

            fig_occ.update_traces(
                marker=dict(
                    size=9,
                    opacity=0.8,
                    line=dict(width=0)
                )
            )

            fig_occ.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),
                xaxis_title="Occupancy",
                yaxis_title="Energy Consumption",
                legend_title="Record Type",
                xaxis=dict(
                    gridcolor=(
                        "rgba(255,255,255,0.06)"
                    )
                ),
                yaxis=dict(
                    gridcolor=(
                        "rgba(255,255,255,0.06)"
                    )
                )
            )

            st.plotly_chart(
                fig_occ,
                use_container_width=True
            )
        else:
            empty_state(
                "Occupancy data is unavailable."
            )

        # ==================================================
        # HIGH ENERGY ALERTS
        # ==================================================

        section_title("High Energy Alerts")

        high_energy_df = df[
            df["EnergyConsumption"]
            > average_energy * 1.2
        ].copy()

        if not high_energy_df.empty:
            alert_columns = [
                "Timestamp",
                "Hour",
                "Temperature",
                "Occupancy",
                "Weekend",
                "Humidity",
                "SquareFootage",
                "HVACUsage",
                "LightingUsage",
                "RenewableEnergy",
                "DayOfWeek",
                "Holiday",
                "EnergyConsumption",
                "Simulated"
            ]

            alert_columns = [
                column
                for column in alert_columns
                if column in high_energy_df.columns
            ]

            st.dataframe(
                high_energy_df[alert_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(
                "No high energy consumption alerts detected."
            )

        # ==================================================
        # ALERTS AND RECOMMENDATIONS
        # ==================================================

        col1, col2 = st.columns(2)

        with col1:
            section_title("Energy Alerts")

            alerts = get_api("/alerts")

            if alerts:
                for alert in alerts[:8]:
                    alert_card(
                        "High Energy Consumption",
                        (
                            "Timestamp: "
                            f'{alert.get("Timestamp", "N/A")}'
                        )
                    )
            else:
                st.success(
                    "No high-energy alerts detected."
                )

        with col2:
            section_title("AI Recommendations")

            recommendations = get_api(
                "/recommendations"
            )

            if recommendations:
                for item in recommendations[:8]:
                    recommendation_card(
                        "AI Recommendation",
                        item.get(
                            "Recommendations",
                            []
                        )
                    )
            else:
                st.info(
                    "No recommendations currently available."
                )

    else:
        st.markdown(
            '<div class="main-title">'
            'Energy Intelligence'
            '</div>',
            unsafe_allow_html=True
        )

        empty_state(
            "Could not load data from the backend. "
            f"Confirm FastAPI is running at {BACKEND_URL}"
        )


# ==========================================================
# PREDICTIVE MAINTENANCE
# ==========================================================

else:
    maintenance = get_api(
        "/maintenance-dashboard"
    )

    if maintenance:
        st.markdown(
            '<div class="status-badge">'
            '● SYSTEM MONITORING'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="last-updated">'
            f'Last updated: '
            f'{datetime.now().strftime("%d %b %Y, %I:%M:%S %p")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="main-title">'
            'Predictive Maintenance'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="main-subtitle">'
            'Monitor equipment health, identify critical '
            'devices, and schedule maintenance before '
            'failures occur.'
            '</div>',
            unsafe_allow_html=True
        )

        # ==================================================
        # MAINTENANCE OVERVIEW
        # ==================================================

        section_title("Maintenance Overview")

        health_scores = get_api(
            "/health-score"
        )

        schedule = get_api(
            "/maintenance-schedule"
        )

        maintenance_records = get_api(
            "/maintenance-records"
        )

        maintenance_fields = get_api(
            "/maintenance-fields"
        )

        metric_field_names = (
            maintenance_fields.get("metric_columns", [])
            if maintenance_fields
            else []
        )

        records_df = (
            pd.DataFrame(maintenance_records)
            if maintenance_records
            else pd.DataFrame()
        )

        health_df_overview = (
            pd.DataFrame(health_scores)
            if health_scores
            else pd.DataFrame()
        )

        # Total Assets = unique devices, exactly as /maintenance-dashboard
        # defines it (a device can have several readings/rows).
        total_assets = maintenance.get(
            "Total Devices",
            len(records_df)
        )

        if not health_df_overview.empty and "Health Score" in health_df_overview.columns:
            avg_health_score = float(
                health_df_overview["Health Score"].mean()
            )
            lowest_score = float(
                health_df_overview["Health Score"].min()
            )
        else:
            avg_health_score = 0.0
            lowest_score = 0.0

        # High-risk readings = individual rows flagged as a failure
        # or classified Critical (Health Score < 50) — not just a
        # raw row count.
        if not records_df.empty and "Health Score" in records_df.columns:
            risky_mask = (
                records_df["Health Score"] < 50
            )

            if "failure" in records_df.columns:
                risky_mask = risky_mask | (
                    pd.to_numeric(
                        records_df["failure"],
                        errors="coerce"
                    ).fillna(0) == 1
                )

            high_risk_readings = int(risky_mask.sum())
        else:
            high_risk_readings = 0

        overview_col1, overview_col2, overview_col3 = st.columns(
            [1.1, 1, 1]
        )

        with overview_col1:
            st.markdown(
                '<div class="stat-panel-title">FLEET HEALTH</div>',
                unsafe_allow_html=True
            )

            st.plotly_chart(
                fleet_health_gauge(avg_health_score),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.markdown(
                f'<div style="text-align:center;color:#7d8fa8;'
                f'font-size:13px;margin-top:-14px;">'
                f'avg score / 100</div>',
                unsafe_allow_html=True
            )

        with overview_col2:
            stat_panel(
                "FLEET OVERVIEW",
                [
                    ("Total Assets", total_assets, "good"),
                    ("High-Risk Readings", high_risk_readings, "")
                ]
            )

            st.write("")

            stat_panel(
                "DEVICE STATUS",
                [
                    ("Healthy", maintenance["Healthy"], "good"),
                    ("Warning", maintenance["Warning"], ""),
                    ("Critical", maintenance["Critical"], "critical")
                ]
            )

        with overview_col3:
            stat_panel(
                "ATTENTION NEEDED",
                [
                    ("Lowest Score", f"{lowest_score:.1f}", "critical"),
                    ("Failures Logged", maintenance["Failures"], "critical")
                ]
            )

            st.write("")

            st.markdown(
                '<div class="live-tag">● LIVE</div>',
                unsafe_allow_html=True
            )

        # ==================================================
        # LIVE DATA — ADD / REMOVE RECORDS 
        # ==================================================

        @st.fragment
        def render_maintenance_live_data():
            local_fields = get_api("/maintenance-fields")
            local_metric_names = (
                local_fields.get("metric_columns", [])
                if local_fields
                else []
            )

            SCORING_METRICS = ["metric3", "metric4", "metric5", "metric6", "metric9"]
            local_metric_names = [
                name for name in local_metric_names
                if name in SCORING_METRICS
            ]

            local_records = get_api("/maintenance-records")
            local_records_df = (
                pd.DataFrame(local_records)
                if local_records
                else pd.DataFrame()
            )

            def handle_add_maintenance_record():
                device = st.session_state.get("new_device_id", "").strip()

                if not device:
                    st.session_state["_maint_flash"] = (
                        "warning", "Please enter a device name or ID."
                    )
                    return

                metric_payload = {
                    field_name: st.session_state.get(f"metric_input_{field_name}", 0.0)
                    for field_name in local_metric_names
                }

                new_record = {
                    "device": device,
                    "failure": st.session_state.get("new_failure_flag", "No"),
                    **metric_payload
                }

                result = post_api("/maintenance-records", new_record)

                if result is not None:
                    st.session_state["_maint_flash"] = ("success", "Record added successfully.")

            def handle_delete_maintenance_record(record_id):
                if delete_api(f"/maintenance-records/{record_id}"):
                    st.session_state["_maint_flash"] = ("success", "Record deleted.")

            section_title("Add / Remove Live Data")

            st.markdown('<div class="data-panel">', unsafe_allow_html=True)
            st.markdown(
                '<div class="data-panel-title">Add New Reading</div>',
                unsafe_allow_html=True
            )

            with st.form("add_record_form", clear_on_submit=True):
                top_col1, top_col2 = st.columns([1.3, 1])

                with top_col1:
                    st.text_input(
                        "Device",
                        placeholder="e.g. Device-01",
                        key="new_device_id"
                    )

                with top_col2:
                    st.selectbox(
                        "Failure?",
                        ["No", "Yes"],
                        key="new_failure_flag"
                    )

                if local_metric_names:
                    metric_input_columns = st.columns(
                        min(len(local_metric_names), 4)
                    )

                    for index, field_name in enumerate(local_metric_names):
                        with metric_input_columns[index % len(metric_input_columns)]:
                            st.number_input(
                                field_name.replace("_", " ").title(),
                                value=0.0,
                                step=0.1,
                                key=f"metric_input_{field_name}"
                            )
                else:
                    st.caption(
                        "No metric columns detected yet — add a record "
                        "once the backend is reachable."
                    )

                st.form_submit_button(
                    "Add Record",
                    on_click=handle_add_maintenance_record
                )

            st.markdown('</div>', unsafe_allow_html=True)

            flash = st.session_state.pop("_maint_flash", None)
            if flash:
                kind, message = flash
                getattr(st, kind)(message)

            st.markdown(
                '<div class="data-panel-title">'
                'Recent Records (click to delete)</div>',
                unsafe_allow_html=True
            )

            if not local_records_df.empty:
                recent_records = local_records_df.tail(10).iloc[::-1]
                preview_metrics = local_metric_names[:2]

                for _, row in recent_records.iterrows():
                    is_failure = str(
                        row.get("failure", 0)
                    ) in ["1", "1.0", "True", "yes", "Yes"]

                    row_class = "record-row failure" if is_failure else "record-row"

                    rc1, rc2, rc3, rc4, rc5 = st.columns(
                        [1.2, 1, 1, 0.9, 0.6]
                    )

                    with rc1:
                        st.markdown(
                            f'<div class="{row_class}">'
                            f'{row.get("device", "N/A")}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with rc2:
                        metric_a = preview_metrics[0] if len(preview_metrics) > 0 else None
                        label_a = metric_a.replace("_", " ").title() if metric_a else "-"
                        value_a = row.get(metric_a, "-") if metric_a else "-"

                        st.markdown(
                            f'<div class="{row_class}">'
                            f'{label_a}: {value_a}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with rc3:
                        metric_b = preview_metrics[1] if len(preview_metrics) > 1 else None
                        label_b = metric_b.replace("_", " ").title() if metric_b else "-"
                        value_b = row.get(metric_b, "-") if metric_b else "-"

                        st.markdown(
                            f'<div class="{row_class}">'
                            f'{label_b}: {value_b}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with rc4:
                        health_value = row.get("Health Score", "-")

                        st.markdown(
                            f'<div class="{row_class}">'
                            f'Score: {health_value}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with rc5:
                        record_id = row.get("id", None)

                        if record_id is not None:
                            st.button(
                                "Delete",
                                key=f"delete_{record_id}",
                                on_click=handle_delete_maintenance_record,
                                args=(int(record_id),)
                            )
            else:
                empty_state(
                    "No live records yet. Add one above to get started."
                )

        render_maintenance_live_data()
        # ==================================================
        # HEALTH ANALYTICS
        # ==================================================

        section_title("Health Analytics")


        total_devices_for_status = (
            maintenance["Healthy"]
            + maintenance["Warning"]
            + maintenance["Critical"]
        )

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            healthy_pct = (
                maintenance["Healthy"] / total_devices_for_status * 100
                if total_devices_for_status else 0
            )
            health_card(
                "healthy-card",
                "Healthy",
                f'{maintenance["Healthy"]} ({healthy_pct:.2f}%)'
            )

        with status_col2:
            warning_pct = (
                maintenance["Warning"] / total_devices_for_status * 100
                if total_devices_for_status else 0
            )
            health_card(
                "warning-card",
                "Warning",
                f'{maintenance["Warning"]} ({warning_pct:.2f}%)'
            )

        with status_col3:
            critical_pct = (
                maintenance["Critical"] / total_devices_for_status * 100
                if total_devices_for_status else 0
            )
            health_card(
                "critical-card",
                "Critical",
                f'{maintenance["Critical"]} ({critical_pct:.2f}%)'
            )

        if health_scores:
            status_table_df = pd.DataFrame(health_scores)

            st.write("")

            table_col1, table_col2, table_col3 = st.columns(3)

            with table_col1:
                st.markdown(
                    '<div class="stat-panel-title">HEALTHY ASSETS</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(
                    status_table_df[
                        status_table_df["Status"] == "Healthy"
                    ][["Device", "Health Score"]],
                    use_container_width=True,
                    hide_index=True,
                    height=250
                )

            with table_col2:
                st.markdown(
                    '<div class="stat-panel-title">WARNING ASSETS</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(
                    status_table_df[
                        status_table_df["Status"] == "Warning"
                    ][["Device", "Health Score"]],
                    use_container_width=True,
                    hide_index=True,
                    height=250
                )

            with table_col3:
                st.markdown(
                    '<div class="stat-panel-title">CRITICAL ASSETS</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(
                    status_table_df[
                        status_table_df["Status"] == "Critical"
                    ][["Device", "Health Score"]],
                    use_container_width=True,
                    hide_index=True,
                    height=250
                )
        else:
            empty_state("No device health data available yet.")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            status_counts = {
                "Healthy": maintenance["Healthy"],
                "Warning": maintenance["Warning"],
                "Critical": maintenance["Critical"]
            }

            if sum(status_counts.values()) > 0:
                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=list(
                                status_counts.keys()
                            ),
                            values=list(
                                status_counts.values()
                            ),
                            hole=0.55,
                            marker=dict(
                                colors=[
                                    "#00e676",
                                    "#ffb020",
                                    "#ff5a6a"
                                ]
                            ),
                            textinfo="label+percent",
                            hovertemplate=(
                                "%{label}: %{value} "
                                "devices (%{percent})"
                                "<extra></extra>"
                            )
                        )
                    ]
                )

                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(
                        l=10,
                        r=10,
                        t=40,
                        b=10
                    ),
                    showlegend=True,
                    title="Device Health Distribution"
                )

                st.plotly_chart(
                    fig_pie,
                    use_container_width=True
                )
            else:
                empty_state(
                    "No device health data available yet."
                )

        with viz_col2:
            if schedule:
                schedule_df = pd.DataFrame(
                    schedule
                )

                priority_counts = (
                    schedule_df["Priority"]
                    .value_counts()
                    .reindex(
                        [
                            "High",
                            "Medium",
                            "Low"
                        ]
                    )
                    .fillna(0)
                )

                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=priority_counts.index,
                            y=priority_counts.values,
                            marker_color=[
                                "#ff5a6a",
                                "#ffb020",
                                "#20a4ff"
                            ],
                            text=(
                                priority_counts
                                .values
                                .astype(int)
                            ),
                            textposition="outside"
                        )
                    ]
                )

                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(
                        l=10,
                        r=10,
                        t=40,
                        b=10
                    ),
                    title=(
                        "Maintenance Priority Breakdown"
                    ),
                    xaxis_title="Priority",
                    yaxis_title="Devices",
                    yaxis=dict(
                        gridcolor=(
                            "rgba(255,255,255,0.06)"
                        )
                    )
                )

                st.plotly_chart(
                    fig_bar,
                    use_container_width=True
                )
            else:
                empty_state(
                    "No schedule data available for "
                    "priority breakdown."
                )

        if health_scores:
            health_df = pd.DataFrame(
                health_scores
            )

            fig_hist = go.Figure(
                data=[
                    go.Histogram(
                        x=health_df["Health Score"],
                        nbinsx=20,
                        marker_color="#20a4ff",
                        opacity=0.85
                    )
                ]
            )

            fig_hist.add_vline(
                x=health_df["Health Score"].mean(),
                line_dash="dash",
                line_color="#ff9d3c",
                annotation_text="Average",
                annotation_font_color="#8fa3bf"
            )

            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(
                    l=10,
                    r=10,
                    t=40,
                    b=10
                ),
                title="Health Score Distribution",
                xaxis_title="Health Score",
                yaxis_title="Number of Devices",
                xaxis=dict(
                    gridcolor=(
                        "rgba(255,255,255,0.06)"
                    )
                ),
                yaxis=dict(
                    gridcolor=(
                        "rgba(255,255,255,0.06)"
                    )
                )
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )
        else:
            empty_state(
                "No health score data available."
            )

        # ==================================================
        # MAINTENANCE ALERTS
        # ==================================================

        section_title("Maintenance Alerts")

        maintenance_alerts = get_api(
            "/maintenance-alerts"
        )

        if maintenance_alerts:
            st.dataframe(
                pd.DataFrame(
                    maintenance_alerts
                ).head(15),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(
                "No maintenance alerts."
            )

        # ==================================================
        # MAINTENANCE SCHEDULE
        # ==================================================

        section_title("Maintenance Schedule")

        if schedule:
            st.dataframe(
                pd.DataFrame(
                    schedule
                ).head(15),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No maintenance schedule available."
            )

    else:
        st.markdown(
            '<div class="main-title">'
            'Predictive Maintenance'
            '</div>',
            unsafe_allow_html=True
        )

        empty_state(
            "Could not load data from the backend. "
            f"Confirm FastAPI is running at {BACKEND_URL}"
        )