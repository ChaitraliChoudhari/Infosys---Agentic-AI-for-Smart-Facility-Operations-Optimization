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

    if module == " Energy Intelligence":
        st.markdown(
            '<div class="sidebar-heading">'
            'Energy Inputs'
            '</div>',
            unsafe_allow_html=True
        )

        input_hour = st.slider(
            "Hour",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )

        input_temperature = st.number_input(
            "Outdoor Temperature (°C)",
            min_value=-30.0,
            max_value=60.0,
            value=24.0,
            step=0.5
        )

        input_occupancy = st.number_input(
            "Occupancy",
            min_value=0,
            max_value=10000,
            value=25,
            step=1
        )

        input_weekend = st.selectbox(
            "Weekend",
            [
                "No",
                "Yes"
            ],
            index=0
        )

        st.caption(
            "Change any input to update the dashboard."
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

    simulation = get_api(
        "/energy/simulate",
        params={
            "hour": input_hour,
            "outdoor_temperature": input_temperature,
            "occupancy": input_occupancy,
            "weekend": input_weekend
        }
    )

    if dashboard is not None and energy_data is not None:
        if simulation is None:
            st.stop()

        df = pd.DataFrame(energy_data)

        if df.empty:
            st.error("The energy dataset is empty.")
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
        # ENERGY OVERVIEW
        # ==================================================

        section_title("Energy Overview")

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
                f"{maximum_energy:.2f}",
                "Highest recorded value"
            )

        with c3:
            metric_card(
                "Minimum Consumption",
                f"{minimum_energy:.2f}",
                "Lowest recorded value"
            )

        with c4:
            metric_card(
                "High Energy Alerts",
                f"{high_alert_count}",
                "Above 120% of average"
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

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            health_card(
                "healthy-card",
                "Healthy Devices",
                maintenance["Healthy"]
            )

        with c2:
            health_card(
                "warning-card",
                "Warning Devices",
                maintenance["Warning"]
            )

        with c3:
            health_card(
                "critical-card",
                "Critical Devices",
                maintenance["Critical"]
            )

        with c4:
            health_card(
                "critical-card",
                "Recorded Failures",
                maintenance["Failures"]
            )

        # ==================================================
        # HEALTH ANALYTICS
        # ==================================================

        section_title("Health Analytics")

        health_scores = get_api(
            "/health-score"
        )

        schedule = get_api(
            "/maintenance-schedule"
        )

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