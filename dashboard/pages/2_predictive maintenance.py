import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from shared import (
    configure_page, sidebar_branding, get_api, post_api, delete_api,
    health_card, empty_state, section_title, stat_panel, fleet_health_gauge,
    BACKEND_URL
)

configure_page("Predictive Maintenance")
sidebar_branding()

maintenance = get_api("/maintenance-dashboard")

if not maintenance:
    st.markdown('<div class="main-title">Predictive Maintenance</div>', unsafe_allow_html=True)
    empty_state(
        "Could not load data from the backend. "
        f"Confirm FastAPI is running at {BACKEND_URL}"
    )
    st.stop()

st.markdown('<div class="status-badge">● SYSTEM MONITORING</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="last-updated">Last updated: '
    f'{datetime.now().strftime("%d %b %Y, %I:%M:%S %p")}</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Predictive Maintenance</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="main-subtitle">Monitor equipment health, identify critical '
    'devices, and schedule maintenance before failures occur.</div>',
    unsafe_allow_html=True
)

# ==================================================
# MAINTENANCE OVERVIEW
# ==================================================

section_title("Maintenance Overview")

health_scores = get_api("/health-score")
schedule = get_api("/maintenance-schedule")
maintenance_records = get_api("/maintenance-records")
maintenance_fields = get_api("/maintenance-fields")

metric_field_names = (
    maintenance_fields.get("metric_columns", []) if maintenance_fields else []
)

records_df = pd.DataFrame(maintenance_records) if maintenance_records else pd.DataFrame()
health_df_overview = pd.DataFrame(health_scores) if health_scores else pd.DataFrame()

# Total Assets = unique devices, exactly as /maintenance-dashboard defines it
# (a device can have several readings/rows).
total_assets = maintenance.get("Total Devices", len(records_df))

if not health_df_overview.empty and "Health Score" in health_df_overview.columns:
    avg_health_score = float(health_df_overview["Health Score"].mean())
    lowest_score = float(health_df_overview["Health Score"].min())
else:
    avg_health_score = 0.0
    lowest_score = 0.0

# High-risk readings = individual rows flagged as a failure or classified
# Critical (Health Score < 50) — not just a raw row count.
if not records_df.empty and "Health Score" in records_df.columns:
    risky_mask = records_df["Health Score"] < 50

    if "failure" in records_df.columns:
        risky_mask = risky_mask | (
            pd.to_numeric(records_df["failure"], errors="coerce").fillna(0) == 1
        )

    high_risk_readings = int(risky_mask.sum())
else:
    high_risk_readings = 0

overview_col1, overview_col2, overview_col3 = st.columns([1.1, 1, 1])

with overview_col1:
    st.markdown('<div class="stat-panel-title">FLEET HEALTH</div>', unsafe_allow_html=True)

    st.plotly_chart(
        fleet_health_gauge(avg_health_score),
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown(
        '<div style="text-align:center;color:#7d8fa8;font-size:13px;margin-top:-14px;">'
        'avg score / 100</div>',
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

    st.markdown('<div class="live-tag">● LIVE</div>', unsafe_allow_html=True)


# ==================================================
# LIVE DATA — ADD / REMOVE RECORDS
# ==================================================

@st.fragment
def render_maintenance_live_data(full_records_df, all_metric_names):
    """
    Reuses the records_df and metric_field_names already fetched at the
    top of the page instead of re-fetching and re-parsing the full
    ~100k+ row maintenance table a second time just to show the last
    10 records. Only a POST/DELETE inside this fragment needs a fresh
    read, and those already call clear_api_cache() so the next full
    page rerun (st.rerun()) picks up the change.
    """
    SCORING_METRICS = ["metric3", "metric4", "metric5", "metric6", "metric9"]
    local_metric_names = [n for n in all_metric_names if n in SCORING_METRICS]

    local_records_df = full_records_df

    def handle_add_maintenance_record():
        device = st.session_state.get("new_device_id", "").strip()

        if not device:
            st.session_state["_maint_flash"] = ("warning", "Please enter a device name or ID.")
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

    st.markdown('<div class="data-panel-title">Add New Reading</div>', unsafe_allow_html=True)

    with st.form("add_record_form", clear_on_submit=True):
        top_col1, top_col2 = st.columns([1.3, 1])

        with top_col1:
            st.text_input("Device", placeholder="e.g. Device-01", key="new_device_id")

        with top_col2:
            st.selectbox("Failure?", ["No", "Yes"], key="new_failure_flag")

        if local_metric_names:
            metric_input_columns = st.columns(min(len(local_metric_names), 4))

            for index, field_name in enumerate(local_metric_names):
                with metric_input_columns[index % len(metric_input_columns)]:
                    st.number_input(
                        field_name.replace("_", " ").title(),
                        value=0.0, step=0.1, key=f"metric_input_{field_name}"
                    )
        else:
            st.caption(
                "No metric columns detected yet — add a record "
                "once the backend is reachable."
            )

        st.form_submit_button("Add Record", on_click=handle_add_maintenance_record)

    flash = st.session_state.pop("_maint_flash", None)
    if flash:
        kind, message = flash
        getattr(st, kind)(message)

    st.markdown(
        '<div class="data-panel-title">Recent Records (click to delete)</div>',
        unsafe_allow_html=True
    )

    if not local_records_df.empty:
        recent_records = local_records_df.tail(10).iloc[::-1]
        preview_metrics = local_metric_names[:2]

        for _, row in recent_records.iterrows():
            is_failure = str(row.get("failure", 0)) in ["1", "1.0", "True", "yes", "Yes"]
            row_class = "record-row failure" if is_failure else "record-row"

            rc1, rc2, rc3, rc4, rc5 = st.columns([1.2, 1, 1, 0.9, 0.6])

            with rc1:
                st.markdown(
                    f'<div class="{row_class}">{row.get("device", "N/A")}</div>',
                    unsafe_allow_html=True
                )

            with rc2:
                metric_a = preview_metrics[0] if len(preview_metrics) > 0 else None
                label_a = metric_a.replace("_", " ").title() if metric_a else "-"
                value_a = row.get(metric_a, "-") if metric_a else "-"

                st.markdown(
                    f'<div class="{row_class}">{label_a}: {value_a}</div>',
                    unsafe_allow_html=True
                )

            with rc3:
                metric_b = preview_metrics[1] if len(preview_metrics) > 1 else None
                label_b = metric_b.replace("_", " ").title() if metric_b else "-"
                value_b = row.get(metric_b, "-") if metric_b else "-"

                st.markdown(
                    f'<div class="{row_class}">{label_b}: {value_b}</div>',
                    unsafe_allow_html=True
                )

            with rc4:
                health_value = row.get("Health Score", "-")

                st.markdown(
                    f'<div class="{row_class}">Score: {health_value}</div>',
                    unsafe_allow_html=True
                )

            with rc5:
                record_id = row.get("id", None)

                if record_id is not None:
                    st.button(
                        "Delete", key=f"delete_{record_id}",
                        on_click=handle_delete_maintenance_record,
                        args=(int(record_id),)
                    )
    else:
        empty_state("No live records yet. Add one above to get started.")


render_maintenance_live_data(records_df, metric_field_names)

# ==================================================
# HEALTH ANALYTICS
# ==================================================

section_title("Health Analytics")

total_devices_for_status = maintenance["Healthy"] + maintenance["Warning"] + maintenance["Critical"]

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    healthy_pct = (
        maintenance["Healthy"] / total_devices_for_status * 100
        if total_devices_for_status else 0
    )
    health_card("healthy-card", "Healthy", f'{maintenance["Healthy"]} ({healthy_pct:.2f}%)')

with status_col2:
    warning_pct = (
        maintenance["Warning"] / total_devices_for_status * 100
        if total_devices_for_status else 0
    )
    health_card("warning-card", "Warning", f'{maintenance["Warning"]} ({warning_pct:.2f}%)')

with status_col3:
    critical_pct = (
        maintenance["Critical"] / total_devices_for_status * 100
        if total_devices_for_status else 0
    )
    health_card("critical-card", "Critical", f'{maintenance["Critical"]} ({critical_pct:.2f}%)')

if health_scores:
    status_table_df = pd.DataFrame(health_scores)

    st.write("")

    table_col1, table_col2, table_col3 = st.columns(3)

    with table_col1:
        st.markdown('<div class="stat-panel-title">HEALTHY ASSETS</div>', unsafe_allow_html=True)
        st.dataframe(
            status_table_df[status_table_df["Status"] == "Healthy"][["Device", "Health Score"]],
            use_container_width=True, hide_index=True, height=250
        )

    with table_col2:
        st.markdown('<div class="stat-panel-title">WARNING ASSETS</div>', unsafe_allow_html=True)
        st.dataframe(
            status_table_df[status_table_df["Status"] == "Warning"][["Device", "Health Score"]],
            use_container_width=True, hide_index=True, height=250
        )

    with table_col3:
        st.markdown('<div class="stat-panel-title">CRITICAL ASSETS</div>', unsafe_allow_html=True)
        st.dataframe(
            status_table_df[status_table_df["Status"] == "Critical"][["Device", "Health Score"]],
            use_container_width=True, hide_index=True, height=250
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
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.55,
                    marker=dict(colors=["#00e676", "#ffb020", "#ff5a6a"]),
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} devices (%{percent})<extra></extra>"
                )
            ]
        )

        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=360, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=True, title="Device Health Distribution"
        )

        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        empty_state("No device health data available yet.")

with viz_col2:
    if schedule:
        schedule_df = pd.DataFrame(schedule)

        priority_counts = (
            schedule_df["Priority"].value_counts()
            .reindex(["High", "Medium", "Low"])
            .fillna(0)
        )

        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    marker_color=["#ff5a6a", "#ffb020", "#20a4ff"],
                    text=priority_counts.values.astype(int),
                    textposition="outside"
                )
            ]
        )

        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=360, margin=dict(l=10, r=10, t=40, b=10),
            title="Maintenance Priority Breakdown",
            xaxis_title="Priority", yaxis_title="Devices",
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
        )

        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        empty_state("No schedule data available for priority breakdown.")

if health_scores:
    health_df = pd.DataFrame(health_scores)

    fig_hist = go.Figure(
        data=[
            go.Histogram(
                x=health_df["Health Score"], nbinsx=20,
                marker_color="#20a4ff", opacity=0.85
            )
        ]
    )

    fig_hist.add_vline(
        x=health_df["Health Score"].mean(),
        line_dash="dash", line_color="#ff9d3c",
        annotation_text="Average", annotation_font_color="#8fa3bf"
    )

    fig_hist.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        title="Health Score Distribution",
        xaxis_title="Health Score", yaxis_title="Number of Devices",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    st.plotly_chart(fig_hist, use_container_width=True)
else:
    empty_state("No health score data available.")

# ==================================================
# MAINTENANCE ALERTS
# ==================================================

section_title("Maintenance Alerts")

maintenance_alerts = get_api("/maintenance-alerts")

if maintenance_alerts:
    st.dataframe(pd.DataFrame(maintenance_alerts).head(15), use_container_width=True, hide_index=True)
else:
    st.success("No maintenance alerts.")

# ==================================================
# MAINTENANCE SCHEDULE
# ==================================================

section_title("Maintenance Schedule")

if schedule:
    st.dataframe(pd.DataFrame(schedule).head(15), use_container_width=True, hide_index=True)
else:
    st.info("No maintenance schedule available.")