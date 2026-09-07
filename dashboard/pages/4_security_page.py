import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from shared import (
    configure_page, sidebar_branding, get_api, post_api, delete_api,
    metric_card, alert_card, empty_state, section_title, BACKEND_URL
)

configure_page("Security Agent")
sidebar_branding()

dashboard = get_api("/security/dashboard")

if not dashboard:
    st.markdown('<div class="main-title">Security Agent</div>', unsafe_allow_html=True)
    empty_state(
        "Could not load data from the backend. "
        f"Confirm FastAPI is running at {BACKEND_URL}, and that "
        "the security_*.csv files exist in dataset/ (run "
        "security_agent/generate_security_data.py if not)."
    )
    st.stop()

st.markdown('<div class="status-badge">● SYSTEM MONITORING</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="last-updated">Last updated: '
    f'{datetime.now().strftime("%d %b %Y, %I:%M:%S %p")}</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Security Agent</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="main-subtitle">Access control monitoring, unauthorized access '
    'detection, CCTV event analysis, visitor movement tracking, and incident '
    'investigation.</div>',
    unsafe_allow_html=True
)

# ==================================================
# OVERVIEW KPIs
# ==================================================

section_title("Security Overview")

k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card(
        "Critical Alerts",
        f'{dashboard["Critical Alerts"]}',
        "Repeated unauthorized access, after-hours restricted entry"
    )

with k2:
    metric_card(
        "High Alerts",
        f'{dashboard["High Alerts"]}',
        "Unauthorized access to restricted areas"
    )

with k3:
    metric_card(
        "Denied Access Attempts",
        f'{dashboard["Denied Attempts"]} / {dashboard["Total Access Attempts"]}',
        "Across all zones"
    )

with k4:
    metric_card(
        "Flagged Visitor Visits",
        f'{dashboard["Flagged Visitor Visits"]} / {dashboard["Total Visitors"]}',
        "Entered a restricted zone unescorted"
    )

# ==================================================
# ACCESS CONTROL MONITORING
# ==================================================

section_title("Access Control Monitoring")

access_summary = get_api("/security/access-logs/summary")

if access_summary:
    access_df = pd.DataFrame(access_summary)

    fig_access = go.Figure()

    fig_access.add_trace(
        go.Bar(
            x=access_df["zone"],
            y=access_df["denial_rate_pct"],
            marker_color=[
                "#ff5a6a" if v >= 8 else "#ffb020" if v >= 4 else "#00e676"
                for v in access_df["denial_rate_pct"]
            ],
            hovertemplate="%{x}<br>Denial rate: %{y:.1f}%<extra></extra>"
        )
    )

    fig_access.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Zone",
        yaxis_title="Denial Rate %",
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    st.plotly_chart(fig_access, use_container_width=True)

    st.dataframe(
        access_df.rename(columns={
            "floor": "Floor", "zone": "Zone", "total_attempts": "Total Attempts",
            "denied": "Denied", "denial_rate_pct": "Denial Rate %"
        }),
        use_container_width=True, hide_index=True
    )
else:
    empty_state("No access log data available yet.")

# ==================================================
# SECURITY ALERTS (unauthorized access detection)
# ==================================================

section_title("Security Alerts")

security_alerts = get_api("/security/alerts")

if security_alerts:
    for alert in security_alerts[:10]:
        alert_card(
            f'{alert["severity"]} — {alert["zone"]} (Floor {int(alert["floor"])})',
            f'{alert["description"]}'
        )
else:
    st.success("No HIGH or CRITICAL security alerts — all clear.")

# ==================================================
# CCTV EVENT ANALYSIS
# ==================================================

section_title("CCTV Event Analysis")

cctv_events = get_api("/security/cctv-events", params={"limit": 500})

if cctv_events:
    cctv_df = pd.DataFrame(cctv_events)

    type_counts = cctv_df["event_type"].value_counts().reset_index()
    type_counts.columns = ["event_type", "count"]

    fig_cctv = go.Figure(
        go.Bar(
            x=type_counts["event_type"],
            y=type_counts["count"],
            marker_color="#20a4ff",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>"
        )
    )

    fig_cctv.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Event Type",
        yaxis_title="Count",
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    st.plotly_chart(fig_cctv, use_container_width=True)

    st.markdown('<div class="data-panel-title">Recent CCTV Events</div>', unsafe_allow_html=True)

    recent_cctv = cctv_df.sort_values("timestamp", ascending=False).head(15)

    st.dataframe(
        recent_cctv[["timestamp", "zone", "camera_id", "event_type", "confidence"]].rename(columns={
            "timestamp": "Timestamp", "zone": "Zone", "camera_id": "Camera",
            "event_type": "Event Type", "confidence": "Confidence"
        }),
        use_container_width=True, hide_index=True
    )
else:
    empty_state("No CCTV event data available yet.")

# ==================================================
# VISITOR MOVEMENT TRACKING
# ==================================================

section_title("Visitor Movement Tracking")

all_visitors = get_api("/security/visitors", params={"limit": 200})

if all_visitors:
    visitors_df = pd.DataFrame(all_visitors)

    flagged_df = visitors_df[visitors_df["flagged_unescorted_restricted_access"] == True]  # noqa: E712

    st.markdown(
        f'<div class="main-subtitle" style="margin-bottom:12px;">'
        f'{len(flagged_df)} of {len(visitors_df)} visits entered a restricted zone unescorted.'
        f'</div>',
        unsafe_allow_html=True
    )

    vc1, vc2 = st.columns([1, 2])

    with vc1:
        show_flagged_only = st.checkbox("Show flagged visits only", value=True, key="visitor_flagged_filter")

        display_df = flagged_df if show_flagged_only else visitors_df
        display_df = display_df.sort_values("entry_time", ascending=False)

        visitor_labels = {
            row["visitor_id"]: f'{row["name"]} — {row["visitor_id"]} (hosted by {row["host_name"]})'
            for _, row in display_df.iterrows()
        }

        if visitor_labels:
            selected_visitor_id = st.selectbox(
                "Select a visit to see its path",
                options=list(visitor_labels.keys()),
                format_func=lambda vid: visitor_labels[vid],
                key="visitor_path_select"
            )
        else:
            selected_visitor_id = None
            empty_state("No visits match this filter.")

    with vc2:
        if selected_visitor_id:
            path_result = get_api(f"/security/visitors/{selected_visitor_id}/path")

            if path_result and path_result.get("path"):
                path_df = pd.DataFrame(path_result["path"]).sort_values("sequence")

                path_str = "  →  ".join(
                    f'{row["zone"]}' for _, row in path_df.iterrows()
                )

                st.markdown(
                    f'<div class="data-panel-title">Path</div>'
                    f'<div class="main-subtitle" style="font-size:1.05rem;">{path_str}</div>',
                    unsafe_allow_html=True
                )

                st.dataframe(
                    path_df[["sequence", "floor", "zone", "timestamp"]].rename(columns={
                        "sequence": "Step", "floor": "Floor", "zone": "Zone", "timestamp": "Timestamp"
                    }),
                    use_container_width=True, hide_index=True
                )
            else:
                empty_state("No movement data for this visit.")
else:
    empty_state("No visitor data available yet.")

# ==================================================
# INCIDENT INVESTIGATION
# ==================================================

section_title("Incident Investigation")

recent_events = get_api("/security/events", params={"limit": 100})

if recent_events:
    events_df = pd.DataFrame(recent_events).sort_values("timestamp", ascending=False)

    event_labels = {
        int(row["event_id"]): (
            f'#{int(row["event_id"])} — {row["severity"]} — {row["event_type"]} '
            f'— {row["zone"]} ({str(row["timestamp"])[:16]})'
        )
        for _, row in events_df.iterrows()
    }

    ic1, ic2 = st.columns([1, 1])

    with ic1:
        selected_event_id = st.selectbox(
            "Select a security event to investigate",
            options=list(event_labels.keys()),
            format_func=lambda eid: event_labels[eid],
            key="incident_event_select"
        )

    with ic2:
        window_minutes = st.slider(
            "Time window (minutes before/after)", min_value=5, max_value=120, value=30, step=5,
            key="incident_window_select"
        )

    if selected_event_id is not None:
        incident = get_api(
            f"/security/incidents/{selected_event_id}",
            params={"window_minutes": window_minutes}
        )

        if incident:
            event = incident["event"]

            metric_card(
                f'Event #{event["event_id"]} — {event["severity"]}',
                event["event_type"],
                event.get("description", "")
            )

            tab_access, tab_cctv, tab_visitors = st.tabs([
                f'Access Logs ({len(incident["related_access_logs"])})',
                f'CCTV Events ({len(incident["related_cctv_events"])})',
                f'Visitor Movements ({len(incident["related_visitor_movements"])})'
            ])

            with tab_access:
                if incident["related_access_logs"]:
                    al_df = pd.DataFrame(incident["related_access_logs"])
                    st.dataframe(
                        al_df[["timestamp", "badge_id", "door_id", "access_granted"]].rename(columns={
                            "timestamp": "Timestamp", "badge_id": "Badge", "door_id": "Door",
                            "access_granted": "Granted"
                        }),
                        use_container_width=True, hide_index=True
                    )
                else:
                    empty_state("No access log activity in this window.")

            with tab_cctv:
                if incident["related_cctv_events"]:
                    cc_df = pd.DataFrame(incident["related_cctv_events"])
                    st.dataframe(
                        cc_df[["timestamp", "camera_id", "event_type", "confidence"]].rename(columns={
                            "timestamp": "Timestamp", "camera_id": "Camera",
                            "event_type": "Event Type", "confidence": "Confidence"
                        }),
                        use_container_width=True, hide_index=True
                    )
                else:
                    empty_state("No CCTV activity in this window.")

            with tab_visitors:
                if incident["related_visitor_movements"]:
                    vm_df = pd.DataFrame(incident["related_visitor_movements"])
                    st.dataframe(
                        vm_df[["timestamp", "visitor_id", "sequence"]].rename(columns={
                            "timestamp": "Timestamp", "visitor_id": "Visitor", "sequence": "Step"
                        }),
                        use_container_width=True, hide_index=True
                    )
                else:
                    empty_state("No visitor movement in this window.")
else:
    empty_state("No security events available yet.")