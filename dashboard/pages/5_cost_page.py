import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from shared import (
    configure_page,
    sidebar_branding,
    metric_card,
    section_title,
    recommendation_card,
    empty_state,
    stat_panel,
    get_api,
    post_api,
)

configure_page("Cost Optimization | Agentic FacilityOps AI")
sidebar_branding()


st.markdown('<div class="main-title">Cost Optimization Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Facility-wide cost intelligence, ROI tracking, and enterprise reporting.</div>',
    unsafe_allow_html=True
)

dashboard = get_api("/cost/dashboard")
breakdown = get_api("/cost/breakdown")
optimizations = get_api("/cost/optimizations")

# ==========================================================
# TOP KPI ROW
# ==========================================================

section_title("Facility Intelligence Overview")

if dashboard:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Cost Reduction",
            f'{dashboard["Cost Reduction %"]}%',
            "Potential savings vs. current spend"
        )
    with col2:
        metric_card(
            "ROI Generated",
            f'₹{dashboard["ROI Generated"]:,.0f}',
            "Total identified savings opportunity"
        )
    with col3:
        metric_card(
            "Facility Health",
            f'{dashboard["Facility Health"]}/100',
            "Composite score across all agents"
        )
    with col4:
        metric_card(
            "Optimizations",
            dashboard["Optimizations"],
            "Active recommendations"
        )
else:
    empty_state("Cost dashboard data is unavailable. Check the FastAPI backend connection.")

# ==========================================================
# BUDGET VARIANCE ROW
# ==========================================================

section_title("Budget & Financial Intelligence")

variance = get_api("/cost/budget-variance")

if variance:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Monthly OpEx",
            f'₹{variance["Total Monthly OpEx"]:,.0f}',
            "Across all cost categories"
        )
    with col2:
        metric_card(
            "Baseline Budget Target",
            f'₹{variance["Baseline Budget Target"]:,.0f}',
            "Approved monthly operating budget"
        )
    with col3:
        variance_label = "Over Budget" if variance["Over Budget"] else "Under Budget"
        metric_card(
            "Budget Variance",
            f'{variance["Budget Variance %"]:+.1f}%',
            variance_label
        )
    with col4:
        metric_card(
            "Variance (₹)",
            f'₹{variance["Budget Variance"]:,.0f}',
            "vs. approved budget"
        )
else:
    empty_state("Budget variance data is unavailable.")

# ==========================================================
# COST DISTRIBUTION
# ==========================================================

section_title("Cost Distribution")

if breakdown and breakdown.get("Categories"):
    categories = breakdown["Categories"]
    labels = [category["category"] for category in categories]
    values = [category["total_cost"] for category in categories]

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=["#20a4ff", "#00e676", "#ffb020", "#ff5a6a"]),
            textfont=dict(color="white")
        )]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(color="#c5ccd6"))
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        for category in categories:
            stat_panel(
                category["category"],
                [
                    ("Total Cost", f'₹{category["total_cost"]:,.0f}', ""),
                    ("Share", f'{category["pct_of_total"]}%', "")
                ]
            )
            st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="last-updated">Total facility operating cost: '
        f'₹{breakdown["Total Cost"]:,.0f}</div>',
        unsafe_allow_html=True
    )
else:
    empty_state("Cost breakdown data is unavailable.")

# ==========================================================
# ROI / PAYBACK SIMULATOR
# ==========================================================

section_title("Interactive Operational ROI & Payback Calculator")
st.markdown(
    '<div class="sidebar-card-text" style="color:#8fa3bf;">'
    'Simulate investment payback on energy retrofits, predictive sensors, or space consolidation.'
    '</div>',
    unsafe_allow_html=True
)

sim_col1, sim_col2 = st.columns([1.3, 1])

with sim_col1:
    subsystem_cost = st.slider(
        "Current Subsystem Monthly Cost (₹)", 0, 200000, 25000, step=1000
    )
    efficiency_gain = st.slider(
        "Target AI Efficiency Optimization Gain (%)", 0, 50, 18
    )
    implementation_cost = st.slider(
        "Estimated One-Time Implementation Cost (₹)", 0, 200000, 10000, step=1000
    )

with sim_col2:
    sim_result = post_api(
        "/cost/roi-simulator",
        {
            "subsystem_monthly_cost": subsystem_cost,
            "target_efficiency_gain_pct": efficiency_gain,
            "one_time_implementation_cost": implementation_cost
        }
    )

    if sim_result:
        payback = sim_result["Payback Period (Months)"]
        roi = sim_result["Expected ROI %"]

        stat_panel(
            "Projected Savings",
            [
                ("Annual Recurring Savings", f'₹{sim_result["Annual Recurring Savings"]:,.0f}', ""),
                ("Payback Period", f'{payback} months' if payback is not None else "N/A", ""),
                ("Expected ROI", f'{roi}%' if roi is not None else "N/A", "")
            ]
        )
    else:
        empty_state("Could not reach the ROI simulator endpoint.")

# ==========================================================
# OPTIMIZATION RECOMMENDATIONS
# ==========================================================

section_title("Optimization Recommendations")

if optimizations:
    for item in optimizations:
        recommendation_card(
            f'{item["Category"]} — {item["Priority"]} Priority',
            f'{item["Recommendation"]} (Potential savings: ₹{item["Potential Savings"]:,.0f})'
        )
else:
    empty_state("No active optimization recommendations.")

# ==========================================================
# CROSS-AGENT ORCHESTRATION CENTER
# ==========================================================

section_title("Cross-Agent Orchestration Center")
st.markdown(
    '<div class="sidebar-card-text" style="color:#8fa3bf;">'
    'Live log of automated actions coordinated across the Energy, Maintenance, '
    'Security and Cost agents.'
    '</div>',
    unsafe_allow_html=True
)

orchestration = get_api("/cost/orchestration-log")

if orchestration:
    for action in orchestration:
        status_color = "#00e676" if action["Status"] == "EXECUTED" else "#ffb020"
        status_icon = "✅" if action["Status"] == "EXECUTED" else "🕒"

        st.markdown(
            f'''
            <div style="border:1px solid rgba(255,255,255,0.08); border-radius:10px;
                        padding:12px 16px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-weight:600;">{action["Action ID"]}</span>
                        <span style="color:#8fa3bf;"> · {" + ".join(action["Agents Involved"])}</span>
                    </div>
                    <div style="color:{status_color}; font-weight:600;">
                        {status_icon} {action["Status"]}
                    </div>
                </div>
                <div style="margin-top:6px; color:#c5ccd6;">{action["Recommendation"]}</div>
                <div style="margin-top:4px; color:#8fa3bf; font-size:0.85em;">
                    Cost Impact: ₹{action["Cost Impact"]:,.0f} · Priority: {action["Priority"]}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
else:
    empty_state("No orchestration actions available.")

# ==========================================================
# FACILITY INTELLIGENCE REPORT — FORMATTED RENDERER
# ==========================================================

def render_facility_report(report):
    st.markdown(
        f'<div class="last-updated">Report generated: {report["Generated At"]}</div>',
        unsafe_allow_html=True
    )

    # ---- Cost Summary ----
    summary = report["Cost Summary"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Cost Reduction", f'{summary["Cost Reduction %"]}%', "vs. current spend")
    with col2:
        metric_card("ROI Generated", f'₹{summary["ROI Generated"]:,.0f}', "Identified savings")
    with col3:
        metric_card("Facility Health", f'{summary["Facility Health"]}/100', "Composite score")
    with col4:
        metric_card("Optimizations", summary["Optimizations"], "Active recommendations")

    # ---- Cost Breakdown ----
    st.markdown('<div class="section-title" style="font-size:18px;">Cost Breakdown by Category</div>',
                unsafe_allow_html=True)

    categories = report["Cost Breakdown"]["Categories"]
    breakdown_df = pd.DataFrame([
        {
            "Category": category["category"],
            "Total Cost (₹)": category["total_cost"],
            "Potential Savings (₹)": category["potential_savings"],
            "Share of Total (%)": category["pct_of_total"]
        }
        for category in categories
    ])
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # ---- Optimizations ----
    st.markdown('<div class="section-title" style="font-size:18px;">Optimization Recommendations</div>',
                unsafe_allow_html=True)

    optimizations_df = pd.DataFrame([
        {
            "Category": item["Category"],
            "Priority": item["Priority"],
            "Recommendation": item["Recommendation"],
            "Potential Savings (₹)": item["Potential Savings"]
        }
        for item in report["Optimizations"]
    ])
    st.dataframe(optimizations_df, use_container_width=True, hide_index=True)

    # ---- Facility Health Components ----
    st.markdown('<div class="section-title" style="font-size:18px;">Facility Health Components</div>',
                unsafe_allow_html=True)

    components = report["Facility Health"]["Components"]
    health_cols = st.columns(4)
    for column, (label, score) in zip(health_cols, components.items()):
        with column:
            metric_card(label, f'{score}/100', "")

    # ---- Per-Agent Summaries ----
    st.markdown('<div class="section-title" style="font-size:18px;">Per-Agent Summary</div>',
                unsafe_allow_html=True)

    agent_tabs = st.tabs(["Energy", "Maintenance", "Occupancy", "Security"])

    with agent_tabs[0]:
        st.table(pd.DataFrame(report["Energy"].items(), columns=["Metric", "Value"]))

    with agent_tabs[1]:
        st.table(pd.DataFrame(report["Maintenance"].items(), columns=["Metric", "Value"]))

    with agent_tabs[2]:
        st.table(pd.DataFrame(report["Occupancy"].items(), columns=["Metric", "Value"]))

    with agent_tabs[3]:
        st.table(pd.DataFrame(report["Security"].items(), columns=["Metric", "Value"]))


section_title("Facility Intelligence Report")

st.markdown(
    '<div class="sidebar-card-text" style="color:#8fa3bf;">'
    'Generate a combined report across Energy, Maintenance, Occupancy, Security and Cost agents.'
    '</div>',
    unsafe_allow_html=True
)

if st.button("Generate Full Report"):
    report = get_api("/cost/report")
    if report:
        render_facility_report(report)
    else:
        empty_state("Could not generate the report — check the backend connection.")