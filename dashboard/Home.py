import streamlit as st
from shared import configure_page, sidebar_branding, empty_state, section_title
from auth import require_login, logout_button      # ← add this import

configure_page("Agentic FacilityOps AI")
require_login()          # ← add this line, before anything else renders

sidebar_branding()
logout_button()          # optional: shows logout button in sidebar

st.markdown('<div class="main-title">Agentic FacilityOps AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="main-subtitle">Pick a module from the sidebar to get started.</div>',
    unsafe_allow_html=True
)

section_title("Modules")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_energy intelligence.py", label="Energy Intelligence", icon="⚡")
    empty_state("Consumption trends, alerts, and AI-driven energy recommendations.")

    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

    st.page_link("pages/4_security_page.py", label="Security Agent", icon="🛡️")
    empty_state("Access control, unauthorized access alerts, CCTV events, and incident investigation.")

with col2:
    st.page_link("pages/2_predictive maintenance.py", label="Predictive Maintenance", icon="🔧")
    empty_state("Equipment health scoring, alerts, and maintenance scheduling.")

with col3:
    st.page_link("pages/3_occupancy_page.py", label="Occupancy Agent", icon="🏢")
    empty_state("Occupancy heatmaps, overcrowding alerts, utilization, and ML forecasting.")

with col4:
    st.page_link("pages/5_cost_page.py", label="Cost Optimization", icon="💰")
    empty_state("Cost distribution, ROI tracking, optimization recommendations, and facility intelligence reports.")