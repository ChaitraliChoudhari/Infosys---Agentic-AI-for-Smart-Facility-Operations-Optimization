import streamlit as st
from shared import configure_page, sidebar_branding, empty_state, section_title

configure_page("Agentic FacilityOps AI")
sidebar_branding()

st.markdown('<div class="main-title">Agentic FacilityOps AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="main-subtitle">Pick a module from the sidebar to get started.</div>',
    unsafe_allow_html=True
)

section_title("Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/energy intelligence.py", label="Energy Intelligence", icon="⚡")
    empty_state("Consumption trends, alerts, and AI-driven energy recommendations.")

    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

    st.page_link("pages/security_page.py", label="Security Agent", icon="🛡️")
    empty_state("Access control, unauthorized access alerts, CCTV events, and incident investigation.")

with col2:
    st.page_link("pages/predictive maintenance.py", label="Predictive Maintenance", icon="🔧")
    empty_state("Equipment health scoring, alerts, and maintenance scheduling.")

with col3:
    st.page_link("pages/occupancy_page.py", label="Occupancy Agent", icon="🏢")
    empty_state("Occupancy heatmaps, overcrowding alerts, utilization, and ML forecasting.")