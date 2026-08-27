import streamlit as st
import requests
import plotly.graph_objects as go

# ==========================================================
# CONFIGURATION
# ==========================================================

BACKEND_URL = "http://127.0.0.1:8000"

CACHE_TTL_SECONDS = 5

# ==========================================================
# PAGE CONFIG + CSS (call once, at the top of every page script)
# ==========================================================

def configure_page(title):
    st.set_page_config(
        page_title=title,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()


def inject_css():
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
    background: linear-gradient(180deg, #131f33 0%, #0f1a2b 100%);
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

.healthy-card, .warning-card, .critical-card {
    padding: 20px 22px;
    border-radius: 14px;
    min-height: 118px;
}

.healthy-card {
    background: linear-gradient(180deg, #14301f 0%, #0f2418 100%);
    border: 1px solid #1e7545;
}

.warning-card {
    background: linear-gradient(180deg, #2e2814 0%, #241f0e 100%);
    border: 1px solid #856d22;
}

.critical-card {
    background: linear-gradient(180deg, #301820 0%, #241118 100%);
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
    background: linear-gradient(180deg, #131f33 0%, #0f1a2b 100%);
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
# SIDEBAR BRANDING (call once per page, near the top of the sidebar)
# ==========================================================

def sidebar_branding():
    with st.sidebar:
        st.markdown(
            '<div class="logo"> Agentic '
            '<span class="logo-blue">FacilityOps</span></div>'
            '<div class="sidebar-subtitle">'
            'AI-powered facility monitoring and intelligence</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-card">'
            '<div class="sidebar-card-title">SYSTEM STATUS</div>'
            '<div class="sidebar-card-text">'
            '<span class="status-dot"></span>Monitoring Active</div>'
            '</div>'
            '<div class="sidebar-card">'
            '<div class="sidebar-card-title">BACKEND</div>'
            '<div class="sidebar-card-text">FastAPI</div>'
            '</div>'
            '<div class="sidebar-card">'
            '<div class="sidebar-card-title">PLATFORM</div>'
            '<div class="sidebar-card-text">Agentic FacilityOps AI</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ==========================================================
# HTML HELPERS
# ==========================================================

def metric_card(title, value, description):
    st.markdown(
        '<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-description">{description}</div>'
        '</div>',
        unsafe_allow_html=True
    )


def health_card(css_class, label, number):
    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="maintenance-label">{label}</div>'
        f'<div class="maintenance-number">{number}</div>'
        '</div>',
        unsafe_allow_html=True
    )


def alert_card(title, text):
    st.markdown(
        '<div class="alert-card">'
        f'<div class="alert-title">⚠ {title}</div>'
        f'<div class="alert-text">{text}</div>'
        '</div>',
        unsafe_allow_html=True
    )


def recommendation_card(title, items):
    if isinstance(items, list):
        recommendation_text = (
            "<br>".join(f"• {item}" for item in items)
            if items else "No recommendation."
        )
    else:
        recommendation_text = str(items)

    st.markdown(
        '<div class="recommendation-card">'
        f'<div class="recommendation-title">🤖 {title}</div>'
        f'<div class="recommendation-text">{recommendation_text}</div>'
        '</div>',
        unsafe_allow_html=True
    )


def empty_state(message):
    st.markdown(f'<div class="empty-state">{message}</div>', unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


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
            number=dict(suffix="", font=dict(color="white", size=34)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#5c7291",
                          tickfont=dict(color="#5c7291", size=10)),
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

    st.markdown(
        f'<div class="stat-panel"><div class="stat-panel-title">{title}</div>{rows_html}</div>',
        unsafe_allow_html=True
    )


# ==========================================================
# API CLIENT
# ==========================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_api(endpoint, params=None):
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"FastAPI connection error on {endpoint}: {error}")
        return None


def post_api(endpoint, payload):
    try:
        response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
        response.raise_for_status()
        clear_api_cache()
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"FastAPI connection error on {endpoint}: {error}")
        return None


def delete_api(endpoint):
    try:
        response = requests.delete(f"{BACKEND_URL}{endpoint}", timeout=30)
        response.raise_for_status()
        clear_api_cache()
        return True
    except requests.exceptions.RequestException as error:
        st.error(f"FastAPI connection error on {endpoint}: {error}")
        return False


def clear_api_cache():
    """Call after any write (POST/DELETE) so the next read is fresh."""
    get_api.clear()