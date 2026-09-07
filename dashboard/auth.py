import streamlit as st

VALID_USERS = {
    "admin": "admin123",
    "operator": "operator123",
}


def _inject_login_css():
    st.markdown(
        """
<style>
/* Hide sidebar completely until logged in */
section[data-testid="stSidebar"] { display: none; }

/* Page background: dark navy with subtle skyline-style gradient */
.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(32,164,255,0.08), transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(32,164,255,0.06), transparent 40%),
        linear-gradient(180deg, #0a1220 0%, #0d1830 60%, #0a1220 100%);
}

/* Card wrapper (targets st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border-radius: 18px !important;
    border: none !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    padding: 8px 6px;
}

.login-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: #2f6fed;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin: 6px auto 14px auto;
    box-shadow: 0 6px 16px rgba(47,111,237,0.35);
}

.login-title {
    text-align: center;
    color: #ffffff;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.3px;
    margin-bottom: 4px;
}

.login-subtitle {
    text-align: center;
    color: #6b7688;
    font-size: 12.5px;
    margin-bottom: 18px;
    padding: 0 12px;
}

/* Field labels */
div[data-testid="stTextInput"] label p {
    color: #4a5468 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* Input boxes */
div[data-testid="stTextInput"] input {
    background: #f4f6fa !important;
    color: #14213d !important;
    border: 1px solid #dfe4ec !important;
    border-radius: 9px !important;
    padding: 10px 12px !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #a3adc2 !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] button {
    background: #2f6fed !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    padding: 10px 0 !important;
    margin-top: 6px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #2559c9 !important;
}

.login-footer {
    text-align: center;
    color: #9aa4b8;
    font-size: 11px;
    margin-top: 14px;
}

div[data-testid="stForm"] { border: none; padding: 0; }
</style>
""",
        unsafe_allow_html=True
    )


def _render_login_form():
    _inject_login_css()

    st.markdown("<div style='margin-top:9vh;'></div>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.1, 1])

    with center:
        with st.container(border=True):
            st.markdown(
                '<div class="login-icon">🔒</div>'
                '<div class="login-title">Agentic FacilityOps</div>'
                '<div class="login-subtitle">Sign in with authorized credentials to access '
                'the facility platform</div>',
                unsafe_allow_html=True
            )

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("User ID", placeholder="Enter user ID")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Sign In  →", use_container_width=True)

            if submitted:
                if VALID_USERS.get(username) == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid User ID or Password")

            st.markdown(
                '<div class="login-footer">Unauthorized access is prohibited</div>',
                unsafe_allow_html=True
            )


def require_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        _render_login_form()
        st.stop()


def logout_button():
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.markdown(
                '<div class="sidebar-card">'
                '<div class="sidebar-card-title">SIGNED IN AS</div>'
                f'<div class="sidebar-card-text">👤 {st.session_state.username}</div>'
                '</div>',
                unsafe_allow_html=True
            )
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.pop("username", None)
                st.rerun()