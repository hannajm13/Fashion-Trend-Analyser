import streamlit as st
import streamlit.components.v1 as components
from supabase import create_client, Client
from utils import trend_explorer

from utils.trend_prediction import (
    render_trend_prediction_page,
    get_prediction_data,
)

from utils.trend_evolution import (
    load_trend_data,
    create_time_series_chart,
    create_trend_units_time_series,
    create_attribute_time_series,
    clean_entities_dataframe,
    render_wordcloud,
    create_weighted_sentiment_fig,
    ATTRIBUTE_LABELS,
    WORDCLOUD_LABELS,
    SENTIMENT_ATTR_LABELS,
    SENTIMENT_MIN_FREQ,
    DEFAULT_SENTIMENT_TOP_N,
    GRAPH_INFO,
    DEFAULT_TOP_N,
    SUBPAGE_DESCRIPTIONS,
)

# ── FIX 1: set_page_config ONCE, at the very top, before anything else ──
st.set_page_config(
    page_title="Fashion Trend Analyser",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── FIX 3: Cache trend data so it doesn't reload on every rerun ──
#@st.cache_data
#def get_trend_data():
    #return load_trend_data("data/entities_final.xlsx", "data/entities_with_sentiment_final.xlsx")



def get_supabase() -> Client:
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    return st.session_state.supabase_client



#def get_trend_data_cached(user_id, _auth_status):
    # = get_supabase()
    #return load_trend_data(supabase, user_id)

@st.cache_data(show_spinner="Getting the data...")
def get_trend_data_cached(user_id, _auth_status):
    try:
        supabase = get_supabase()
        return load_trend_data(supabase, user_id)

    except Exception:
        return load_trend_data(
            "data/entities_final.xlsx",
            "data/entities_with_sentiment_final.xlsx",
        )


def get_current_user_id():
    # Login has been removed. The app always reads the base dataset only.
    return None


def get_trend_data():
    user_id = get_current_user_id()
    auth_status = st.session_state.get("auth_status", "guest")
    return get_trend_data_cached(user_id, auth_status)


def init_auth_state():
    # Login has been removed. The app should open directly on HOME.
    # Keep these values fixed so Supabase queries use the public/base dataset.
    st.session_state.auth_status = "guest"
    st.session_state.user_id = None
    st.session_state.user_email = None


def clear_auth_state():
    keys_to_clear = [
        "auth_status",
        "user_id",
        "user_email",
        "supabase_client",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]



def logout_user():
    try:
        supabase = get_supabase()
        supabase.auth.sign_out()
    except Exception:
        pass

    clear_auth_state()
    st.session_state.auth_status = "not_started"
    st.rerun()


def continue_as_guest():
    clear_auth_state()
    st.session_state.auth_status = "guest"
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.rerun()


def login_user(email, password):
    supabase = get_supabase()

    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password,
    })

    if response.user is None:
        raise Exception("Login failed. Please check your email and password.")

    st.session_state.auth_status = "authenticated"
    st.session_state.user_id = response.user.id
    st.session_state.user_email = response.user.email

    st.cache_data.clear()
    st.rerun()


def signup_user(email, password):
    supabase = get_supabase()

    response = supabase.auth.sign_up({
        "email": email,
        "password": password,
    })

    return response


# ── FIX 4: Cache chart generation so number_input changes don't re-render everything ──
@st.cache_data
def get_time_series_chart(df, text_col, top_n, title, season_entities_df=None):
    return create_time_series_chart(df, text_col=text_col, top_n=top_n, title=title, season_entities_df=season_entities_df)

@st.cache_data
def get_trend_units_time_series(df, ngram_type, top_n, title, season_entities_df=None):
    return create_trend_units_time_series(df, ngram_type=ngram_type, top_n=top_n, title=title, season_entities_df=season_entities_df)

@st.cache_data
def get_attribute_time_series(df, label, top_n, title, season_entities_df=None):
    return create_attribute_time_series(df, label=label, top_n=top_n, title=title, season_entities_df=season_entities_df)

@st.cache_data
def get_wordcloud(df, label):
    return render_wordcloud(df, label)

@st.cache_data
def get_clean_entities(df):
    return clean_entities_dataframe(df)


# ── FIX 5: All CSS in ONE st.markdown block ──
# ── CSS BLOCK ───────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #544541;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

.block-container {
    padding-top: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
    max-width: 100%;
}

/* MAIN TITLE HEADER - keep same visual style */
.top-header {
    background-color: #40322f;
    color: #c2dcff;
    text-align: center;
    font-weight: 800;
    font-size: 22px;
    padding: 18px 0;
    letter-spacing: 1px;
}

.top-header-link,
.top-header-link:visited,
.top-header-link:hover,
.top-header-link:active {
    text-decoration: none !important;
    display: block;
    color: inherit !important;
}

/* NAVIGATION */
.nav-bar {
    background-color: #c2dcff;
    display: flex;
    justify-content: space-around;
    align-items: center;
    height: 65px;
}

.nav-item {
    color: #312421;
    font-size: 22px;
    font-weight: 800;
    text-decoration: none;
    padding: 18px 55px;
    border-radius: 25px;
}

.nav-active {
    background-color: #f5f3f5;
}

.main-area {
    padding: 10px;
}



/* TREND EVOLUTION TEXT */
.subpage-description {
    color: #544541;
    font-size: 15px;
    font-weight: 700;
    text-align: center;
    max-width: 900px;
    margin: 0 auto 25px auto;
    line-height: 1.5;
}

.graph-header-title {
    color: #c2dcff;
    font-size: 20px;
    font-weight: 800;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 4px;
}

.graph-header-desc {
    color: #f5f3f5;
    font-size: 14px;
    font-weight: 500;
    text-align: center;
    margin-bottom: 12px;
}

/* BUTTONS */
/* BUTTONS */
/* BUTTONS - base */
            
/* ════════════════════════════════════════════════════════════════
   BUTTONS — GENERAL BASE STYLE
════════════════════════════════════════════════════════════════ */

div[data-testid="stButton"] > button {
    background-color: #c2dcff;
    color: #312421;
    font-size: 26px;
    font-weight: 800;
    border: none;
    height: 72px;
    border-radius: 0px;
    transition: background-color 0.15s ease, color 0.15s ease;
    margin-top: 10px;
}

div[data-testid="stButton"] > button p {
    font-weight: 800 !important;
}

div[data-testid="stButton"] > button:hover {
    background-color: #f5f3f5 !important;
    color: #312421 !important;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #f5f3f5 !important;
    color: #312421 !important;
    border: 2px solid #c1dffa !important;
}


/* ════════════════════════════════════════════════════════════════
   MAIN NAV POLKA DOT BAND
════════════════════════════════════════════════════════════════ */

div[class*="st-key-main_nav_band_dots_"] {
    position: relative;
    padding: 25px 0 20px;
}

div[class*="st-key-main_nav_band_dots_"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        radial-gradient(var(--dot-color, rgba(49,36,33,0.16)) 2.5px, transparent 2.5px),
        radial-gradient(var(--dot-color, rgba(49,36,33,0.16)) 2.5px, transparent 2.5px);
    background-size: 24px 24px;
    background-position: 0 0, 12px 12px;
    pointer-events: none;
}

div[class*="st-key-main_nav_band_dots_"] > div {
    position: relative;
    z-index: 1;
}

/* Per-page dot tint */
.st-key-main_nav_band_dots_analysis {
    --dot-color: rgba(180, 205, 230, 0.6);
}

.st-key-main_nav_band_dots_forecast {
    --dot-color: rgba(98, 142, 144, 0.45);
}

.st-key-main_nav_band_dots_explorer {
    --dot-color: rgba(135, 167, 179, 0.5);
}


/* ════════════════════════════════════════════════════════════════
   MAIN NAV BUTTONS
════════════════════════════════════════════════════════════════ */

.st-key-side_nav_home div[data-testid="stButton"] > button {
    background-color: #c1dffa !important;
    font-size: 32px !important;
    height: 58px !important;
    margin-top: 20px !important;
}   

.st-key-nav_trend_evolution div[data-testid="stButton"] > button {
    background-color: #c1dffa !important;
    font-size: 32px !important;
    height: 58px !important;
    margin-top: 20px !important;
}

.st-key-nav_trend_forecast div[data-testid="stButton"] > button {
    background-color: #a6c8eb !important;
    font-size: 32px !important;
    height: 58px !important;
    margin-top: 20px !important;
}

.st-key-nav_trend_explorer div[data-testid="stButton"] > button {
    background-color: #95b3cf !important;
    font-size: 32px !important;
    height: 58px !important;
    margin-top: 20px !important;
}

/* Main nav hover */
.st-key-side_nav_home[data-testid="stButton"] > button:hover,
.st-key-nav_trend_evolution div[data-testid="stButton"] > button:hover,
.st-key-nav_trend_forecast div[data-testid="stButton"] > button:hover,
.st-key-nav_trend_explorer div[data-testid="stButton"] > button:hover {
    background-color: #f5f3f5 !important;
    color: #312421 !important;
}


/* ════════════════════════════════════════════════════════════════
   SUBPAGE DRAWER / BUTTON PANEL
════════════════════════════════════════════════════════════════ */

/* Trend Evolution subpage panel */
.st-key-subnav_drawer_trend_evolution,
.st-key-subpage_button_band_trend_evolution {
    background-color: #e1f0ff !important;
    border-radius: 0 0 16px 16px;
    padding: 14px 22px 18px;
    margin: -4px 0 20px;
}

/* Trend Forecast subpage panel */
.st-key-subnav_drawer_trend_forecast,
.st-key-subpage_button_band_trend_forecast {
    background-color: #c1dffa !important;
    border-radius: 0 0 16px 16px;
    padding: 14px 22px 18px;
    margin: -4px 0 20px;
}


/* ════════════════════════════════════════════════════════════════
   SUBPAGE BUTTONS — FINAL SIZE CONTROL
   Change only these values if you want them bigger/smaller.
════════════════════════════════════════════════════════════════ */

div[class*="st-key-subnav_drawer_"] div[data-testid="stButton"] > button,
div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button {
    background-color: #766161 !important;
    color: #f5f3f5 !important;
    height: 28px !important;
    min-height: 28px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    padding: 0 18px !important;
    white-space: nowrap !important;
    border: none !important;
}

/* Streamlit puts button text inside p / markdown containers, so force those too */
div[class*="st-key-subnav_drawer_"] div[data-testid="stButton"] > button p,
div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button p,
div[class*="st-key-subnav_drawer_"] div[data-testid="stButton"] > button div[data-testid="stMarkdownContainer"] p,
div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button div[data-testid="stMarkdownContainer"] p {
    font-size: 15px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    color: inherit !important;
}

/* Subpage hover */
div[class*="st-key-subnav_drawer_"] div[data-testid="stButton"] > button:hover,
div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button:hover {
    background-color: #f5f3f5 !important;
    color: #544541 !important;
    border: none !important;
}

/* Active subpage button */
div[class*="st-key-subnav_drawer_"] div[data-testid="stButton"] > button[kind="primary"],
div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #f5f3f5 !important;
    color: #312421 !important;
    border: 1.5px solid #766161 !important;
}

/* Active subpage button text */
div[class*="st-key-subnav_drawer_"] div[data-testid="stButton"] > button[kind="primary"] p,
div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button[kind="primary"] p {
    color: #312421 !important;
}




            /* Clickable main header as a Streamlit button.
   This keeps the same visual style but stays inside the same page/session. */
            
/* FIX MAIN HEADER TO TOP WITH NO TOP GAP */
            
[data-testid="stHeader"] {
    height: 0rem !important;
    background: transparent !important;
}
            
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}


.st-key-header_home_button div[data-testid="stButton"] > button {
    background-color: #3b2f2d !important;
    color: #c2dcff !important;
    text-align: center !important;
    font-weight: 800 !important;
    font-size: 17px !important;
    margin-top: 0px !important;
    padding: 10px 0 !important;
    height: auto !important;
    min-height: 42px !important;
    border-radius: 0px !important;
    border: none !important;
    letter-spacing: 1px !important;
    width: 100% !important;
}

.st-key-header_home_button div[data-testid="stButton"] > button:hover {
    background-color: #312421 !important;
    color: #c2dcff !important;
    border: none !important;
}

.st-key-header_home_button div[data-testid="stButton"] > button p {
    color: #c2dcff !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
}
            

/* Thinner header, no gap above it */
.st-key-header_home_button {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}

.st-key-header_home_button div[data-testid="stButton"] > button {
    min-height: 72px !important;
    height: 72px !important;
    padding: 8px 0 !important;
    font-size: 15px !important;
}

/* Kill Streamlit's default vertical gap between blocks so nothing floats above the header */
[data-testid="stVerticalBlock"]:has(> div > .st-key-header_home_button) {
    gap: 0rem !important;
}



/* NUMBER INPUT */
div[data-testid="stNumberInput"] label {
    color: #c2dcff !important;
    font-size: 18px !important;
    font-weight: 800 !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #f5f3f5 !important;
    color: #312421 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* Center and resize graph number inputs */
div[data-testid="stNumberInput"] {
    margin-left: auto !important;
    margin-right: auto !important;
}

div[data-testid="stNumberInput"] label {
    text-align: center !important;
    display: block !important;
    font-size: 12px !important;
}

div[data-testid="stNumberInput"] input {
    height: 32px !important;
    font-size: 12px !important;
    text-align: center !important;
}

/* STRONGER SEASON TOGGLE FIX */
div[data-testid="stToggle"] {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
}

/* Streamlit nests toggle text inside markdown containers, so target all levels */
div[data-testid="stToggle"] label,
div[data-testid="stToggle"] label *,
div[data-testid="stToggle"] p,
div[data-testid="stToggle"] span,
div[data-testid="stToggle"] div[data-testid="stMarkdownContainer"],
div[data-testid="stToggle"] div[data-testid="stMarkdownContainer"] *,
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] *,
div[data-testid="stWidgetLabel"],
div[data-testid="stWidgetLabel"] * {
    color: #c2dcff !important;
    font-weight: 600 !important;
}

/* Make Season label readable */
div[data-testid="stToggle"] label p {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Optional helper if you are using this spacer before toggle */
.season-toggle-pad {
    height: 58px;
}
            
/* Make Season toggle label bold */
div[data-testid="stToggle"] label,
div[data-testid="stToggle"] label *,
div[data-testid="stToggle"] p,
div[data-testid="stToggle"] span {
    font-weight: 600 !important;
    color: var(--soft-blue) !important;
}
            
/* Wide, left-aligned textbox — scoped so it doesn't affect your small graph number inputs */
div[class*="st-key-topn_wide_"] div[data-testid="stNumberInput"] {
    width: 100% !important;
    max-width: 100% !important;
}
div[class*="st-key-topn_wide_"] div[data-testid="stNumberInput"] label {
    text-align: left !important;
    font-size: 16px !important;
    font-weight: 900 !important;
}
div[class*="st-key-topn_wide_"] div[data-testid="stNumberInput"] input {
    text-align: left !important;
    height: 42px !important;
    font-size: 15px !important;
}

/* HOME PAGE */
.home-container {
    width: 100%;
    margin: 0;
    padding: 0;
}

/* Full-screen welcome section */
.home-hero {
    min-height: calc(100vh - 130px);
    background-color: #312421;
    color: #f5f3f5;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 90px 70px;
    box-sizing: border-box;
    margin-top: 28px; 
}

.home-hero-inner {
    max-width: 1050px;
    text-align: center;
}

.home-hero h1 {
    margin: 0 0 18px 0;
    font-size: 35px;
    font-weight: 900;
    color: #c2dcff;
    letter-spacing: 0.5px;
}

.home-subtitle {
    color: #c2dcff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 34px;
}

.home-text {
    font-size: 18px;
    line-height: 1.8;
    font-weight: 500;
    color: #f5f3f5;
    max-width: 920px;
    margin: 0 auto;
}

.home-text b {
    color: #c2dcff;
    font-weight: 900;
    font-size: 22px;
}

/* Content after hero */
.home-content {
    max-width: 1180px;
    margin: 0 auto;
    padding: 35px 22px 55px 22px;
}

.home-section-title {
    color: #c2dcff;
    font-size: 24px;
    font-weight: 900;
    text-align: center;
    margin: 30px 0 18px 0;
}

/* Entity cards */
.entity-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 14px;
}

.entity-card {
    background-color: #312421;
    border: 2px solid #c2dcff;
    border-radius: 22px;
    padding: 17px 19px;
    min-height: 125px;
    color: #c2dcff;
    box-shadow: 0 5px 12px rgba(49,36,33,0.14);
}

.entity-card-title {
    color: #c2dcff;
    font-size: 17px;
    font-weight: 900;
    margin-bottom: 8px;
}

.entity-card-desc {
    color: #f5f3f5;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.5;
}

/* How-to cards */
.how-card {
    background-color: #f5f3f5;
    border: 2px solid #c2dcff;
    border-radius: 24px;
    padding: 22px 26px;
    color: #312421;
    margin-bottom: 12px;
}

.how-card b {
    color: #312421;
    font-size: 17px;
    font-weight: 900;
}

.how-card p {
    margin: 8px 0 0 0;
    font-size: 15px;
    line-height: 1.55;
    font-weight: 500;
    color: #312421;
}
            

.stButton > button[kind="primary"],
.stButton > button[kind="primary"] * {
    color: #312421 !important;
}

.stButton > button[kind="primary"] {
    background-color: #c2dcff !important;
    border: 1px solid #c2dcff !important;
}


/* LOGIN / LOGOUT BUTTON OVERLAYED ON EXISTING HEADER */
.st-key-header_auth_button {
    position: relative !important;
    z-index: 1000 !important;
    width: fit-content !important;
    margin-left: auto !important;
    margin-right: 38px !important;
    margin-top: -70px !important;
    margin-bottom: 12px !important;
    padding: 0 !important;
}

.st-key-header_auth_button div[data-testid="stButton"] {
    width: auto !important;
    margin: 0 !important;
    padding: 0 !important;
}

.st-key-header_auth_button div[data-testid="stButton"] > button {
    background-color: #312421 !important;
    color: #c2dcff !important;
    border: 1.5px solid #c2dcff !important;
    border-radius: 0px !important;
    width: 92px !important;
    min-width: 92px !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 12px !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    letter-spacing: 0px !important;
}

.st-key-header_auth_button div[data-testid="stButton"] > button p {
    color: #c2dcff !important;
    font-size: 14px !important;
    font-weight: 800 !important;
}

.st-key-header_auth_button div[data-testid="stButton"] > button:hover {
    background-color: #312421 !important;
    color: #c2dcff !important;
    border: 1.5px solid #c2dcff !important;
}



            /* MOVE POLKA DOT BAND + SUBPAGE BUTTONS UP */
div[class*="st-key-subpage_button_band_"],
div[class*="st-key-subnav_drawer_"],
div[class*="st-key-main_nav_band_dots_"] {
    margin-top: -18px !important;
}

/* Kill Streamlit's own top padding on the outer app view container,
   not just .block-container */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0rem !important;
    padding-left: var(--side-menu-width) !important;
}

[data-testid="stMain"] {
    padding-top: 0rem !important;
    padding-left: 0rem !important;   /* no longer stacking on top of .main's padding */
}

.block-container {
    padding-top: 0rem !important;
    padding-left: 0rem !important;   /* same here */
    padding-right: 0rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}

/* Remove the default vertical gap Streamlit puts between EVERY block.
   This is almost always the real cause of "mystery space" between
   custom containers. Safe to zero globally since you're spacing
   things manually with padding/margin anyway. */
[data-testid="stVerticalBlock"] {
    gap: 0rem !important;
}

[data-testid="stElementContainer"],
[data-testid="element-container"] {
    margin: 0 !important;
}

/* Belt-and-suspenders: target the header + band specifically in case
   the global gap rule above gets overridden by a later Streamlit build */
           
div:has(> div > .st-key-header_home_button) {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    gap: 0 !important;
}

div:has(> div[class*="st-key-subpage_button_band_"]) {
    margin-top: 0 !important;
    padding-top: 0 !important;
}  



@media (max-width: 900px) {
    .entity-grid {
        grid-template-columns: 1fr;
    }

    .home-hero {
        min-height: calc(100vh - 130px);
        padding: 38px 24px;
    }

    .home-hero h1 {
        font-size: 32px;
    }

    .home-text {
        font-size: 16px;
    }

}
</style>
""", unsafe_allow_html=True)

# ── NEW UI OVERRIDES: fixed left menu + right content panel ───────────────
st.markdown("""
<style>
:root {
    --side-menu-width: 178px;
    --app-bg: #544541;
    --deep-brown: #312421;
    --soft-blue: #c2dcff;
    --off-white: #f5f3f5;
    --off-white-rgba: rgba(245, 243, 245, 0.80);
}

html {
    scroll-behavior: smooth;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--app-bg) !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Move the whole Streamlit page into the right-side content panel. */
.block-container {
    padding-top: 0rem !important;
    padding-left: var(--side-menu-width) !important;
    padding-right: 0rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}

/* Fixed vertical menu on the left. */
.left-sidebar-menu {
    position: fixed;
    left: 0;
    top: 0;
    width: var(--side-menu-width);
    height: 100vh;
    z-index: 9999;
    background-color: var(--deep-brown);
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    padding: 20px 7px 14px 7px;
    border-right: 3px solid #f5f3f5;
}

.side-nav-link,
.side-nav-link:visited {
    display: flex;
    align-items: center;
    min-height: 32px;
    padding: 0 14px;
    margin: 0 0 7px 0;
    text-decoration: none !important;
    color: var(--soft-blue) !important;
    font-size: 13px;
    font-weight: 900;
    letter-spacing: 0.55px;
    border-radius: 0px;
    transition: background-color 0.16s ease, color 0.16s ease;
}

.side-nav-link:hover,
.side-nav-link.active {
    background-color: var(--soft-blue);
    color: var(--deep-brown) !important;
}

.side-nav-back,
.side-nav-back:visited {
    margin-top: auto;
    color: var(--soft-blue) !important;
    text-decoration: none !important;
    font-size: 11px;
    font-weight: 900;
    padding: 7px 14px;
    transition: background-color 0.16s ease, color 0.16s ease;
}

.side-nav-back:hover {
    background-color: var(--soft-blue);
    color: var(--deep-brown) !important;
}
            




/* Streamlit-button sidebar*/
div[class*="st-key-left_sidebar_menu"] {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    width: var(--side-menu-width) !important;
    height: 100vh !important;
    z-index: 9999 !important;
    background-color: var(--deep-brown) !important;
    box-sizing: border-box !important;
    padding: 20px 7px 14px 7px !important;
    border-right: 1px solid rgba(194, 220, 255, 0.06) !important;
}

div[class*="st-key-left_sidebar_menu"] div[data-testid="stButton"] {
    margin: 0 0 7px 0 !important;
    width: 100% !important;
}

div[class*="st-key-left_sidebar_menu"] div[data-testid="stButton"] > button {
    min-height: 36px !important;
    height: 36px !important;
    width: 100% !important;
    border-radius: 0px !important;
    border: none !important;
    background-color: transparent !important;
    color: var(--soft-blue) !important;
    padding: 0 10px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    letter-spacing: 0.55px !important;
}

div[class*="st-key-left_sidebar_menu"] div[data-testid="stButton"] > button p {
    color: inherit !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    letter-spacing: 0.55px !important;
    text-align: left !important;
}

div[class*="st-key-left_sidebar_menu"] div[data-testid="stButton"] > button:hover,
div[class*="st-key-left_sidebar_menu"] div[data-testid="stButton"] > button[kind="primary"] {
    background-color: var(--soft-blue) !important;
    color: var(--deep-brown) !important;
}

div[class*="st-key-left_sidebar_menu"] div[data-testid="stButton"] > button:hover p,
div[class*="st-key-left_sidebar_menu"] div[data-testid="stButton"] > button[kind="primary"] p {
    color: var(--deep-brown) !important;
}

/* Keep sidebar fixed */
div[class*="st-key-left_sidebar_menu"] {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    width: var(--side-menu-width) !important;
    height: 100vh !important;
}

/* Remove the manual spacer */
div[class*="st-key-sidebar_bottom_spacer"] {
    height: 0px !important;
    min-height: 0px !important;
}

/* Force Back to Top to bottom of sidebar */
.sidebar-back-top {
    position: fixed !important;
    left: 0 !important;
    bottom: 14px !important;
    width: var(--side-menu-width) !important;
    box-sizing: border-box !important;

    display: block !important;
    color: var(--soft-blue) !important;
    text-decoration: none !important;
    font-size: 13px !important;
    font-weight: 900 !important;
    padding: 12px 14px !important;
    text-align: left !important;

    background-color: transparent !important;
    transition: background-color 0.16s ease, color 0.16s ease;
}

.sidebar-back-top:hover {
    background-color: var(--soft-blue) !important;
    color: var(--deep-brown) !important;
}

/* Sidebar border*/
div[class*="st-key-left_sidebar_menu"] {
    border-right: 1px solid #3b2f2d !important;
}


/* Header remains clickable, but only spans the second/right panel now. */
.st-key-header_home_button div[data-testid="stButton"] > button {
    min-height: 88px !important;
    height: 88px !important;
    border-radius: 0px !important;
    background-color: #40322f important!
}

/* Keep login/logout in the same top-right position of the right panel. */
.st-key-header_auth_button {
    margin-top: -72px !important;
    margin-bottom: 16px !important;
}
            

            

/* Subpage button band with staggered polka dots. */
div[class*="st-key-subpage_button_band_"] {
    position: relative;
    overflow: hidden;
    background-color: var(--app-bg) !important;
    padding: 34px 34px 18px 34px;
    margin: 0 !important;
    min-height: 96px;
}

div[class*="st-key-subpage_button_band_"]::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image:
        radial-gradient(circle, rgba(194, 220, 255, 0.72) 0 5.5px, transparent 6px),
        radial-gradient(circle, rgba(194, 220, 255, 0.72) 0 5.5px, transparent 6px);
    background-size: 62px 48px;
    background-position: 0 0, 31px 24px;
}

div[class*="st-key-subpage_button_band_"] > div {
    position: relative;
    z-index: 1;
}

/* Subpage buttons: centred pills on top of the polka dot band. */
div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button {
    height: 44px !important;
    min-height: 44px !important;
    border-radius: 999px !important;
    background-color: var(--soft-blue) !important;
    color: var(--deep-brown) !important;
    border: none !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button p {
    color: var(--deep-brown) !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button[kind="primary"] {
    background-color: var(--off-white) !important;
    color: var(--deep-brown) !important;
    border: 2px solid var(--soft-blue) !important;
}

div[class*="st-key-subpage_button_band_"] div[data-testid="stButton"] > button:hover {
    background-color: var(--off-white) !important;
    color: var(--deep-brown) !important;
    border: 2px solid var(--soft-blue) !important;
}

        
/* Description sits below the dotted band. */
.st-key-subpage_description_panel {
    background-color: #4b3d3a !important;
    margin: 0 !important;
    border-top: 1.5px solid #716e74 !important;
    border-bottom: 1.5px solid #716e74 !important;
    padding-top: 25px !important;
    padding-bottom: 43px !important;   /* increase this until it looks centered */
    padding-left: 28px !important;
    padding-right: 28px !important;
}

.subpage-description {
    color: var(--off-white) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-align: center !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    line-height: 1.45 !important;
}
        
/* Pull the polka dot band flush under the header */
div[class*="st-key-subpage_button_band_"] {
    margin-top: 0px !important;
    padding: 16px 34px 16px 34px !important;   /* symmetric top/bottom -> centers buttons in the band */
}
            
/* Smaller, denser dots 
div[class*="st-key-subpage_button_band_"]::before {
    background-image:
        radial-gradient(circle, rgba(194, 220, 255, 0.72) 0 3px, transparent 3.5px),
        radial-gradient(circle, rgba(194, 220, 255, 0.72) 0 3px, transparent 3.5px);
    background-size: 34px 26px;
    background-position: 0 0, 17px 13px;
} */

div[class*="st-key-subpage_button_band_"]::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;

    opacity: 0.32; /* change this to control transparency */

    background-image:
        radial-gradient(circle, rgb(194, 220, 255) 0 3px, transparent 3.5px),
        radial-gradient(circle, rgb(194, 220, 255) 0 3px, transparent 3.5px);

    background-size: 34px 26px;
    background-position: 0 0, 17px 13px;
}

 


            

            


/* Chart/title area now belongs to the right content panel, not the full browser. */
.graph-header-title {
    color: var(--soft-blue) !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    text-align: center !important;
    margin-top: 30px !important;
    margin-bottom: 6px !important;
}

.graph-header-desc {
    color: var(--off-white) !important;
    font-size: 14px !important;
    font-weight: 650 !important;
    text-align: center !important;
    max-width: 900px !important;
    margin: 0 auto 14px auto !important;
}

div[data-testid="stPlotlyChart"] {
    width: calc(100vw - var(--side-menu-width) - 50px) !important;
    max-width: calc(100vw - var(--side-menu-width) - 50px) !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-bottom: 36px !important;
}
            
div[data-testid="stElementContainer"][height][overflow] {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
}
            
/* Remove internal vertical scrollbars from Plotly charts */
div[data-testid="stPlotlyChart"] {
    overflow: visible !important;
    overflow-y: visible !important;
    height: auto !important;
}

div[data-testid="stPlotlyChart"] > div {
    overflow: visible !important;
    overflow-y: visible !important;
    height: auto !important;
}

div[data-testid="stPlotlyChart"] .js-plotly-plot,
div[data-testid="stPlotlyChart"] .plot-container,
div[data-testid="stPlotlyChart"] .svg-container {
    overflow: visible !important;
}

/* Resize the small textbox above every chart. */
div[data-testid="stNumberInput"] {
    max-width: 360px !important;
}

div[data-testid="stNumberInput"] label {
    color: var(--soft-blue) !important;
    font-size: 16px !important;
    font-weight: 900 !important;
}

div[data-testid="stNumberInput"] input {
    height: 42px !important;
    border-radius: 12px !important;
}

div[data-testid="stNumberInput"] {
    width: 100% !important;
    max-width: 100% !important;
}

div[data-testid="stNumberInput"] label {
    text-align: center !important;
    display: block !important;
    font-size: 12px !important;
    font-weight: 800 !important;
}

div[data-testid="stNumberInput"] input {
    width: 100% !important;
    height: 32px !important;
    font-size: 12px !important;
    text-align: center !important;
}

/* Word cloud iframe should follow the second panel width. */
iframe {
    max-width: calc(100vw - var(--side-menu-width) - 70px) !important;
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

            /* Thinner header, no gap above it */
.st-key-header_home_button {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}

.st-key-header_home_button div[data-testid="stButton"] > button {
    min-height: 60px !important;
    height: 60px !important;
    padding: 8px 0 !important;
    font-size: 15px !important;
}

/* Kill Streamlit's default vertical gap between blocks so nothing floats above the header */
[data-testid="stVerticalBlock"]:has(> div > .st-key-header_home_button) {
    gap: 0rem !important;
}
            

/* Explicit spacing restored after global Streamlit gaps were removed. */
div[class*="st-key-topn_wide_"] {
    margin-top: 10px !important;
    margin-bottom: 22px !important;
}

div[data-testid="stNumberInput"] {
    margin-bottom: 18px !important;
}

div[data-testid="stPlotlyChart"] {
    margin-top: 16px !important;
    margin-bottom: 48px !important;
}
            

/* Trend Forecast page only */
div[class*="st-key-forecast_topn_wide_"] {
    padding-left: 35px !important;
    padding-right: 25px !important;
    box-sizing: border-box !important;
    width: 100% !important;
}

div[class*="st-key-forecast_topn_wide_"] div[data-testid="stNumberInput"] {
    width: 100% !important;
    max-width: 100% !important;
}

div[class*="st-key-forecast_topn_wide_"] div[data-testid="stNumberInput"] label {
    text-align: left !important;
    font-size: 16px !important;
    font-weight: 900 !important;
}

div[class*="st-key-forecast_topn_wide_"] div[data-testid="stNumberInput"] input {
    text-align: left !important;
    height: 42px !important;
    font-size: 15px !important;
}
            


            
iframe {
    margin-top: 18px !important;
    margin-bottom: 48px !important;
}

@media (max-width: 900px) {
    :root {
        --side-menu-width: 132px;
    }
    .side-nav-link, .side-nav-back {
        font-size: 10px;
        padding-left: 10px;
        padding-right: 10px;
    }
    div[class*="st-key-subpage_button_band_"] {
        padding-left: 18px;
        padding-right: 18px;
    }
}
            

div[class*="st-key-nav_trend_evolution"] div[data-testid="stButton"] > button,
div[class*="st-key-nav_trend_evolution"] div[data-testid="stButton"] > button p {
    white-space: nowrap !important;
}

div[class*="st-key-nav_trend_evolution"] div[data-testid="stButton"] > button {
    padding: 0 0px !important;
    font-size: 8px !important;
}

div[class*="st-key-nav_trend_evolution"] div[data-testid="stButton"] > button p {
    white-space: nowrap !important;
    font-size: 8px !important;
    letter-spacing: 0.25px !important;
}
            
</style>
""", unsafe_allow_html=True)

# ---------------- Session state ----------------
# ---------------- Session state ----------------
VALID_PAGES = {"HOME", "TREND EVOLUTION", "TREND FORECASTING", "TREND EXPLORER"}

if "page" not in st.session_state:
    st.session_state.page = "HOME"

# The header is an HTML link, so it uses a small query parameter to return home.
# Navigation buttons clear it again so normal page state continues to work.
query_page = st.query_params.get("page")
if isinstance(query_page, list):
    query_page = query_page[0] if query_page else None

# Capture any deep-link params (e.g. from a word-cloud click opening a new tab)
# BEFORE query_params gets cleared below, or they'd be lost on this first rerun.
query_trend = st.query_params.get("trend")
query_view = st.query_params.get("view")

if query_page in VALID_PAGES:
    st.session_state.page = query_page
    if query_view == "profile_only" and query_trend:
        st.session_state["deep_link_trend"] = query_trend
        st.session_state["deep_link_view"] = "profile_only"
    try:
        st.query_params.clear()
    except Exception:
        pass

def go_to_page(page_name):
    st.session_state.page = page_name
    # Leaving Trend Explorer via normal nav should drop any stale deep link,
    # so the profile-only view doesn't reappear if the user comes back later.
    st.session_state.pop("deep_link_trend", None)
    st.session_state.pop("deep_link_view", None)
    try:
        st.query_params.clear()
    except Exception:
        pass


# ---------------- Label colours ----------------
label_colours = {
    "ITEM": "#766161",
    "BRAND": "#87a7b3",
    "COLOR": "#e1f1dd",
    "MATERIAL": "#e1d0b3",
    "PATTERN": "#a18d6d",
    "STYLE": "#b4cde6",
    "SEASON": "#cdc7be",
    "PRODUCT": "#f5efe6",
    "DETAIL": "#628e90"
}

HOME_ENTITY_INFO = [
    ("Clothing & Accessories", "#766161", "Clothing items, footwear and accessories mentioned in articles, such as blazers, boots, handbags, trousers and jewellery."),
    ("Brands", "#87a7b3", "Fashion houses, labels and retailers appearing in editorial coverage, such as Gucci, Zara, Miu Miu and Chanel."),
    ("Colours", "#e1f1dd", "Colour names and shades used to describe fashion pieces, such as ivory, cobalt blue, burgundy, camel and chocolate brown."),
    ("Materials & Fabrics", "#e1d0b3", "Fabric and material references, such as leather, silk, denim, cashmere, suede, wool and satin."),
    ("Prints & Patterns", "#a18d6d", "Surface patterns, prints and motifs, such as plaid, floral, leopard print, stripes and houndstooth."),
    ("Style Aesthetics", "#b4cde6", "Broader fashion aesthetics or style movements, such as minimalist, Y2K, bohemian, preppy and quiet luxury."),
    ("Season Mentions", "#cdc7be", "Seasonal references and collection periods, such as Fall, Winter, Spring/Summer 2026 or Resort collections."),
    ("Signature Products", "#f5efe6", "Named or recognisable fashion products, such as the Samba sneaker, Birkin bag or other iconic pieces."),
    ("Design Details", "#628e90", "Specific garment cuts, silhouettes and construction details, such as oversized, cropped, sheer, wide-leg or pleated."),
]


def handle_auth_query_params():
    auth_action = st.query_params.get("auth")

    if auth_action == "login":
        clear_auth_state()
        st.session_state.auth_status = "not_started"
        st.query_params.clear()
        st.rerun()

    elif auth_action == "logout":
        try:
            supabase = get_supabase()
            supabase.auth.sign_out()
        except Exception:
            pass

        clear_auth_state()
        st.session_state.auth_status = "not_started"
        st.query_params.clear()
        st.rerun()


def render_auth_page():
    st.markdown(
        """
        <style>
        /* Overall page background spacing */
        .block-container {
            padding-top: 3rem !important;
        }
        /* Auth card */
        .auth-banner {
        position: relative;
        width: 100%;
        height: 70px;
        background: #312421;
        border-radius: 0px;
        margin: 0 auto 28px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        }
        
        .auth-title {
        color: #c2dcff;
        font-size: 15px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.5px;
        margin: 0;
        z-index: 2;
        
        }
        .auth-subtitle {
            color: #f5f3f5;
            font-size: 14px;
            text-align: center;
            opacity: 0.88;
            margin-bottom: 22px;
            line-height: 1.6;
        }
        .auth-note {
            color: #c2dcff;
            font-size: 13px;
            text-align: center;
            opacity: 0.85;
            margin-top: 16px;
            line-height: 1.5;
        }
        /* Center Login / Create Account tabs */
        div[data-testid="stTabs"] div[role="tablist"] {
            justify-content: center !important;
            gap: 18px !important;
        }
        div[data-testid="stTabs"] button {
            flex: 0 0 auto !important;
            color: #f5f3f5 !important;
            font-weight: 800 !important;
            font-size: 15px !important;
            padding-left: 18px !important;
            padding-right: 18px !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #c2dcff !important;
        }
        /* Text input labels */
        div[data-testid="stTextInput"] label {
            color: #f5f3f5 !important;
            font-weight: 700 !important;
            font-size: 14px !important;
        }
        /* Smaller text boxes */
        div[data-testid="stTextInput"] {
            max-width: 330px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 14px !important;
            text-align: center !important;
        }
        /* Smaller centered buttons */
        .stButton {
            max-width: 330px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .stButton > button {
            border-radius: 999px !important;
            font-weight: 800 !important;
            height: 42px !important;
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
    # This centers the whole login card on the page
    left_col, center_col, right_col = st.columns([1.4, 1.2, 1.4])
    with center_col:
        st.markdown(
            """
            <div class="auth-banner">
            <div class="auth-title">FASHION TREND ANALYSER</div>
            </div>
            """,
            unsafe_allow_html=True,
        
        )
        st.markdown(
        """
        <div class="auth-subtitle">
            Log in to analyse your own uploaded fashion data, or continue as guest to explore the base dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )
        
        login_tab, signup_tab = st.tabs(["Login", "Create Account"])
        with login_tab:
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_clicked = st.button("Login", use_container_width=True, type = "primary")
            if login_clicked:
                if not login_email or not login_password:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        login_user(login_email.strip(), login_password)
                    except Exception as e:
                        st.error(f"Login failed: {e}")
        with signup_tab:
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            signup_clicked = st.button("Create Account", use_container_width=True)
            if signup_clicked:
                if not signup_email or not signup_password or not confirm_password:
                    st.error("Please fill in all signup fields.")
                elif signup_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(signup_password) < 6:
                    st.error("Password should be at least 6 characters.")
                else:
                    try:
                        response = signup_user(signup_email.strip(), signup_password)
                        if response.session is not None and response.user is not None:
                            st.session_state.auth_status = "authenticated"
                            st.session_state.user_id = response.user.id
                            st.session_state.user_email = response.user.email
                            st.cache_data.clear()
                            st.success("Account created successfully. Logging you in...")
                            st.rerun()
                        else:
                            st.success("Account created. Please check your email to confirm your account, then log in.")
                    except Exception as e:
                        st.error(f"Signup failed: {e}")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue as Guest", use_container_width=True):
            continue_as_guest()
        st.markdown(
            """
            <div class="auth-note">
                Guest mode shows the base fashion dataset only.
            </div>
            """,
            unsafe_allow_html=True,
        )
        


def auth_gate():
    init_auth_state()

    if st.session_state.auth_status in ["authenticated", "guest"]:
        return

    render_auth_page()
    st.stop()

def render_user_status():
    col1, col2 = st.columns([5, 1])

    with col2:
        if st.session_state.get("auth_status") == "authenticated":
            st.markdown(
                f"""
                <div style="
                    color:#c2dcff;
                    font-size:13px;
                    font-weight:800;
                    text-align:right;
                    margin-bottom:6px;
                ">
                    {st.session_state.get("user_email")}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Logout", use_container_width=True):
                logout_user()

        elif st.session_state.get("auth_status") == "guest":
            st.markdown(
                """
                <div style="
                    color:#c2dcff;
                    font-size:13px;
                    font-weight:800;
                    text-align:right;
                    margin-bottom:6px;
                ">
                    Guest Mode
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Login", use_container_width=True):
                clear_auth_state()
                st.session_state.auth_status = "not_started"
                st.rerun()

def render_home_page():
    # IMPORTANT:
    # This HTML is built as one continuous string so Streamlit cannot treat
    # indented HTML lines as a Markdown/code block.

    # Palette pulled straight from label_colours so the UI mirrors what the
    # model actually tags articles with.
    label_colours = {
        "ITEM": "#766161",
        "BRAND": "#87a7b3",
        "COLOR": "#e1f1dd",
        "MATERIAL": "#e1d0b3",
        "PATTERN": "#a18d6d",
        "STYLE": "#b4cde6",
        "SEASON": "#cdc7be",
        "PRODUCT": "#f5efe6",
        "DETAIL": "#628e90",
    }

    style_css = (
        '<style>'
        '@import url("https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&display=swap");'
        '.home-container{font-family:"Inter",sans-serif;color:#312421;max-width:1040px;margin:0 auto;}'
        '.home-hero{background:#312421;border-radius:18px;padding:56px 48px 44px;margin-bottom:40px;}'
        '.home-hero-inner{max-width:720px;}'
        '.home-eyebrow{font-family:"Inter",sans-serif;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#cdc7be;margin-bottom:14px;}'
        '.home-hero h1{font-family:"Fraunces",serif;font-weight:600;font-size:42px;line-height:1.12;color:#f5efe6;margin:0 0 14px;letter-spacing:-.01em;}'
        '.home-subtitle{font-family:"Fraunces",serif;font-style:italic;font-weight:400;font-size:17px;color:#c6dbf1;margin-bottom:28px;}'
        '.swatch-strip{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:30px; justify-content:center}'
        '.swatch-chip{width:28px;height:28px;border-radius:6px;border:1px solid rgba(245,239,230,.25);}'
        '.home-text{font-size:15.5px;line-height:1.7;color:#e1e1e1;}'
        '.home-text b{color:#f5efe6;font-weight:600;}'
        '.home-content{padding:0 4px;}'
        '.home-section-title{font-family:"Fraunces",serif;font-style:italic;font-weight:500;font-size:26px;color:#f5efe6;margin:8px 0 22px;}'
        '.entity-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:48px;}'
        '.entity-card{background:#f5f3f5;border-radius:12px;padding:18px 18px 20px;border:1px solid rgba(49,36,33,.08);transition:transform .15s ease,box-shadow .15s ease;}'
        '.entity-card:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(49,36,33,.08);}'
        '.entity-swatch-row{display:flex;align-items:center;gap:8px;margin-bottom:10px;}'
        '.entity-swatch{width:16px;height:16px;border-radius:4px;border:1px solid rgba(49,36,33,.15);flex-shrink:0;}'
        '.entity-card-title{font-family:"Inter",sans-serif;font-weight:600;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:#544541;}'
        '.entity-card-desc{font-size:14px;line-height:1.55;color:#544541;}'
        '.how-list{display:flex;flex-direction:column;}'
        '.how-card{display:flex;gap:20px;padding:20px 4px;border-top:1px solid rgba(49,36,33,.1);}'
        '.how-card:last-child{border-bottom:1px solid rgba(49,36,33,.1);}'
        '.how-index{font-family:"Fraunces",serif;font-style:italic;font-weight:500;font-size:22px;color:#87a7b3;min-width:36px;}'
        '.how-body b{font-family:"Fraunces",serif;font-style:normal;font-weight:600;font-size:16.5px;color:#312421;display:block;margin-bottom:4px;}'
        '.how-body p{font-size:14.5px;line-height:1.6;color:#544541;margin:0;}'
        '@media (max-width:700px){'
        '.home-hero{padding:36px 24px;}'
        '.home-hero h1{font-size:30px;}'
        '.entity-grid{grid-template-columns:1fr;}'
        '}'
        '</style>'
    )

    def entity_card(label_key, title, desc):
        colour = label_colours[label_key]
        return (
            '<div class="entity-card">'
            '<div class="entity-swatch-row">'
            f'<div class="entity-swatch" style="background:{colour};"></div>'
            f'<div class="entity-card-title">{title}</div>'
            '</div>'
            f'<div class="entity-card-desc">{desc}</div>'
            '</div>'
        )

    def how_card(index, title, desc):
        return (
            '<div class="how-card">'
            f'<div class="how-index">{index}</div>'
            f'<div class="how-body"><b>{title}</b><p>{desc}</p></div>'
            '</div>'
        )

    swatch_strip = ''.join(
        f'<div class="swatch-chip" style="background:{colour};" title="{key}"></div>'
        for key, colour in label_colours.items()
    )

    home_html = (
        '<div class="home-container">'
        + style_css +
        '<section class="home-hero">'
        '<div class="home-hero-inner">'
        '<div class="home-eyebrow">Trend intelligence &middot; Aug 2025 &ndash; Jan 2026</div>'
        '<h1>Welcome to <br>Fashion Trend Analyser</h1>'
        '<div class="home-subtitle">Reading the season, one article at a time.</div>'
        f'<div class="swatch-strip">{swatch_strip}</div>'
        '<div class="home-text">'
        '<b>What is this tool?</b><br><br>'
        'The Fashion Trend Analyser uses an Artificial Intelligence (AI) model to read and understand fashion '
        'articles and editorial content. It automatically identifies key fashion terms such as clothing items, '
        'colours, materials, brands and design details, then tracks how frequently and positively these terms '
        'are discussed over time.'
        '<br><br>'
        'The tool is designed for fashion designers, retailers, fashion trend analysts, data analysts and fashion '
        'enthusiasts who want data-driven insights into emerging fashion trends.'
        '</div>'
        '</div>'
        '</section>'
        '<div class="home-content">'
        '<div class="home-section-title">The vocabulary the model reads</div>'
        '<div class="entity-grid">'
        + entity_card("ITEM", "Clothing & Accessories", "Clothing items, footwear and accessories mentioned in articles, such as blazers, boots, handbags, trousers and jewellery.")
        + entity_card("BRAND", "Brands", "Fashion houses, labels and retailers appearing in editorial coverage, such as Gucci, Zara, Miu Miu and Chanel.")
        + entity_card("COLOR", "Colours", "Colour names and shades used to describe fashion pieces, such as ivory, cobalt blue, burgundy, camel and chocolate brown.")
        + entity_card("MATERIAL", "Materials & Fabrics", "Fabric and material references, such as leather, silk, denim, cashmere, suede, wool and satin.")
        + entity_card("PATTERN", "Prints & Patterns", "Surface patterns, prints and motifs, such as plaid, floral, leopard print, stripes and houndstooth.")
        + entity_card("STYLE", "Style Aesthetics", "Broader fashion aesthetics or style movements, such as minimalist, Y2K, bohemian, preppy and quiet luxury.")
        + entity_card("SEASON", "Season Mentions", "Seasonal references and collection periods, such as Fall, Winter, Spring/Summer 2026 or Resort collections.")
        + entity_card("PRODUCT", "Signature Products", "Named or recognisable fashion products, such as the Samba sneaker, Birkin bag or other iconic pieces.")
        + entity_card("DETAIL", "Design Details", "Specific garment cuts, silhouettes and construction details, such as oversized, cropped, sheer, wide-leg or pleated.")
        + '</div>'
        '<div class="home-section-title">How to use this tool?</div>'
        '<div class="how-list">'
        + how_card("01", "Trend Evolution", "Explore how fashion terms are rising or falling over time. Switch between frequency trends, trend spotlight word clouds and trends weighted based on both frequency and postive or negative associated perception.")
        + how_card("02", "Trend Forecasting", "View forecasts showing how selected fashion trends are expected to grow or decline over the next two months.")
        + how_card("03", "Trend Explorer", "View trend summaries as well as trend leaderboards and use trend profiles to analyse and even compare trends in more detail.")
        + '</div>'
        '</div>'
        '</div>'
    )

    st.markdown(home_html, unsafe_allow_html=True)

def render_left_sidebar():
    """Fixed left-side menu using Streamlit buttons.

    This keeps navigation inside the current Streamlit session, so clicking a
    page changes st.session_state.page and reruns the current tab only.
    """
    items = [
        ("HOME", "HOME", "side_nav_home"),
        ("TREND EVOLUTION", "TREND EVOLUTION", "side_nav_trend_evolution"),
        ("TREND FORECAST", "TREND FORECASTING", "side_nav_trend_forecast"),
        ("TREND EXPLORER", "TREND EXPLORER", "side_nav_trend_explorer"),
    ]

    with st.container(key="left_sidebar_menu"):
        for label, page_name, key in items:
            if st.button(
                label,
                key=key,
                use_container_width=True,
                type="primary" if st.session_state.page == page_name else "secondary",
            ):
                go_to_page(page_name)
                st.rerun()

        with st.container(key="sidebar_bottom_spacer"):
            st.markdown("", unsafe_allow_html=True)

        st.markdown(
            '<a class="sidebar-back-top" href="#top" '
            'onclick="window.parent.scrollTo({top: 0, left: 0, behavior: \'smooth\'}); return false;">'
            'Back to Top</a>',
            unsafe_allow_html=True,
        )





# Auth/login page removed: always start as guest/base-data user.
init_auth_state()

# ---------------- Header ----------------
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
if st.button("FASHION TREND ANALYSER", use_container_width=True, key="header_home_button"):
    go_to_page("HOME")
    st.rerun()

# Fixed vertical menu shown on every page.
render_left_sidebar()
# ════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════
if st.session_state.page == "HOME":
    render_home_page()


# ════════════════════════════════════════════════════════════════
# PAGE: TREND EVOLUTION
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "TREND EVOLUTION":
    page_slug = "analysis"
    st.empty()
    
    if "trend_sub_page" not in st.session_state:
        st.session_state.trend_sub_page = "Trend Frequency Over Time"

    with st.container(key="subpage_button_band_trend_evolution"):
        spacer_l, sub_col1, sub_col2, spacer_r = st.columns([0.5, 1.7, 1.7, 0.5])
        with sub_col1:
            if st.button(
                "Trend Frequency Over Time", key="btn_freq_ts", use_container_width=True,
                type="primary" if st.session_state.trend_sub_page == "Trend Frequency Over Time" else "secondary",
            ):
                st.session_state.trend_sub_page = "Trend Frequency Over Time"
                st.rerun()
        with sub_col2:
            if st.button(
                "Trend Perception Over Time", key="btn_sentiment_ts", use_container_width=True,
                type="primary" if st.session_state.trend_sub_page == "Trend Perception Over Time" else "secondary",
            ):
                st.session_state.trend_sub_page = "Trend Perception Over Time"
                st.rerun()

    with st.container(key="subpage_description_panel"):
        st.markdown(
            f'<div class="subpage-description">{SUBPAGE_DESCRIPTIONS[st.session_state.trend_sub_page]}</div>'
            '<div class="description-bottom-band"></div>',
            unsafe_allow_html=True,
        )

    # Load data once (cached)
    # Load data once from Supabase
    (
        entities_df, phrases_df, trend_units_df, trend_units_from_phrases_df,
        sent_entities_df, sent_phrases_df, sent_trend_units_df,
        sent_trend_units_from_phrases_df
    ) = get_trend_data()

    clean_ent_df = get_clean_entities(entities_df)

    def graph_header(key):
        friendly_title, description = GRAPH_INFO[key]
        st.markdown(
            f'<div class="graph-header-title">{friendly_title}</div>'
            f'<div class="graph-header-desc">{description}</div>',
            unsafe_allow_html=True,
        )
        return friendly_title

    def trend_controls(top_key, season_key, default_value=DEFAULT_TOP_N, max_value=None):
    
        spacer_l, col_n, col_toggle = st.columns([0.3, 8.5, 1.2], gap="small")
        with col_n:
            with st.container(key=f"topn_wide_{top_key}"):
                kwargs = {
                    "label": "Top number of trends to display",
                    "value": int(default_value),
                    "min_value": 1,
                    "key": top_key,
                }
                if max_value is not None:
                    kwargs["max_value"] = max_value
                top_n = st.number_input(**kwargs)
        with col_toggle:
            st.markdown('<div class="season-toggle-pad"></div>', unsafe_allow_html=True)
            show_season = st.toggle("Season", key=season_key)
        
        return top_n, show_season
    

    if st.session_state.trend_sub_page == "Trend Frequency Over Time":

        title = graph_header("Phrases")
        phrase_top_n, show_season = trend_controls("topn_phrases", "season_phrases")
        phrase_fig = get_time_series_chart(phrases_df, "phrase", phrase_top_n, title, entities_df if show_season else None)
        if phrase_fig: st.plotly_chart(phrase_fig, width="stretch", config={"responsive": True})

        title = graph_header("Trend Units")
        trend_unit_top_n, show_season = trend_controls("topn_trend_units", "season_trend_units")
        trend_fig = get_time_series_chart(trend_units_df, "trend_unit", trend_unit_top_n, title, entities_df if show_season else None)
        if trend_fig: st.plotly_chart(trend_fig,  width="stretch", config={"responsive": True})

        #bigram_top_n = st.number_input("Top number of trends to display", value=DEFAULT_TOP_N, key="topn_bigrams")
        #bigram_fig = get_trend_units_time_series(trend_units_from_phrases_df, "bigram", bigram_top_n, "Bigram Trend Units Over Time")
        #if bigram_fig: st.plotly_chart(bigram_fig, use_container_width=True)

        title = graph_header("Trigram Trend Units")
        trigram_top_n, show_season = trend_controls("topn_trigrams", "season_trigrams")
        trigram_fig = get_trend_units_time_series(trend_units_from_phrases_df, "trigram", trigram_top_n, "Three Word Fashion Terms", entities_df if show_season else None)
        if trigram_fig: st.plotly_chart(trigram_fig,  width="stretch", config={"responsive": True})

        for label in ATTRIBUTE_LABELS:
            df_to_use = clean_ent_df if label == "ITEM" else entities_df
            title = graph_header(label)
            attr_top_n, show_season = trend_controls(f"topn_{label}", f"season_{label}")
            attr_fig = get_attribute_time_series(df_to_use, label, attr_top_n, title, entities_df if show_season else None)
            if attr_fig:
                st.plotly_chart(attr_fig,  width="stretch", config={"responsive": True})
            else:
                st.markdown(f'<div style="color:#c2dcff;font-weight:700;margin-bottom:20px;">No data for {title}.</div>', unsafe_allow_html=True)

    elif st.session_state.trend_sub_page == "Trend Perception Over Time":
        import matplotlib
        matplotlib.use("Agg")
        @st.cache_data
        def cached_sentiment_fig(df, text_col, top_n, title, season_entities_df=None):
            return create_weighted_sentiment_fig(df, text_col, top_n, title, season_entities_df=season_entities_df)

        def sentiment_section(label, df, text_col, key_suffix):
            friendly_title = graph_header(label)
            chart_title = f"Frequency * Perception of {friendly_title}"
            top_n, show_season = trend_controls(
                f"sent_topn_{key_suffix}",
                f"sent_season_{key_suffix}",
                default_value=DEFAULT_SENTIMENT_TOP_N.get(label, DEFAULT_TOP_N),
                max_value=100,
            )
            fig = cached_sentiment_fig(
                df, text_col, top_n, chart_title,
                season_entities_df=sent_entities_df if show_season else None,
            )
            if fig:
                st.plotly_chart(fig,  width="stretch", config={"responsive": True})
            else:
                st.markdown(
                    f'<div style="color:#c2dcff;font-weight:700;margin-bottom:20px;">'
                    f'No data for {friendly_title} with frequency ≥ {SENTIMENT_MIN_FREQ}.</div>',
                    unsafe_allow_html=True
                )

        sentiment_section("Phrases",     sent_phrases_df,     "phrase",     "phrases")
        sentiment_section("Trend Units", sent_trend_units_df, "trend_unit", "trend_units")

        trigram_sent_df = sent_trend_units_from_phrases_df[
            sent_trend_units_from_phrases_df["ngram_type"] == "trigram"
        ] if "ngram_type" in sent_trend_units_from_phrases_df.columns else sent_trend_units_from_phrases_df
        sentiment_section("Trigram Trend Units", trigram_sent_df, "trend_unit", "trigram_trend_units")

        sent_clean_ent_df = get_clean_entities(sent_entities_df)
        for label in SENTIMENT_ATTR_LABELS:
            df_to_use = sent_clean_ent_df if label == "ITEM" else sent_entities_df
            sentiment_section(label, df_to_use[df_to_use["label"] == label], "entity", label)



    elif st.session_state.trend_sub_page == "Trend Spotlight":
        for label in WORDCLOUD_LABELS:
            friendly_title, description = GRAPH_INFO.get(label, (label, ""))
            st.markdown(
                f'<div style="color:#c2dcff;font-size:25px;font-weight:800;margin-top:25px;margin-bottom:4px;text-align:center;">{friendly_title}</div>'
                f'<div class="graph-header-desc">{description}</div>',
                unsafe_allow_html=True,
            )
            df_to_use = clean_ent_df if label == "ITEM" else entities_df
            html = get_wordcloud(df_to_use, label)
            if html:
                components.html(html, height=500)
            else:
                st.markdown(f'<div style="color:#c2dcff;font-weight:700;margin-bottom:20px;">No data available for {friendly_title}.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE: TREND FORECASTING
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "TREND FORECASTING":
    page_slug = "forecast"
    trend_data = get_trend_data()

    (
        entities_df,
        phrases_df,
        trend_units_df,
        trend_units_from_phrases_df,
        sent_entities_df,
        sent_phrases_df,
        sent_trend_units_df,
        *_
    ) = trend_data

    render_trend_prediction_page(
        entities_df=entities_df,
        phrases_df=phrases_df,
        trend_units_df=trend_units_df,
        trend_units_from_phrases_df=trend_units_from_phrases_df,
        sent_entities_df=sent_entities_df,
        sent_phrases_df=sent_phrases_df,
        sent_trend_units_df=sent_trend_units_df,
        forecast_file="data/all_future_forecasts.xlsx",
    )


# ════════════════════════════════════════════════════════════════
# PAGE: TREND EXPLORER
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "TREND EXPLORER":
    page_slug = "explorer"

    (
        entities_df,
        phrases_df,
        trend_units_df,
        trend_units_from_phrases_df,
        sent_entities_df,
        sent_phrases_df,
        sent_trend_units_df,
        sent_trend_units_from_phrases_df,
    ) = get_trend_data()

    results = {
        "entities": entities_df,
        "phrases": phrases_df,
        "trend_units": trend_units_df,
        "trend_units_from_phrases": trend_units_from_phrases_df,
        "entities_sentiment": sent_entities_df,
        "phrases_sentiment": sent_phrases_df,
        "trend_units_sentiment": sent_trend_units_df,
        "trend_units_from_phrases_sentiment": sent_trend_units_from_phrases_df,
    }

    clean_ent_df = get_clean_entities(entities_df)
    predictions = get_prediction_data()

    trend_explorer.render_explorer(results, predictions, clean_ent_df)
