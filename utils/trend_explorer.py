"""
trend_explorer.py
------------------
Trend Explorer page for the fashion trend Streamlit app.

Usage from app.py:

    from utils import trend_explorer
    trend_explorer.render_explorer(results, predictions)

or:

    trend_explorer.render(results, predictions)

`results` is expected to contain any subset of:

    results = {
        "entities": entities_df,
        "phrases": phrases_df,
        "trend_units": trend_units_df,
        "entities_sentiment": sent_entities_df,
        "phrases_sentiment": sent_phrases_df,
        "trend_units_sentiment": sent_trend_units_df,
    }

`predictions` is optional.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import re
import json
import streamlit.components.v1 as components


# ════════════════════════════════════════════════════════════════
# CONFIG -- adjust here if your real column names differ
# ════════════════════════════════════════════════════════════════

ENTITY_TEXT_COL = "entity"
PHRASE_TEXT_COL = "phrase"
TREND_UNIT_TEXT_COL = "trend_unit"
LABEL_COL = "label"
LABELS_COL = "labels"
DATE_COL = "date"
SCORE_COL = "roberta_score_signed"
WEIGHTED_SCORE_INTERNAL_COL = "weighted_sentiment"
WEIGHTED_SENTIMENT_TOOLTIP = (
    "Weighted Sentiment is calculated as frequency × sentiment/perception of the trend for the selected "
    "period. "
    "Higher positive values indicate more strongly positive coverage, while lower negative "
    "values indicate more strongly negative coverage."
)

AVG_SENTIMENT_TOOLTIP = (
    "Average Sentiment is the mean sentiment/perception score for all mentions of this trend "
    "for the selected period. Values closer to +1 indicate a more positive perception, "
    "values closer to -1 indicate a more negative perception, and values around 0 indicate "
    "neutral or mixed sentiment."
)


SUMMARY_CARD_TOOLTIPS = {
    "Top Trend": "The most mentioned trend in the selected data range.",
    "Most Discussed Season": "The season term with the highest number of mentions in the selected data range.",
    "Top Phrase": "The most mentioned fashion-related term in the selected data range.",
    "Top Trend Unit": "The two fashion attributes which appear the most frequently together.",
    "Fastest Growing": "The trend with the largest month-over-month increase in mentions.",
    "Most Positive": "The trend with the highest average sentiment score, regardless of mention frequency.",
    "Most Negative": "The trend with the lowest average sentiment score, regardless of mention frequency.",
    "Discussed Most Positively": "The trend with the highest weighted sentiment, calculated as frequency × sentiment score.",
    "Discussed Most Negatively": "The trend with the lowest weighted sentiment, calculated as frequency × sentiment score.",
    "Most Discussed Category": "The category with the highest total number of mentions in the selected data range.",
    "Active Trends": "The number of trends that meet the minimum mention threshold in the selected data range.",
}

SUBPAGE_DESCRIPTIONS = {
    "Trend Summary": (
        "Review the overall trend landscape, including total mentions, weighted sentiment, top trends by category, "
        "and emerging trends based on recent movement. Use this page to quickly identify which fashion trends are "
        "rising, stable, fading, or gaining attention across the article dataset."
    ),
    "Trend Spotlight": (
        "The larger a word appears, the more frequently it was mentioned across fashion articles. "
        "Hover over any word to see its exact count and share of mentions and click on a word to open up "
        "its trend profile and view the trend in more detail."
    ),
    "Trend Comparison": (
        "Review the trend leaderboard and search and compare selected fashion trends side by side. Use this page to inspect each trend profile, "
        "compare mentions and sentiment over time, view forecasted sentiment where available, and explore the "
        "common phrases linked to each trend."
    ),
}
MIN_MENTIONS = 3
RISING_THRESHOLD = 15
DECLINING_THRESHOLD = -15

ENTITY_CATEGORIES = [
    "ITEM", "DETAIL", "STYLE", "MATERIAL", "PATTERN", "COLOR", "BRAND", "PRODUCT",
]
ENTITY_NAMES = [
    "Clothing & Accessories", "Design Details", "Style Aesthetics", "Materials & Fabrics",
    "Prints & Patterns", "Colours", "Brands", "Signature Products",
]
ENTITY_DISPLAY_NAMES = dict(zip(ENTITY_CATEGORIES, ENTITY_NAMES))
CATEGORY_OPTIONS = ["ALL", "PHRASES", "TREND UNITS"] + ENTITY_CATEGORIES
CARD_CATEGORIES = ENTITY_CATEGORIES + ["SEASON"] 

# App palette
APP_BG = "#544541"
CARD_BG = "#312421"
SOFT_BLUE = "#c2dcff"
OFF_WHITE = "#f5f3f5"
MUTED_BLUE = "#a6c8eb"
BROWN_CARD = "#766161"
BEIGE = "#f5efe6"
LIGHT_BLUE = "#e1f0ff"
LIGHT_BEIGE = "#dddcdb"
BRIGHT_BEIGE = "#e1d0b3"
CARD_BG_RGBA = "rgba(49, 36, 33, 0.55)"
BLUE = "#b4cde6"
LIGHT_WHITE = "#ede9e9"
FIGURE_BG = "#4b3d3a"
DARK_BROWN = "#40322f"
OFF_WHITE_RGBA = "rgba(245, 243, 245, 0.80)"
WHITE = "#FFFFFF"
GREY = "#636363"
BUTTON_BROWN ="#40322f"
# Trend comparison colours. Change these values if you want each compared
# trend to use different chart, profile card, and word cloud colours.
COMPARE_TREND_STYLES = [
    {
        "chart": CARD_BG,
        "card_bg": CARD_BG,
        "card_font": OFF_WHITE,
        "title_font": WHITE,
        "profile_label_font": BEIGE,
        "profile_badge_bg": None,
        "profile_badge_font": None,
        "status_badge_bg": None,
        "wordcloud_bg": SOFT_BLUE,
        "wordcloud_font": CARD_BG,
        "lolipop_title": WHITE,
    },
    {
        "chart": BLUE,
        "card_bg": BLUE,
        "card_font": DARK_BROWN, #values
        "title_font": CARD_BG,
        "profile_label_font": DARK_BROWN, #titles
        "profile_badge_bg": GREY, #badge
        "profile_badge_font": WHITE,
        "status_badge_bg": GREY,
        "status_badge_font":  None,
        "wordcloud_bg": BRIGHT_BEIGE,
        "wordcloud_font": CARD_BG,
        "lolipop_title": BLUE,
    },
    {
        "chart": LIGHT_WHITE,
        "card_bg": LIGHT_WHITE,
        "card_font": APP_BG, #values
        "title_font": FIGURE_BG,
        "profile_label_font": APP_BG, #titles
        "profile_badge_bg": APP_BG, #badge 
        "profile_badge_font": WHITE,
        "status_badge_bg": APP_BG,
        "status_badge_font": None,
        "wordcloud_bg": MUTED_BLUE,
        "wordcloud_font": CARD_BG,
        "lolipop_title": LIGHT_WHITE,
    },
]

LABEL_COLORS = {
    "ITEM": "#766161",
    "BRAND": "#87a7b3",
    "COLOR": "#e1f1dd",
    "MATERIAL": "#e1d0b3",
    "PATTERN": "#a18d6d",
    "STYLE": "#b4cde6",
    "SEASON": "#cdc7be",
    "PRODUCT": "#f5efe6",
    "DETAIL": "#628e90",
    "PHRASE": "#c2dcff",
    "TREND_UNIT": "#a6c8eb",
}

STATUS_COLORS = {
    "Rising": "#8fd3a0",
    "Emerging": "#a7c7e7",
    "Stable": "#c2dcff",
    "Declining": "#e2a0a0",
    "Fading": "#9c8b86",
}

STATUS_ICON = {
    "Rising": "▲",
    "Emerging": "✦",
    "Stable": "●",
    "Declining": "▼",
    "Fading": "▽",
}

WORDCLOUD_LABELS = [
    "ITEM", "COLOR", "MATERIAL", "PATTERN", "STYLE",
     "DETAIL","BRAND", "PRODUCT"
]

ENTITY_DISPLAY_NAMES = {
    "ITEM": "Clothing & Accessories",
    "DETAIL": "Design Details",
    "STYLE": "Style Aesthetics",
    "MATERIAL": "Materials & Fabrics",
    "PATTERN": "Prints & Patterns",
    "COLOR": "Colours",
    "BRAND": "Brands",
    "PRODUCT": "Signature Products",
}

# ════════════════════════════════════════════════════════════════
# STYLING
# ════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    if "te_sub_page" not in st.session_state:
        st.session_state["te_sub_page"] = "Trend Summary"

    active_summary_css = ""
    active_comparison_css = ""
    active_spotlight_css = ""

    if st.session_state.get("te_sub_page") == "Trend Summary":
        active_summary_css = f"""
        div[class*="st-key-te_nav_summary"] div[data-testid="stButton"] button,
        div[class*="st-key-te_nav_summary"] button[data-testid^="stBaseButton"] {{
            background: {BEIGE} !important;
            background-color: {BEIGE} !important;
            border: 2px solid {BROWN_CARD} !important;
            color: {CARD_BG} !important;
            box-shadow: none !important;
        }}
        div[class*="st-key-te_nav_summary"] div[data-testid="stButton"] button *,
        div[class*="st-key-te_nav_summary"] button[data-testid^="stBaseButton"] * {{
            color: {CARD_BG} !important;
        }}
        """

    if st.session_state.get("te_sub_page") == "Trend Comparison":
        active_comparison_css = f"""
        div[class*="st-key-te_nav_comparison"] div[data-testid="stButton"] button,
        div[class*="st-key-te_nav_comparison"] button[data-testid^="stBaseButton"] {{
            background: {BEIGE} !important;
            background-color: {BEIGE} !important;
            border: 2px solid {BROWN_CARD} !important;
            color: {CARD_BG} !important;
            box-shadow: none !important;
        }}
        div[class*="st-key-te_nav_comparison"] div[data-testid="stButton"] button *,
        div[class*="st-key-te_nav_comparison"] button[data-testid^="stBaseButton"] * {{
            color: {CARD_BG} !important;
        }}
        """
    if st.session_state.get("te_sub_page") == "Trend Spotlight":
        active_spotlight_css = f"""
        div[class*="st-key-te_nav_spotlight"] div[data-testid="stButton"] button,
        div[class*="st-key-te_nav_spotlight"] button[data-testid^="stBaseButton"] {{
            background: {BEIGE} !important;
            background-color: {BEIGE} !important;
            border: 2px solid {BROWN_CARD} !important;
            color: {CARD_BG} !important;
            box-shadow: none !important;
        }}
        div[class*="st-key-te_nav_spotlight"] div[data-testid="stButton"] button *,
        div[class*="st-key-te_nav_spotlight"] button[data-testid^="stBaseButton"] * {{
            color: {CARD_BG} !important;
        }}
        """
    st.markdown(
        f"""
        <style>
        /* ════════════════════════════════════════════════════════════════
           TREND EXPLORER — PAGE-SCOPED SPACING + HEADER
        ════════════════════════════════════════════════════════════════ */

        div[class*="st-key-trend_explorer_page_body"] {{
            padding-left: 35px !important;
            padding-right: 35px !important;
            padding-bottom: 72px !important;
            box-sizing: border-box !important;
        }}

        div[class*="st-key-trend_explorer_page_body"] > div {{
            padding-bottom: 18px !important;
        }}

        /* Trend Explorer subpage buttons: same pill style as the previous header. */
        .te-static-subpage-header {{
            width: min(430px, 78%);
            height: 44px;
            min-height: 44px;
            margin: 10px auto 0 auto;
            border-radius: 999px;
            background-color: {BEIGE};
            color: {CARD_BG};
            border: 2px solid {BROWN_CARD};
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 17px;
            font-weight: 800;
            letter-spacing: 0.2px;
            position: relative;
            z-index: 1;
            box-sizing: border-box;
        }}

        div[class*="st-key-subpage_button_band_trend_explorer"] div[data-testid="stHorizontalBlock"]
        div[class*="st-key-subpage_button_band_trend_profile_only"] div[data-testid="stHorizontalBlock"]
        {{
            position: relative;
            z-index: 1;
        }} 
        
        

        div[class*="st-key-te_nav_summary"] div[data-testid="stButton"] button,
        div[class*="st-key-te_nav_comparison"] div[data-testid="stButton"] button,
        div[class*="st-key-te_nav_spotlight"] div[data-testid="stButton"] button,
        div[class*="st-key-te_nav_summary"] button[data-testid^="stBaseButton"],
        div[class*="st-key-te_nav_comparison"] button[data-testid^="stBaseButton"],
        div[class*="st-key-te_nav_spotlight"] button[data-testid^="stBaseButton"]
        {{
            height: 44px !important;
            min-height: 44px !important;
            border-radius: 999px !important;
            background: {DARK_BROWN} !important;
            background-color: {DARK_BROWN} !important;
            color: {BEIGE} !important;
            border: 2px solid {APP_BG} !important;
            font-size: 17px !important;
            font-weight: 800 !important;
            letter-spacing: 0.2px !important;
            box-sizing: border-box !important;
            position: relative !important;
            z-index: 1 !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        div[class*="st-key-te_nav_summary"] div[data-testid="stButton"] button *,
        div[class*="st-key-te_nav_comparison"] div[data-testid="stButton"] button *,
        div[class*="st-key-te_nav_spotlight"] div[data-testid="stButton"] button *,
        div[class*="st-key-te_nav_summary"] button[data-testid^="stBaseButton"] *,
        div[class*="st-key-te_nav_comparison"] button[data-testid^="stBaseButton"] *,
        div[class*="st-key-te_nav_spotlight"] button[data-testid^="stBaseButton"] * 
        {{
            color: {BEIGE} !important;
            font-size: 17px !important;
            font-weight: 800 !important;
        }}

        div[class*="st-key-te_nav_summary"] div[data-testid="stButton"] button:hover,
        div[class*="st-key-te_nav_comparison"] div[data-testid="stButton"] button:hover,
        div[class*="st-key-te_nav_spotlight"] div[data-testid="stButton"] button:hover,
        div[class*="st-key-te_nav_summary"] button[data-testid^="stBaseButton"]:hover,
        div[class*="st-key-te_nav_comparison"] button[data-testid^="stBaseButton"]:hover,
        div[class*="st-key-te_nav_spotlight"] button[data-testid^="stBaseButton"]:hover
        {{
            background: {BEIGE} !important;
            background-color: {BEIGE} !important;
            color: {CARD_BG} !important;
            border: 2px solid {APP_BG} !important;
            box-shadow: none !important;
        }}

        div[class*="st-key-te_nav_summary"] div[data-testid="stButton"] button:hover *,
        div[class*="st-key-te_nav_comparison"] div[data-testid="stButton"] button:hover *,
        div[class*="st-key-te_nav_spotlight"] div[data-testid="stButton"] button:hover *,
        div[class*="st-key-te_nav_summary"] button[data-testid^="stBaseButton"]:hover *,
        div[class*="st-key-te_nav_comparison"] button[data-testid^="stBaseButton"]:hover *,
        div[class*="st-key-te_nav_spotlight"] button[data-testid^="stBaseButton"]:hover * 
        {{
            color: {CARD_BG} !important;
        }}


        /* Make the Trend Explorer polka dot band beige, without changing other pages. */
        div[class*="st-key-subpage_button_band_trend_explorer"]::before,
        div[class*="st-key-subpage_button_band_trend_profile_only"]::before {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: 0.36;
            background-image:
                radial-gradient(circle, rgb(245, 239, 230) 0 3px, transparent 3.5px),
                radial-gradient(circle, rgb(245, 239, 230) 0 3px, transparent 3.5px);
            background-size: 34px 26px;
            background-position: 0 0, 17px 13px;
        }}

        .te-section-title {{
            color: {SOFT_BLUE};
            font-size: 22px;
            font-weight: 900;
            text-align: center;
            margin: 28px 0 12px 0;
            padding-top: 8px;
            letter-spacing: 0.2px;
        }}

        .te-section-subtitle {{
            color: {OFF_WHITE};
            font-size: 13px;
            font-weight: 600;
            text-align: center;
            max-width: 900px;
            margin: 0 auto 22px auto;
            opacity: 0.92;
            padding-bottom: 4px;
        }}

        .te-section-line {{
            width: 800px;
            height: 1px;
            background-color: rgba(245, 243, 245, 0.78);
            margin: -10px auto 26px auto;
            border-radius: 999px;
        }}

        .te-tooltip {{
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted rgba(245, 243, 245, 0.78);
            cursor: help;
        }}

        .te-tooltip::after {{
            content: attr(data-tooltip);
            position: absolute;
            left: 50%;
            bottom: calc(100% + 9px);
            transform: translateX(-50%);
            width: 285px;
            background-color: {CARD_BG};
            color: {OFF_WHITE};
            border: 1px solid rgba(194, 220, 255, 0.35);
            border-radius: 10px;
            padding: 9px 11px;
            font-size: 12px;
            line-height: 1.35;
            font-weight: 700;
            text-transform: none;
            letter-spacing: 0;
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            z-index: 999999;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        }}

        .te-tooltip:hover::after {{
            opacity: 1;
            visibility: visible;
        }}

        .te-card {{
            background-color: {CARD_BG};
            border-radius: 16px;
            padding: 18px 20px;
            margin: 0 0 22px 0;
            border: 1px solid rgba(194, 220, 255, 0.18);
            min-height: 118px;
            box-sizing: border-box;
            position: relative;
            overflow: visible !important;
            z-index: 1;
        }}

        .te-card:hover {{
            z-index: 9999;
        }}

        div[class*="st-key-trend_explorer_page_body"],
        div[class*="st-key-trend_explorer_page_body"] div[data-testid="stVerticalBlock"],
        div[class*="st-key-trend_explorer_page_body"] div[data-testid="stHorizontalBlock"],
        div[class*="st-key-trend_explorer_page_body"] div[data-testid="column"] {{
            overflow: visible !important;
        }}

        .te-card h4 {{
            margin: 0 0 8px 0;
            font-size: 0.78rem;
            font-weight: 800;
            color: {BEIGE};
            opacity: 0.82;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .te-card .te-value {{
            font-size: 1.35rem;
            font-weight: 900;
            color: {OFF_WHITE};
            line-height: 1.25;
        }}

        .te-card .te-sub {{
            font-size: 0.8rem;
            color: {SOFT_BLUE};
            opacity: 0.92;
            margin-top: 6px;
            line-height: 1.4;
        }}

        .te-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 900;
            background-color: rgba(194, 220, 255, 0.10);
            color: {SOFT_BLUE};
            white-space: nowrap;
        }}

        .te-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
            flex: 0 0 auto;
        }}

        .te-table-wrap {{
            margin: 0 0 30px 0;
            background-color: {CARD_BG};
            border-radius: 16px;
            padding: 16px 18px;
            border: 1px solid rgba(194, 220, 255, 0.18);
            overflow: visible !important;
            box-sizing: border-box;
            position: relative;
            z-index: 10;
        }}

        .te-table-wrap:hover {{
            z-index: 9999;
        }}

        table.te-table {{
            width: 100%;
            border-collapse: collapse;
            color: {OFF_WHITE};
            font-size: 0.9rem;
        }}

        table.te-table th {{
            text-align: left;
            padding: 9px 10px;
            border-bottom: 1px solid rgba(194, 220, 255, 0.28);
            color: {SOFT_BLUE};
            font-weight: 900;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        table.te-table td {{
            padding: 10px 10px;
            border-bottom: 1px solid rgba(194, 220, 255, 0.08);
            vertical-align: middle;
            color: {OFF_WHITE};
            font-weight: 650;
        }}

        table.te-table td b {{
            color: {OFF_WHITE};
            font-weight: 800;
        }}

        table.te-table tr:hover {{
            background-color: rgba(245, 243, 245, 0.06);
        }}

        .te-profile-row {{
            display: flex;
            justify-content: space-between;
            gap: 14px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(194, 220, 255, 0.10);
            font-size: 0.9rem;
            color: {LIGHT_BEIGE};
        }}

        .te-profile-row span:first-child {{
            opacity: 0.70;
            font-weight: 700;
        }}

        .te-profile-row b {{
            color: {OFF_WHITE};
            text-align: right;
        }}

        .te-emerging-card {{
            background-color: {CARD_BG};
            border-left: 4px solid {STATUS_COLORS['Emerging']};
            border-radius: 12px;
            padding: 14px 15px;
            margin-bottom: 22px;
            border-top: 1px solid rgba(194, 220, 255, 0.12);
            border-right: 1px solid rgba(194, 220, 255, 0.12);
            border-bottom: 1px solid rgba(194, 220, 255, 0.12);
        }}

        .te-note {{
            color: {SOFT_BLUE};
            background-color: rgba(49, 36, 33, 0.72);
            border: 1px solid rgba(194, 220, 255, 0.22);
            border-radius: 12px;
            padding: 12px 15px;
            margin: 8px 0 26px 0;
            font-size: 0.9rem;
            font-weight: 700;
        }}

        .te-forecast-note {{
            color: {SOFT_BLUE};
            font-size: 0.84rem;
            font-weight: 750;
            margin-top: 8px;
            margin-bottom: 24px;
            opacity: 0.95;
        }}

        div[class*="st-key-trend_explorer_filters"],
        div[class*="st-key-trend_explorer_controls"],
        div[class*="st-key-trend_explorer_profile_select"] {{
            margin: 0 0 30px 0 !important;
            padding: 18px 22px 14px 22px !important;
            background-color: {CARD_BG_RGBA} !important;
            #border: 1px solid {CARD_BG_RGBA} !important;
            border-radius: 16px !important;
            box-sizing: border-box !important;
        }}

        div[class*="st-key-trend_explorer_filters"] label,
        div[class*="st-key-trend_explorer_filters"] label *,
        div[class*="st-key-trend_explorer_controls"] label,
        div[class*="st-key-trend_explorer_controls"] label *,
        div[class*="st-key-trend_explorer_profile_select"] label,
        div[class*="st-key-trend_explorer_profile_select"] label * {{
            color: {OFF_WHITE} !important;
            font-weight: 700 !important;
            font-size: 13px !important;
        }}

        div[class*="st-key-trend_explorer_filters"] input,
        div[class*="st-key-trend_explorer_controls"] input,
        div[class*="st-key-trend_explorer_profile_select"] input {{
            background-color: {OFF_WHITE} !important;
            color: {CARD_BG} !important;
            border-radius: 10px !important;
            font-weight: 750 !important;
        }}

        div[class*="st-key-trend_explorer_filters"] div[data-baseweb="select"] > div,
        div[class*="st-key-trend_explorer_controls"] div[data-baseweb="select"] > div,
        div[class*="st-key-trend_explorer_profile_select"] div[data-baseweb="select"] > div {{
            background-color: {OFF_WHITE} !important;
            border-radius: 10px !important;
            color: {CARD_BG} !important;
            font-weight: 750 !important;
        }}

        div[class*="st-key-trend_explorer_controls"] div[data-testid="stSlider"] label,
        div[class*="st-key-trend_explorer_controls"] div[data-testid="stSlider"] label * {{
            color: {SOFT_BLUE} !important;
            font-weight: 900 !important;
        }}

        .te-reset-pad {{
            height: 25px;
        }}

        .st-key-te_reset_search div[data-testid="stButton"] > button {{
            background-color: {BROWN_CARD} !important;
            color: {OFF_WHITE} !important;
            border: 1px solid rgba(194, 220, 255, 0.28) !important;
            border-radius: 10px !important;
            height: 42px !important;
            min-height: 42px !important;
            margin-top: 0 !important;
            padding: 0 14px !important;
            font-size: 13px !important;
            font-weight: 900 !important;
        }}

        .st-key-te_reset_search div[data-testid="stButton"] > button p {{
            color: {OFF_WHITE} !important;
            font-size: 13px !important;
            font-weight: 900 !important;
        }}

        .st-key-te_reset_search div[data-testid="stButton"] > button:hover {{
            background-color: {OFF_WHITE} !important;
            color: {CARD_BG} !important;
            border: 1px solid {SOFT_BLUE} !important;
        }}

        .st-key-te_reset_search div[data-testid="stButton"] > button:hover p {{
            color: {CARD_BG} !important;
        }}

        .st-key-te_view_more_emerging div[data-testid="stButton"] > button {{
            background-color: {BROWN_CARD} !important;
            color: {OFF_WHITE} !important;
            border: 1px solid rgba(194, 220, 255, 0.28) !important;
            border-radius: 999px !important;
            min-height: 40px !important;
            font-size: 13px !important;
            font-weight: 900 !important;
            padding: 0 22px !important;
        }}

        .st-key-te_view_more_emerging div[data-testid="stButton"] > button p {{
            color: {OFF_WHITE} !important;
            font-size: 13px !important;
            font-weight: 900 !important;
        }}

        .st-key-te_view_more_emerging div[data-testid="stButton"] > button:hover {{
            background-color: {OFF_WHITE} !important;
            color: {CARD_BG} !important;
            border: 1px solid {SOFT_BLUE} !important;
        }}

        .st-key-te_view_more_emerging div[data-testid="stButton"] > button:hover p {{
            color: {CARD_BG} !important;
        }}

        div[class*="st-key-trend_explorer_controls_emerging"] div[data-testid="stNumberInput"] {{
            width: 100% !important;
        }}
        
        div[class*="st-key-trend_explorer_controls_emerging"] div[data-testid="stNumberInput"] > div {{
            width: 100% !important;
            height: 38px !important;
            min-height: 38px !important;
            background-color: var(--off-white-rgba) !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            box-shadow: none !important;
        }}
            
        div[class*="st-key-trend_explorer_controls_emerging"] div[data-baseweb="input"] {{
            height: 38px !important;
            min-height: 38px !important;
            background-color: var(--off-white-rgba) !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}
        
        div[class*="st-key-trend_explorer_controls_emerging"] div[data-baseweb="input"] > div {{
            height: 38px !important;
            min-height: 38px !important;
            background-color: var(--off-white-rgba) !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}
        
        div[class*="st-key-trend_explorer_controls_emerging"] div[data-testid="stNumberInput"] input {{
            height: 38px !important;
            min-height: 38px !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            text-align: left !important;
            font-weight: 900 !important;
        }}
        
        div[class*="st-key-trend_explorer_controls_emerging"] div[data-testid="stNumberInput"] button {{
            height: 38px !important;
            min-height: 38px !important;
            background-color: var(--off-white-rgba) !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }}

        .te-filter-title-card {{
            margin: 24px 0 12px 0 !important;
            padding: 16px 22px;
            background-color: {CARD_BG_RGBA};
            border-radius: 16px;
            box-sizing: border-box;
            color: {OFF_WHITE};
            font-size: 15px !important;
            font-weight: 700 !important;
            text-align: left;
            letter-spacing: 0.2px;
        }}

        div[class*="st-key-te_add_compare"] div[data-testid="stButton"] button,
        div[class*="st-key-te_remove_compare"] div[data-testid="stButton"] button {{
            background-color: {BROWN_CARD} !important;
            color: {OFF_WHITE} !important;
            border: 1px solid rgba(194, 220, 255, 0.28) !important;
            border-radius: 999px !important;
            min-height: 32px !important;
            height: 32px !important;
            width: 34px !important;
            padding: 0 !important;
            font-size: 15px !important;
            font-weight: 800 !important;
            box-shadow: none !important;
        }}

        div[class*="st-key-te_add_compare"] div[data-testid="stButton"] button p,
        div[class*="st-key-te_remove_compare"] div[data-testid="stButton"] button p,
        div[class*="st-key-te_add_compare"] div[data-testid="stButton"] button *,
        div[class*="st-key-te_remove_compare"] div[data-testid="stButton"] button * {{
            color: {OFF_WHITE} !important;
            font-size: 15px !important;
            font-weight: 800 !important;
        }}

        div[class*="st-key-te_add_compare"] div[data-testid="stButton"] button:hover,
        div[class*="st-key-te_remove_compare"] div[data-testid="stButton"] button:hover {{
            background-color: {OFF_WHITE} !important;
            color: {CARD_BG} !important;
            border: 1px solid {APP_BG} !important;
            box-shadow: none !important;
        }}

        div[class*="st-key-te_add_compare"] div[data-testid="stButton"] button:hover p,
        div[class*="st-key-te_remove_compare"] div[data-testid="stButton"] button:hover p,
        div[class*="st-key-te_add_compare"] div[data-testid="stButton"] button:hover *,
        div[class*="st-key-te_remove_compare"] div[data-testid="stButton"] button:hover * {{
            color: {CARD_BG} !important;
        }}




        div[data-testid="stPlotlyChart"] {{
            margin-top: 14px !important;
            margin-bottom: 52px !important;
        }}

        @media (max-width: 900px) {{
            div[class*="st-key-trend_explorer_page_body"] {{
                padding-left: 18px !important;
                padding-right: 18px !important;
            }}

            .te-static-subpage-header {{
                width: 92%;
                font-size: 15px;
            }}
        }}

        div[class*="st-key-subpage_description_panel"],
        div[class*="st-key-subpage_description_panel_profile_only"]
        {{
            border-top: 1.5px solid #8e827d !important;
            border-bottom: 1.5px solid #8e827d !important;
        }}

        div[class*="st-key-te_emerging_status_filter"] div[data-baseweb="tag"],
        div[class*="st-key-te_emerging_status_filter"] span[data-baseweb="tag"] {{
            background-color: {BLUE} !important;
            color: {CARD_BG} !important;
            border: 1px solid rgba(194, 220, 255, 0.28) !important;
            border-radius: 9px !important;
            font-weight: 700 !important;
        }}
        
        div[class*="st-key-te_emerging_status_filter"] div[data-baseweb="tag"] *,
        div[class*="st-key-te_emerging_status_filter"] span[data-baseweb="tag"] * {{
            color: {CARD_BG} !important;
            fill: {OFF_WHITE_RGBA} !important;
        }}
        
        div[class*="st-key-te_emerging_status_filter"] div[data-baseweb="tag"] svg,
        div[class*="st-key-te_emerging_status_filter"] span[data-baseweb="tag"] svg {{
            fill: {OFF_WHITE_RGBA} !important;
        }}

        /* EMERGING LABEL */
        div[class*="st-key-trend_explorer_controls_emerging"] label,
        div[class*="st-key-trend_explorer_controls_emerging"] label *,
        div[class*="st-key-trend_explorer_controls_emerging"] label p {{
            color: {OFF_WHITE} !important;
            font-weight: 700 !important;
            font-size: 13px !important;
        }}

        div[data-testid="stMultiSelect"] > div > div {{
            padding-left: 14px !important;
            border-radius: 12px !important;   /* reduce the curve so tags have less rounded corner to clash with */
            overflow: visible !important;      /* let tags render fully instead of being clipped */
        }}
        
        /* Give the first tag itself a bit of breathing room from the edge */
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"]:first-child {{
            margin-left: 6px !important;
        }}

        div[data-testid="stMultiSelect"] input::-webkit-contacts-auto-fill-button,
        div[data-testid="stMultiSelect"] input::-webkit-credentials-auto-fill-button,
        div[data-testid="stMultiSelect"] input::-webkit-caps-lock-indicator {{
            visibility: hidden !important;
            display: none !important;
            pointer-events: none !important;
            position: absolute !important;
            right: 0 !important;
        }}

        /* Push the whole value/tag row to the right, away from the left edge */
        div[data-testid="stMultiSelect"] > div > div {{
            padding-left: 0px !important;
        }}
        
        /* If tags are individually flexed, nudge the first one specifically */
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"]:first-child {{
            margin-left: 20px !important;
        }}

        /* EMERGING TOOLTIP */

        div[class*="st-key-trend_explorer_controls_emerging"] .te-tooltip {{
            color: {OFF_WHITE} !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            margin-bottom: 6px !important;
            display: inline-block !important;
            transform: translate(10px, -4px) !important;
        }}

        div[class*="st-key-te_emerging_min_latest_mentions"] label,
        div[class*="st-key-te_emerging_min_latest_mentions"] label *,
        div[class*="st-key-te_emerging_min_latest_mentions"] div[data-testid="stWidgetLabel"],
        div[class*="st-key-te_emerging_min_latest_mentions"] div[data-testid="stWidgetLabel"] * {{
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
            min-height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
        }}

        div[class*="st-key-te_emerging_min_latest_mentions"] {{
            margin-top: -4px !important;
            margin-left: 6px !important;
        }}
            
        div[class*="st-key-te_emerging_min_latest_mentions"] > div {{
            margin-top: 0px !important;
            padding-top: 0px !important;
        }}

        div[class*="st-key-trend_explorer_controls_emerging"] div[data-testid="stNumberInput"] label,
        div[class*="st-key-trend_explorer_controls_emerging"] div[data-testid="stTextInput"] label {{
            width: 100% !important;
            display: block !important;
            text-align: left !important;
        }}

        /* Trend explorer compare +/- buttons: small fixed circles */
        div[class*="st-key-trend_explorer_profile_select"] div[data-testid="stButton"] > button {{
            width: 34px !important;
            min-width: 34px !important;
            height: 34px !important;
            min-height: 34px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            background-color: #766161 !important;
            color: #f5f3f5 !important;
            font-size: 18px !important;
            font-weight: 900 !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}
        
        div[class*="st-key-trend_explorer_profile_select"] div[data-testid="stButton"] > button p {{
            font-size: 18px !important;
            font-weight: 900 !important;
            margin: 0 !important;
            }}
            
        div[class*="st-key-trend_explorer_profile_select"] div[data-testid="stButton"] > button:hover {{
            background-color: #f5f3f5 !important;
            color: #312421 !important;
            }}
            
        /* Align the button vertically with the dropdown (dropdowns have a label above them,
        buttons don't, so nudge the button down to match) */
        div[class*="st-key-trend_explorer_profile_select"] .te-reset-pad {{
            height: 35px;
        }}

        /* EMERGING FILTER CONTAINER */

        div[class*="st-key-trend_explorer_controls_emerging"] {{
            background-color: #473b39 !important;
            border-radius: 16px !important;
            padding: 18px 18px 16px 18px !important;
            border: none !important;
        }}

        /* Word cloud iframe should follow the second panel width. */
        .te-wordcloud-frame iframe {{
            max-width: calc(100vw - var(--side-menu-width) - 70px) !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}

        .te-wordcloud-frame iframe {{
            margin-top: 18px !important;
            margin-bottom: 48px !important;
        }}

        div[class*="st-key-lollipop_iframe_"] iframe {{
            max-width: 100% !important;
            width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }}

        .st-key-te_reset_filters div[data-testid="stButton"] > button {{
            background-color: {BROWN_CARD} !important;
            color: {OFF_WHITE} !important;
            border: 1px solid rgba(194, 220, 255, 0.28) !important;
            border-radius: 10px !important;
            height: 42px !important;
            min-height: 42px !important;
            margin-top: 0 !important;
            padding: 0 14px !important;
            font-size: 13px !important;
            font-weight: 900 !important;
        }}
        
        .st-key-te_reset_filters div[data-testid="stButton"] > button p {{
            color: {OFF_WHITE} !important;
            font-size: 13px !important;
            font-weight: 900 !important;
        }}
        
        .st-key-te_reset_filters div[data-testid="stButton"] > button:hover {{
            background-color: {OFF_WHITE} !important;
            color: {CARD_BG} !important;
            border: 1px solid {SOFT_BLUE} !important;
            }}
            
        .st-key-te_reset_filters div[data-testid="stButton"] > button:hover p {{
                color: {CARD_BG} !important;
        }}

        {active_summary_css}
        {active_comparison_css}
        {active_spotlight_css}

        </style>
        """,
        unsafe_allow_html=True,
    )




def _inject_top_category_css():
    st.markdown(
        f"""
        <style>
        .entity-card {{
            background-color: {CARD_BG};
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 14px;
            border: 1px solid rgba(194, 220, 255, 0.12);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        .entity-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.35);
        }}
        .entity-swatch-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .entity-swatch {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
            flex-shrink: 0;
        }}
        .entity-card-title {{
            font-weight: 700;
            font-size: 0.95rem;
            color: {SOFT_BLUE};
        }}
        .entity-card-desc {{
            font-size: 0.85rem;
            color: {SOFT_BLUE};
            opacity: 0.8;
            line-height: 1.55;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def _get_wordcloud_cached(df: pd.DataFrame, label: str):
    return render_wordcloud(df, label)


def _safe(value) -> str:
    return html.escape(str(value))


def _get_compare_style(index: int) -> dict:
    return COMPARE_TREND_STYLES[index % len(COMPARE_TREND_STYLES)]


def _set_te_sub_page(page_name: str) -> None:
    st.session_state["te_sub_page"] = page_name


def _page_header() -> None:
    """Top-of-page Trend Explorer subpage buttons."""
    if "te_sub_page" not in st.session_state:
        st.session_state["te_sub_page"] = "Trend Summary"

    with st.container(key="subpage_button_band_trend_explorer"):
        spacer_l, col_summary, col_comparison, col_spotlight, spacer_r = st.columns([0.5, 1.5, 1.5, 1.5, 0.5], gap="medium")
        with col_summary:
            if st.button(
                "Trend Summary",
                key="te_nav_summary",
                width="stretch",
                type="secondary",
            ):
                _set_te_sub_page("Trend Summary")
                st.rerun()
        with col_comparison:
            if st.button(
                "Trend Comparison",
                key="te_nav_comparison",
                width="stretch",
                type="secondary",
            ):
                _set_te_sub_page("Trend Comparison")
                st.rerun()
        with col_spotlight:
            if st.button(
                "Trend Spotlight",
                key="te_nav_spotlight",
                width="stretch",
                type="secondary",
            ):
                _set_te_sub_page("Trend Spotlight")
                st.rerun()

    current_sub_page = st.session_state.get("te_sub_page", "Trend Summary")
    description = SUBPAGE_DESCRIPTIONS.get(current_sub_page, "")
    
    with st.container(key="subpage_description_panel"):
        st.markdown(
            f"""
            <div class="subpage-description">
                {_safe(description)}
            </div>
            <div class="description-bottom-band"></div>
        """,
        unsafe_allow_html=True,
    )


def _section_title(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="te-section-subtitle">{_safe(subtitle)}</div>' if subtitle else ""
    line_html = '<div class="te-section-line"></div>' if subtitle else ""
    st.markdown(
        f'<div class="te-section-title">{_safe(title)}</div>{subtitle_html}{line_html}',
        unsafe_allow_html=True,
    )


def _tooltip_label(label: str, tooltip: str) -> str:
    return f'<span class="te-tooltip" data-tooltip="{_safe(tooltip)}">{_safe(label)}</span>'


def _note(message: str) -> None:
    st.markdown(f'<div class="te-note">{_safe(message)}</div>', unsafe_allow_html=True)


def _badge(category: str, bg_color: str | None = None, text_color: str | None = None) -> str:
    category = str(category).strip().upper()
    color = LABEL_COLORS.get(category, LIGHT_BLUE)
    label = category.replace("_", " ").title()
    badge_bg = f"background-color:{bg_color};" if bg_color else ""
    badge_color = text_color or LIGHT_BLUE
    return (
        f'<span class="te-badge" style="{badge_bg} color:{badge_color};">'
        f'<span class="te-dot" style="background-color:{color};"></span>{_safe(label)}</span>'
    )


def _status_badge(status: str, bg_color: str | None = None, text_color: str | None = None) -> str:
    color = text_color or STATUS_COLORS.get(status, SOFT_BLUE)
    bg_style = f"background-color:{bg_color};" if bg_color else ""
    icon = STATUS_ICON.get(status, "")
    return f'<span class="te-badge" style="{bg_style} color:{color};">{icon} {_safe(status)}</span>'


def _kpi_card(title: str, value: str, sub: str = "", title_tooltip: str | None = None) -> str:
    title_html = _tooltip_label(title, title_tooltip) if title_tooltip else _safe(title)
    return f"""
    <div class="te-card">
        <h4>{title_html}</h4>
        <div class="te-value">{_safe(value)}</div>
        <div class="te-sub">{sub}</div>
    </div>
    """



def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.0f}%"


def _fmt_score(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}"


# ════════════════════════════════════════════════════════════════
# DATA PREP
# ════════════════════════════════════════════════════════════════

def _pick(results: dict, sentiment_key: str, raw_key: str):
    df = results.get(sentiment_key)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        df = results.get(raw_key)
    return df


def _standardize_source(df, text_col, category_value_or_col):
    """Return a frame with columns: name, category, filter_label, date, score."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if text_col not in df.columns or DATE_COL not in df.columns:
        return None

    cols = [text_col, DATE_COL]

    if SCORE_COL in df.columns:
        cols.append(SCORE_COL)

    if category_value_or_col in df.columns and category_value_or_col not in cols:
        cols.append(category_value_or_col)

    if LABEL_COL in df.columns and LABEL_COL not in cols:
        cols.append(LABEL_COL)

    if LABELS_COL in df.columns and LABELS_COL not in cols:
        cols.append(LABELS_COL)

    d = df[cols].copy()
    d = d.rename(columns={text_col: "name"})

    if SCORE_COL not in d.columns:
        d[SCORE_COL] = 0.0

    if category_value_or_col in d.columns:
        d["category"] = d[category_value_or_col].astype(str).str.strip().str.upper()
    else:
        d["category"] = category_value_or_col

    if LABELS_COL in d.columns:
        d["filter_label"] = d[LABELS_COL].astype(str).str.strip().str.upper()
    elif LABEL_COL in d.columns:
        d["filter_label"] = d[LABEL_COL].astype(str).str.strip().str.upper()
    else:
        d["filter_label"] = d["category"]

    d["name"] = d["name"].astype(str).str.strip().str.lower()
    d = d[d["name"] != ""]

    d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
    d[SCORE_COL] = pd.to_numeric(d[SCORE_COL], errors="coerce").fillna(0.0)

    d = d.dropna(subset=[DATE_COL])

    return d[["name", "category", "filter_label", DATE_COL, SCORE_COL]]


@st.cache_data(show_spinner=False)
def _build_master(entities_df, phrases_df, trend_units_df):
    frames = [
        _standardize_source(entities_df, ENTITY_TEXT_COL, LABEL_COL),
        _standardize_source(phrases_df, PHRASE_TEXT_COL, "PHRASE"),
        _standardize_source(trend_units_df, TREND_UNIT_TEXT_COL, "TREND_UNIT"),
    ]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["name", "category", "filter_label", DATE_COL, SCORE_COL])
    return pd.concat(frames, ignore_index=True)



def _apply_category_filter(master_df, category_filter):
    if master_df.empty or category_filter == "ALL":
        return master_df

    if category_filter == "PHRASES":
        return master_df[master_df["category"] == "PHRASE"]

    if category_filter == "TREND UNITS":
        return master_df[master_df["category"] == "TREND_UNIT"]

    if "filter_label" not in master_df.columns:
        return master_df[master_df["category"] == category_filter]

    label_match = (
        master_df["filter_label"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.contains(str(category_filter).upper(), regex=False, na=False)
    )

    return master_df[
        (master_df["category"] == category_filter)
        | label_match
    ]

@st.cache_data(show_spinner=False)
def _filter_master(master_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, category_filter: str) -> pd.DataFrame:
    """Cacheable filter step for date range and category selection."""
    if master_df is None or master_df.empty:
        return pd.DataFrame(columns=["name", "category", "filter_label", DATE_COL, SCORE_COL])

    d = master_df[
        (master_df[DATE_COL] >= pd.Timestamp(start_date))
        & (master_df[DATE_COL] <= pd.Timestamp(end_date))
    ].copy()
    return _apply_category_filter(d, category_filter).copy()




def _classify_status(current, prev):
    if prev == 0 and current > 0:
        return "Emerging"
    if current == 0 and prev > 0:
        return "Fading"
    if prev == 0 and current == 0:
        return "Fading"
    change = ((current - prev) / prev) * 100
    if change >= RISING_THRESHOLD:
        return "Rising"
    if change <= DECLINING_THRESHOLD:
        return "Declining"
    return "Stable"

def _filter_wordcloud_df(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if DATE_COL in d.columns and start_date is not None and end_date is not None:
        d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
        d = d[
            (d[DATE_COL] >= pd.Timestamp(start_date)) &
            (d[DATE_COL] <= pd.Timestamp(end_date))
        ]

    return d


@st.cache_data(show_spinner=False)
def _aggregate(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame(
            columns=["name", "category", "mentions", "avg_sentiment", "weighted_sentiment",
                     "monthly_change_pct", "status", "peak_period"]
        )

    d = master_df.copy()
    d["period"] = d[DATE_COL].dt.to_period("M").dt.to_timestamp()

    overall = d.groupby("name").agg(
        mentions=("name", "count"),
        avg_sentiment=(SCORE_COL, "mean"),
        weighted_sentiment=(SCORE_COL, "sum"),
    ).reset_index()

    # Weighted Sentiment = frequency × sentiment score. With row-level sentiment
    # values, summing SCORE_COL across all mentions is equivalent to
    # mention frequency × average sentiment for the selected period.

    cat_mode = (
        d.groupby("name")["category"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    overall = overall.merge(cat_mode, on="name", how="left")

    peak = (
        d.groupby(["name", "period"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates(subset="name")
        .rename(columns={"period": "peak_period"})[["name", "peak_period"]]
    )
    overall = overall.merge(peak, on="name", how="left")

    periods_sorted = sorted(d["period"].dropna().unique())
    last_p = periods_sorted[-1] if len(periods_sorted) >= 1 else None
    prev_p = periods_sorted[-2] if len(periods_sorted) >= 2 else None

    counts = d.groupby(["name", "period"]).size().unstack(fill_value=0)

    def _count_at(name, p):
        if p is None or name not in counts.index or p not in counts.columns:
            return 0
        return int(counts.loc[name, p])

    overall["current_count"] = overall["name"].apply(lambda n: _count_at(n, last_p))
    overall["prev_count"] = overall["name"].apply(lambda n: _count_at(n, prev_p))
    overall["monthly_change_pct"] = overall.apply(
        lambda r: 100.0 if (r["prev_count"] == 0 and r["current_count"] > 0)
        else (0.0 if r["prev_count"] == 0 else ((r["current_count"] - r["prev_count"]) / r["prev_count"]) * 100),
        axis=1,
    )
    overall["status"] = overall.apply(
        lambda r: _classify_status(r["current_count"], r["prev_count"]), axis=1
    )
    return overall.sort_values("mentions", ascending=False).reset_index(drop=True)



# ════════════════════════════════════════════════════════════════
# FORECAST HELPERS
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _forecast_lookup(predictions, term, source_type, metric_type):
    if not predictions:
        return None
    key = f"{source_type}_{metric_type}"
    df = predictions.get(key)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "term" not in df.columns or "date" not in df.columns or "forecast_value" not in df.columns:
        return None
    sub = df[df["term"].astype(str).str.strip().str.lower() == str(term).strip().lower()]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["forecast_value"] = pd.to_numeric(sub["forecast_value"], errors="coerce")
    return sub.dropna(subset=["date", "forecast_value"]).sort_values("date")


def _category_to_source_type(category: str) -> str:
    if category == "PHRASE":
        return "phrase"
    if category == "TREND_UNIT":
        return "trend"
    return "entity"


# ════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════

def _build_profile_fig(trend_name: str, sub_df: pd.DataFrame, forecast_df: pd.DataFrame | None = None):
    if sub_df is None or sub_df.empty:
        return None

    d = sub_df.copy()
    d["period"] = d[DATE_COL].dt.to_period("M").dt.to_timestamp()
    monthly = (
        d.groupby("period")
        .agg(mentions=("name", "count"), sentiment=(SCORE_COL, "mean"))
        .reset_index()
        .sort_values("period")
    )

    if monthly.empty:
        return None

    title = f"{trend_name.title()} — Mentions and Sentiment"
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=monthly["period"],
            y=monthly["mentions"],
            name="Mentions",
            marker_color=CARD_BG,
            opacity=0.82,
            hovertemplate="%{x|%b %Y}<br>Mentions: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["period"],
            y=monthly["sentiment"],
            mode="lines+markers",
            name="Average sentiment",
            line=dict(color=OFF_WHITE, width=3),
            marker=dict(size=8, color=OFF_WHITE),
            hovertemplate="%{x|%b %Y}<br>Sentiment: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["forecast_value"],
                mode="lines+markers",
                name="Forecast sentiment",
                line=dict(color=MUTED_BLUE, width=3, dash="dash"),
                marker=dict(size=8, color=MUTED_BLUE),
                hovertemplate="%{x|%b %Y}<br>Forecast: %{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20, color=SOFT_BLUE)),
        paper_bgcolor= "#4b3d3a",
        plot_bgcolor= "#4b3d3a",
        font=dict(color=SOFT_BLUE, size=12),
        height=430,
        margin=dict(l=55, r=55, t=75, b=95),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=SOFT_BLUE, size=11),
        ),
    )

    fig.update_xaxes(
        title_text="Month",
        tickformat="%b %Y",
        gridcolor="rgba(194,220,255,0.12)",
        linecolor="rgba(194,220,255,0.28)",
        zerolinecolor="rgba(194,220,255,0.18)",
        tickfont=dict(color="rgba(245,243,245,0.95)", size=12),
        title_font=dict(color="rgba(245,243,245,0.95)", size=13),
    )
    fig.update_yaxes(
        title_text="Mentions",
        secondary_y=False,
        gridcolor="rgba(194,220,255,0.12)",
        linecolor="rgba(194,220,255,0.28)",
        zerolinecolor="rgba(194,220,255,0.18)",
        tickfont=dict(color="rgba(245,243,245,0.95)", size=12),
        title_font=dict(color="rgba(245,243,245,0.95)", size=13),
    )
    fig.update_yaxes(
        title_text="Sentiment",
        secondary_y=True,
        zeroline=True,
        zerolinecolor="rgba(194,220,255,0.55)",  # more visible
        zerolinewidth=1.5,
        gridcolor="rgba(0,0,0,0)",
        linecolor="rgba(194,220,255,0.28)",
        tickfont=dict(color="rgba(245,243,245,0.95)", size=12),
        title_font=dict(color="rgba(245,243,245,0.95)", size=13),
    )

    return fig



# ════════════════════════════════════════════════════════════════
# UI SECTIONS
# ════════════════════════════════════════════════════════════════

def _clear_explorer_filters(min_date, max_date):
    st.session_state["te_date_range"] = (min_date, max_date)
    st.session_state["te_category_filter"] = "ALL"
    st.session_state["te_show_forecast"] = False

def _render_filters(master_df):
    if master_df.empty:
        min_d, max_d = pd.Timestamp("2025-08-01"), pd.Timestamp("2026-01-31")
    else:
        min_d, max_d = master_df[DATE_COL].min(), master_df[DATE_COL].max()

    min_date = min_d.date()
    max_date = max_d.date()

    st.markdown('<div class="te-filter-title-card">Explorer Filters</div>', unsafe_allow_html=True)

    with st.container(key="trend_explorer_filters"):
        c1, c2, c3, c4 = st.columns([1.55, 1.05, 1.15, 0.62], gap="medium")

        with c1:
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="te_date_range",
            )

        with c2:
            category_filter = st.selectbox(
                "Trend category",
                CATEGORY_OPTIONS,
                index=0,
                key="te_category_filter",
            )

        with c3:
            show_forecast = st.checkbox(
                "Include Feb–Mar 2026 forecast",
                value=False,
                key="te_show_forecast",
            )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        filters_changed = (
            pd.Timestamp(start_date).date() != min_date
            or pd.Timestamp(end_date).date() != max_date
            or category_filter != "ALL"
            or show_forecast is True
        )

        with c4:
            st.markdown('<div class="te-reset-pad"></div>', unsafe_allow_html=True)
            if filters_changed:
                st.button(
                    "Reset",
                    key="te_reset_filters",
                    on_click=_clear_explorer_filters,
                    args=(min_date, max_date),
                )

    return pd.Timestamp(start_date), pd.Timestamp(end_date), category_filter, show_forecast

def _render_filters_ex(master_df):
    if master_df.empty:
        min_d, max_d = pd.Timestamp("2025-08-01"), pd.Timestamp("2026-01-31")
    else:
        min_d, max_d = master_df[DATE_COL].min(), master_df[DATE_COL].max()

    st.markdown('<div class="te-filter-title-card">Explorer Filters</div>', unsafe_allow_html=True)

    with st.container(key="trend_explorer_filters"):
        c1, c2, c3 = st.columns([1.55, 1.05, 1.15], gap="medium")

        with c1:
            date_range = st.date_input(
                "Date range",
                value=(min_d.date(), max_d.date()),
                min_value=min_d.date(),
                max_value=max_d.date(),
                key="te_date_range",
            )

        with c2:
            category_filter = st.selectbox(
                "Trend category",
                CATEGORY_OPTIONS,
                index=0,
                key="te_category_filter",
            )

        with c3:
            show_forecast = st.checkbox(
                "Include Feb–Mar 2026 forecast",
                value=False,
                key="te_show_forecast",
            )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d.date(), max_d.date()

    return pd.Timestamp(start_date), pd.Timestamp(end_date), category_filter, show_forecast



def _render_kpis(agg: pd.DataFrame):
    _section_title(
        "Trend Summary",
        "A quick overview of all trend signals in the selected data range.",
    )
    if agg.empty:
        _note("No trend data available for the selected filters.")
        return

    eligible = agg[agg["mentions"] >= MIN_MENTIONS]
    if eligible.empty:
        eligible = agg

    trend_eligible = eligible[eligible["category"] != "SEASON"]
    if trend_eligible.empty:
        trend_eligible = eligible

    top_trend = trend_eligible.sort_values("mentions", ascending=False).iloc[0]
    fastest = trend_eligible.sort_values("monthly_change_pct", ascending=False).iloc[0]
    most_positive = trend_eligible.sort_values("avg_sentiment", ascending=False).iloc[0]
    most_negative = trend_eligible.sort_values("avg_sentiment", ascending=True).iloc[0]
    most_discussed_positive = trend_eligible.sort_values("weighted_sentiment", ascending=False).iloc[0]
    most_discussed_negative = trend_eligible.sort_values("weighted_sentiment", ascending=True).iloc[0]

    season_subset = eligible[eligible["category"] == "SEASON"].sort_values("mentions", ascending=False)
    phrase_subset = eligible[eligible["category"] == "PHRASE"].sort_values("mentions", ascending=False)
    trend_unit_subset = eligible[eligible["category"] == "TREND_UNIT"].sort_values("mentions", ascending=False)

    top_season = season_subset.iloc[0] if not season_subset.empty else None
    top_phrase = phrase_subset.iloc[0] if not phrase_subset.empty else None
    top_trend_unit = trend_unit_subset.iloc[0] if not trend_unit_subset.empty else None

    cat_totals = agg.groupby("category")["mentions"].sum().sort_values(ascending=False)
    top_category = cat_totals.index[0] if not cat_totals.empty else "—"
    active_trends = int((agg["mentions"] >= MIN_MENTIONS).sum())

    def _tooltip_for(title: str) -> str:
        return SUMMARY_CARD_TOOLTIPS.get(title, "")

    def _render_card(col, title, value, sub):
        with col:
            st.markdown(
                _kpi_card(title, value, sub, title_tooltip=_tooltip_for(title)),
                unsafe_allow_html=True,
            )

    cards = [
        (
            "Active Trends",
            str(active_trends),
            f"≥ {MIN_MENTIONS} mentions in range",
        ),
        (
            "Top Trend",
            top_trend["name"].title(),
            f"{int(top_trend['mentions'])} mentions · {top_trend['category'].title()} · {_status_badge(top_trend['status'])}",
        ),
        (
            "Top Phrase",
            top_phrase["name"].title() if top_phrase is not None else "—",
            f"{int(top_phrase['mentions'])} mentions · {_status_badge(top_phrase['status'])}" if top_phrase is not None else "No phrase mentions in range",
        ),
        (
            "Top Trend Unit",
            top_trend_unit["name"].title() if top_trend_unit is not None else "—",
            f"{int(top_trend_unit['mentions'])} mentions · {_status_badge(top_trend_unit['status'])}" if top_trend_unit is not None else "No trend unit mentions in range",
        ),
        (
            "Most Discussed Season",
            top_season["name"].title() if top_season is not None else "—",
            f"{int(top_season['mentions'])} mentions · {_status_badge(top_season['status'])}" if top_season is not None else "No season mentions in range",
        ),
        (
            "Most Discussed Category",
            top_category.replace("_", " ").title(),
            f"{int(cat_totals.iloc[0])} total mentions" if not cat_totals.empty else "No mentions",
        ),
        (
            "Fastest Growing",
            fastest["name"].title(),
            f"{_fmt_pct(fastest['monthly_change_pct'])} MoM · {fastest['category'].title()}",
        ),
        (
            "Most Positive",
            most_positive["name"].title(),
            f"Sentiment {_fmt_score(most_positive['avg_sentiment'])} · {most_positive['category'].title()}",
        ),
        (
            "Most Negative",
            most_negative["name"].title(),
            f"Sentiment {_fmt_score(most_negative['avg_sentiment'])} · {most_negative['category'].title()}",
        ),
        (
            "Discussed Most Positively",
            most_discussed_positive["name"].title(),
            f"Weighted Sentiment {_fmt_score(most_discussed_positive['weighted_sentiment'])} · "
            f"{int(most_discussed_positive['mentions'])} mentions · {most_discussed_positive['category'].title()}",
        ),
        (
            "Discussed Most Negatively",
            most_discussed_negative["name"].title(),
            f"Weighted Sentiment {_fmt_score(most_discussed_negative['weighted_sentiment'])} · "
            f"{int(most_discussed_negative['mentions'])} mentions · {most_discussed_negative['category'].title()}",
        ),
    ]

    row_sizes = [3, 3, 3, 2]
    card_index = 0
    for row_size in row_sizes:
        cols = st.columns(row_size)
        for col in cols:
            if card_index >= len(cards):
                break
            _render_card(col, *cards[card_index])
            card_index += 1




def _clear_trend_search() -> None:
    st.session_state["te_search"] = ""


def _show_more_emerging() -> None:
    st.session_state["te_emerging_limit"] = int(st.session_state.get("te_emerging_limit", 5)) + 5

CARD_CATEGORIES = ENTITY_CATEGORIES + ["SEASON"]  # 8 + SEASON = 9 total, 3 per row


def _inject_top_category_css():
    st.markdown(
        f"""
        <style>
        .entity-card {{
            background-color: {CARD_BG};
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 14px;
            border: 1px solid rgba(194, 220, 255, 0.12);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        .entity-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.35);
        }}
        .entity-swatch-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .entity-swatch {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
            flex-shrink: 0;
        }}
        .entity-card-title {{
            font-weight: 700;
            font-size: 0.95rem;
            color: {SOFT_BLUE};
        }}
        .entity-card-desc {{
            font-size: 0.85rem;
            color: {OFF_WHITE};
            opacity: 0.88;
            line-height: 1.55;
        }}
        .entity-trend-name {{
            color: {OFF_WHITE};
            font-size: 1.03rem;
            font-weight: 900;
            opacity: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_top_by_category(agg_all: pd.DataFrame):
    """
    Top trend per label, 9 cards / 3 per row.

    IMPORTANT: pass in an `agg_all` that was aggregated from the date-filtered
    master data BEFORE the user's category filter is applied — this section
    is meant to always show all 9 categories side by side, regardless of
    which single category the sidebar filter is currently set to.
    """
    _section_title("Top Trend by Category")
    _inject_top_category_css()

    if agg_all.empty:
        st.info("No category data available for the selected date range.")
        return

    cols_per_row = 3
    for i in range(0, len(CARD_CATEGORIES), cols_per_row):
        row_categories = CARD_CATEGORIES[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, cat in zip(cols, row_categories):
            with col:
                subset = agg_all[agg_all["category"] == cat]
                colour = LABEL_COLORS.get(cat, SOFT_BLUE)

                if subset.empty:
                    title = "No trend yet"
                    desc = "Not enough mentions in this range."
                else:
                    top = subset.sort_values("mentions", ascending=False).iloc[0]
                    title = top["name"].title()
                    desc = (
                        f"{int(top['mentions'])} mentions · "
                        f"{_tooltip_label('Weighted Sentiment', WEIGHTED_SENTIMENT_TOOLTIP)} {_fmt_score(top['weighted_sentiment'])}<br>"
                        f"{_status_badge(top['status'])}"
                    )

                display_category = ENTITY_DISPLAY_NAMES.get(cat, cat.title())

                card_html = (
                    '<div class="entity-card">'
                    '<div class="entity-swatch-row">'
                    f'<div class="entity-swatch" style="background:{colour};"></div>'
                    f'<div class="entity-card-title">{_safe(display_category)}</div>'
                    '</div>'
                    f'<div class="entity-card-desc"><span class="entity-trend-name">{_safe(title)}</span><br>{desc}</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

def _render_leaderboard(
    agg: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    category_filter: str = "ALL",
) -> tuple[pd.DataFrame, str | None]:
    _section_title("Trend Leaderboard", "Search, sort, and select one trend for deeper inspection.")
    if start_date is not None and end_date is not None:
        date_start = pd.Timestamp(start_date).strftime('%d %b %Y')
        date_end = pd.Timestamp(end_date).strftime('%d %b %Y')
        category_text = str(category_filter).replace("_", " ").title()
        
        st.markdown(
            f"""
            <div style="width:100%; color:{OFF_WHITE}; font-size:0.8rem; margin:0 0 12px 0; text-align:center;">
                <span style="font-weight:900;">Filters applied:</span>
                <span style="font-weight:650;"> Date range:</span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(date_start)}</span>
                <span style="font-weight:650;"> to </span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(date_end)}</span>
                <span style="font-weight:650;"> · Trend category:</span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(category_filter)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if agg.empty:
        _note("No trends to show for the selected filters.")
        return pd.DataFrame(), None

    with st.container(key="trend_explorer_controls"):
        c1, c2, c3, c4 = st.columns([2.0, 1.0, 1.0, 0.62], gap="medium")
        with c1:
            search = st.text_input("Search trends", "", key="te_search")
        with c2:
            sort_by = st.selectbox(
                "Sort by",
                ["Mentions", "Average Sentiment", "Weighted Sentiment", "Monthly Change"],
                index=0,
                key="te_sort",
            )
        with c3:
            top_n = st.slider("Show top", 5, 50, 20, key="te_top_n")
        with c4:
            st.markdown('<div class="te-reset-pad"></div>', unsafe_allow_html=True)
            if str(search).strip():
                st.button("Reset", key="te_reset_search", on_click=_clear_trend_search)

    view = agg.copy()
    if search:
        view = view[view["name"].str.contains(search.strip().lower(), na=False)]

    sort_map = {
        "Mentions": "mentions",
        "Average Sentiment": "avg_sentiment",
        "Weighted Sentiment": "weighted_sentiment",
        "Monthly Change": "monthly_change_pct",
    }

    view = view.sort_values(sort_map[sort_by], ascending=False).head(top_n).reset_index(drop=True)

    if view.empty:
        _note("No trends match your search. Use Reset to clear the search box and return to the full leaderboard.")
        return pd.DataFrame(), None

    rows = []

    for _, r in view.iterrows():
        rows.append(
            "<tr>"
            f"<td><b>{_safe(str(r['name']).title())}</b></td>"
            f"<td>{_badge(r['category'])}</td>"
            f"<td>{int(r['mentions'])}</td>"
            f"<td>{_fmt_score(r['avg_sentiment'])}</td>"
            f"<td>{_fmt_score(r['weighted_sentiment'])}</td>"
            f"<td>{_fmt_pct(r['monthly_change_pct'])}</td>"
            f"<td>{_status_badge(r['status'])}</td>"
            "</tr>"
        )

    rows_html = "".join(rows)

    table_html = (
        '<div class="te-table-wrap">'
        '<table class="te-table">'
        '<thead>'
        '<tr>'
        '<th>Trend</th>'
        '<th>Category</th>'
        '<th>Mentions</th>'
        f'<th>{_tooltip_label("Average Sentiment", AVG_SENTIMENT_TOOLTIP)}</th>'
        f'<th>{_tooltip_label("Weighted Sentiment", WEIGHTED_SENTIMENT_TOOLTIP)}</th>'
        '<th>Monthly Change</th>'
        '<th>Status</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        '</div>'
    )

    st.markdown(table_html, unsafe_allow_html=True)

    return view, None


def _set_compare_count(count: int) -> None:
    st.session_state["te_compare_count"] = min(3, max(1, int(count)))


def _add_compare_trend() -> None:
    _set_compare_count(int(st.session_state.get("te_compare_count", 1)) + 1)


def _remove_compare_trend(slot: int) -> None:
    compare_count = int(st.session_state.get("te_compare_count", 1))

    if slot == 2 and compare_count >= 3:
        st.session_state["te_profile_choice_2"] = st.session_state.get("te_profile_choice_3")
        st.session_state.pop("te_profile_choice_3", None)
    elif slot == 2:
        st.session_state.pop("te_profile_choice_2", None)
    elif slot == 3:
        st.session_state.pop("te_profile_choice_3", None)

    _set_compare_count(compare_count - 1)


def _profile_selector_row(
    label: str,
    choices: list[str],
    state_key: str,
    default_index: int,
    button_label: str | None = None,
    button_key: str | None = None,
    button_callback=None,
    button_args: tuple = (),
) -> str:
    if st.session_state.get(state_key) not in choices:
        st.session_state[state_key] = choices[min(default_index, len(choices) - 1)]

    with st.container(key=f"trend_explorer_profile_select_{state_key}"):
        select_col, button_col, spacer_col = st.columns([0.82, 0.08, 0.10], gap="small")
        with select_col:
            selected = st.selectbox(
                label,
                choices,
                index=choices.index(st.session_state[state_key]),
                key=state_key,
            )
        with button_col:
            st.markdown('<div class="te-reset-pad"></div>', unsafe_allow_html=True)
            if button_label and button_key and button_callback:
                st.button(button_label, key=button_key, on_click=button_callback, args=button_args)

    return selected


def _render_profile_selectors(view: pd.DataFrame, clicked_trend: str | None = None) -> list[str]:
    _section_title(
        "Trend Profile",
        "Inspect the selected trend's category, mentions, raw weighted sentiment, peak month, movement status, related terms, and monthly profile chart.",
    )

    if view is None or view.empty:
        return []

    choices = view["name"].str.title().tolist()
    if not choices:
        return []

    if clicked_trend:
        st.session_state["te_profile_choice"] = clicked_trend.title()

    if "te_compare_count" not in st.session_state:
        st.session_state["te_compare_count"] = 1

    compare_count = int(st.session_state.get("te_compare_count", 1))
    compare_count = min(3, max(1, compare_count))
    st.session_state["te_compare_count"] = compare_count

    state_keys = [
        "te_profile_choice",
        "te_profile_choice_2",
        "te_profile_choice_3",
    ]

    # Keep selected values valid when the leaderboard/search result changes.
    for i in range(compare_count):
        default_index = min(i, len(choices) - 1)
        if st.session_state.get(state_keys[i]) not in choices:
            st.session_state[state_keys[i]] = choices[default_index]

    selected = []

    # One single filter box for all profile dropdowns.
    with st.container(key="trend_explorer_profile_select"):
        column_widths = []

        for i in range(compare_count):
            column_widths.append(1.0)   # dropdown width

            # Each extra dropdown gets its own subtract button.
            if i > 0:
                column_widths.append(0.12)

        # One plus button at the end until 3 dropdowns are visible.
        if compare_count < 3:
            column_widths.append(0.12)

        cols = st.columns(column_widths, gap="small")

        col_index = 0

        for i in range(compare_count):
            with cols[col_index]:
                label = "View trend profile for:" if compare_count == 1 else f"Trend {i + 1}"
                selected_value = st.selectbox(
                    label,
                    choices,
                    index=choices.index(st.session_state[state_keys[i]]),
                    key=state_keys[i],
                )
                selected.append(selected_value)

            col_index += 1

            if i > 0:
                with cols[col_index]:
                    st.markdown('<div class="te-reset-pad"></div>', unsafe_allow_html=True)
                    st.button(
                        "-",
                        key=f"te_remove_compare_{i + 1}",
                        on_click=_remove_compare_trend,
                        args=(i + 1,),
                    )
                col_index += 1

        if compare_count < 3:
            with cols[col_index]:
                st.markdown('<div class="te-reset-pad"></div>', unsafe_allow_html=True)
                st.button(
                    "+",
                    key="te_add_compare",
                    on_click=_add_compare_trend,
                )

    unique_selected = []
    for item in selected:
        normalized = str(item).strip().lower()
        if normalized and normalized not in unique_selected:
            unique_selected.append(normalized)

    return unique_selected



def _get_common_phrases_for_trend(trend_name: str, phrase_source_df: pd.DataFrame, top_n: int = 30) -> pd.Series:
    if phrase_source_df is None or not isinstance(phrase_source_df, pd.DataFrame) or phrase_source_df.empty:
        return pd.Series(dtype=int)

    if "category" in phrase_source_df.columns and "name" in phrase_source_df.columns:
        d = phrase_source_df[phrase_source_df["category"] == "PHRASE"].copy()
        phrase_col = "name"
    elif PHRASE_TEXT_COL in phrase_source_df.columns:
        d = phrase_source_df.copy()
        phrase_col = PHRASE_TEXT_COL
    else:
        return pd.Series(dtype=int)

    pattern = str(trend_name).strip().lower()
    if not pattern:
        return pd.Series(dtype=int)

    phrases = d[phrase_col].astype(str).str.strip().str.lower()
    hits = phrases[phrases.str.contains(pattern, regex=False, na=False)]
    if hits.empty:
        return pd.Series(dtype=int)
    return hits.value_counts().head(top_n)




def _strip_entity_from_phrase(phrase: str, entity_name: str) -> str:
    """Remove the trend's own name from a phrase, collapse whitespace, title-case the rest.
    e.g. 'chocolate brown sweater' + 'chocolate brown' -> 'Sweater'
    """
    if not isinstance(phrase, str) or not phrase.strip():
        return ""
    pattern = re.compile(re.escape(entity_name.strip()), re.IGNORECASE)
    cleaned = pattern.sub("", phrase)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def _get_common_phrases_for_trend_cleaned(
    trend_name: str, phrase_source_df: pd.DataFrame, top_n: int = 12
) -> pd.Series:
    """Top phrases containing trend_name, with trend_name stripped out and
    duplicate results (after stripping) merged/summed together."""
    if phrase_source_df is None or phrase_source_df.empty or PHRASE_TEXT_COL not in phrase_source_df.columns:
        return pd.Series(dtype=int)

    hits = phrase_source_df[
        phrase_source_df[PHRASE_TEXT_COL].astype(str).str.lower().str.contains(trend_name.lower(), na=False)
    ]
    if hits.empty:
        return pd.Series(dtype=int)

    cleaned = hits[PHRASE_TEXT_COL].astype(str).apply(lambda p: _strip_entity_from_phrase(p, trend_name))
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return pd.Series(dtype=int)

    return cleaned.value_counts().head(top_n)


def _lollipop_fig_vertical(title: str, counts: pd.Series, style: dict, height: int = 420) -> go.Figure:
    chart_color = style.get("card_bg", SOFT_BLUE)
    #card_bg = style.get("card_bg", APP)
    card_bg = APP_BG
    font_color = style.get("card_font", OFF_WHITE)
    title_color = style.get("title_font", font_color)

    fig = go.Figure()

    if counts is None or counts.empty:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16, color=title_color)),
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            height=height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="No common phrases found for this trend.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13, color=font_color),
            )],
        )
        return fig

    counts = counts.sort_values(ascending=False)  # biggest first, left to right
    words = counts.index.tolist()
    values = counts.values.tolist()
    max_val = max(values)

    # stems
    for word, val in zip(words, values):
        fig.add_shape(
            type="line",
            x0=word, x1=word, y0=0, y1=val,
            line=dict(color=chart_color, width=3),
        )

    # dots + value labels
    fig.add_trace(
        go.Scatter(
            x=words,
            y=values,
            mode="markers+text",
            marker=dict(size=16, color=chart_color, line=dict(width=1.5, color=font_color)),
            text=[str(v) for v in values],
            textposition="top center",
            textfont=dict(color=font_color, size=12),
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16, color=title_color)),
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        height=height,
        margin=dict(l=40, r=20, t=60, b=80),
        showlegend=False,
    )
    fig.update_xaxes(
        tickangle=-30,
        tickfont=dict(color=font_color, size=12),
        gridcolor="rgba(0,0,0,0)",
        linecolor="rgba(0,0,0,0.15)",
    )
    fig.update_yaxes(
        visible=False,
        range=[0, max_val * 1.25],
    )
    return fig



def _truncate_label(text: str, max_len: int = 18) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _estimate_left_margin(words: list, px_per_char: int = 7, base: int = 30, cap: int = 200) -> int:
    """Manual replacement for automargin — predictable instead of runaway."""
    longest = max((len(str(w)) for w in words), default=0)
    return int(min(cap, base + longest * px_per_char))

def _add_lollipop_title_badge(
    fig: go.Figure,
    title: str,
    badge_bg: str,
    badge_text: str,
    font_size: int = 13,
) -> None:
    title_text = str(title)

    # Estimate the pixel width the text actually needs, then size the badge
    # in real pixels — not as a fraction of the figure's (variable) width.
    avg_char_px = font_size * 0.62  # rough width of a bold char at this font size
    text_width_px = sum(avg_char_px * (0.55 if ch == " " else 1.0) for ch in title_text)

    padding_x = 22       # horizontal breathing room on each side of the text
    min_badge_width = 70  # floor so very short titles don't look like a stub
    badge_width_px = max(min_badge_width, text_width_px + padding_x * 2)
    badge_height_px = 30
    corner_r = 10  # pixel radius — stays a true circle now, since everything's in px

    half_w = badge_width_px / 2
    half_h = badge_height_px / 2
    y_anchor = 1.117  # paper-fraction anchor point, same spot as before

    x0, x1 = -half_w, half_w
    y0, y1 = -half_h, half_h
    r = corner_r

    rounded_path = (
        f"M {x0 + r},{y0} "
        f"L {x1 - r},{y0} "
        f"Q {x1},{y0} {x1},{y0 + r} "
        f"L {x1},{y1 - r} "
        f"Q {x1},{y1} {x1 - r},{y1} "
        f"L {x0 + r},{y1} "
        f"Q {x0},{y1} {x0},{y1 - r} "
        f"L {x0},{y0 + r} "
        f"Q {x0},{y0} {x0 + r},{y0} "
        f"Z"
    )

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        xsizemode="pixel", ysizemode="pixel",
        xanchor=0.5, yanchor=y_anchor,
        path=rounded_path,
        fillcolor=badge_bg,
        line=dict(color="rgba(0,0,0,0)", width=0),
        layer="above",
    )

    fig.add_annotation(
        text=f"<b>{_safe(title_text)}</b>",
        x=0.5, y=y_anchor,
        xref="paper", yref="paper",
        xanchor="center", yanchor="middle",
        showarrow=False,
        font=dict(size=font_size, color=badge_text),
    )

def _lollipop_fig(title: str, counts: pd.Series, style: dict, height: int = 420) -> go.Figure:
    
    chart_color = style.get("card_bg", SOFT_BLUE)
    card_bg = FIGURE_BG
    font_color = OFF_WHITE
    line_color = style.get("card_bg", font_color)
    title_badge_bg = APP_BG
    title_badge_text = style.get("lolipop_title", font_color)

    fig = go.Figure()

    if counts is None or counts.empty:
        fig.update_layout(
            margin=dict(l=40, r=40, t=80, b=40),
            paper_bgcolor=card_bg, plot_bgcolor=card_bg, height=height,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(
                text="No common phrases found for this trend.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13, color=font_color),
            )],
        )
        _add_lollipop_title_badge(fig, title, title_badge_bg, title_badge_text)
        return fig

    counts = counts.sort_values(ascending=True)
    raw_words = counts.index.tolist()
    words = [_truncate_label(w) for w in raw_words]
    values = counts.values.tolist()
    max_val = max(values)

    left_margin = _estimate_left_margin(words)

    # stems
    for word, val in zip(words, values):
        fig.add_shape(
            type="line", x0=0, x1=val, y0=word, y1=word,
            line=dict(color=chart_color, width=3),
        )

    # dots only — no data-space text, so range doesn't need to reserve room for labels
    fig.add_trace(
        go.Scatter(
            x=values, y=words, mode="markers",
            marker=dict(size=16, color=chart_color, line=dict(width=1.5, color=line_color)),
            customdata=list(zip(raw_words, values)),
            hovertemplate="%{customdata[0]}<br>Count: %{customdata[1]}<extra></extra>",
            showlegend=False,
        )
    )

    # value labels as fixed-pixel-offset annotations — independent of data scale
    for word, val in zip(words, values):
        fig.add_annotation(
            x=val, y=word, xref="x", yref="y",
            text=str(val), showarrow=False,
            xanchor="left", xshift=12,
            font=dict(color=font_color, size=12),
        )

    fig.update_layout(
        margin=dict(l=left_margin, r=55, t=85, b=30),  # fixed right margin, calculated left margin
        paper_bgcolor=card_bg, plot_bgcolor=card_bg, height=height,
        showlegend=False,
    )
    _add_lollipop_title_badge(fig, title, title_badge_bg, title_badge_text)

    fig.update_xaxes(
        visible=False,
        range=[0, max_val * 1.08],  # small, fixed padding — annotations no longer live in this space
    )
    fig.update_yaxes(
        tickfont=dict(color=font_color, size=12),
        ticklabelstandoff=6,
        gridcolor="rgba(0,0,0,0)",
        linecolor=line_color,
        automargin=False,  # replaced by manual left_margin above
    )
    return fig




def _render_single_lollipop_scrollable(fig: go.Figure, words: list, left_margin: int, height: int = 440) -> None:
    """Wrap the chart in a horizontally scrollable container. Forces Plotly to
    re-measure its container via ResizeObserver, since components.html's iframe
    never fires a window 'resize' event that Plotly's own responsive:true relies on."""
    fig.update_layout(autosize=True)
    div_id = f"lollipop_{abs(hash(str(words) + str(left_margin))) % 10_000_000}"
    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        default_width="100%",
        default_height=f"{height}px",
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )
    content_min_width = max(320, left_margin + 220)
    wrapped = f"""
    <div style="width:100%; overflow-x:auto; overflow-y:hidden; background:{FIGURE_BG}; border-radius:10px;">
        <div id="{div_id}_wrap" style="min-width:{content_min_width}px; width:100%;">
            {plot_html}
        </div>
    </div>
    <script>
    (function() {{
        var gd = document.getElementById("{div_id}");
        if (!gd) return;
        function forceResize() {{
            if (window.Plotly && gd) {{
                Plotly.Plots.resize(gd);
            }}
        }}
        // Window resize events won't fire inside this iframe, so watch the
        // actual container element instead.
        var wrapEl = document.getElementById("{div_id}_wrap");
        if (window.ResizeObserver && wrapEl) {{
            var ro = new ResizeObserver(function() {{ forceResize(); }});
            ro.observe(wrapEl);
        }}
        // Fallback in case layout settles slightly after initial mount.
        window.addEventListener('load', forceResize);
        setTimeout(forceResize, 60);
        setTimeout(forceResize, 300);
    }})();
    </script>
    """
    components.html(wrapped, height=height + 20, scrolling=False)




def _render_lollipop_comparison(trend_names: list[str], phrase_source_df: pd.DataFrame) -> None:
    valid_names = [name for name in trend_names if name]
    if not valid_names:
        return
    _section_title("Common Phrases", "Top phrases the selected trend(s) appears in - these are the words most used in combination with the selected trend(s). ")
    cols = st.columns(len(valid_names))
    for i, (col, trend_name) in enumerate(zip(cols, valid_names)):
        style = _get_compare_style(i)
        counts = _get_common_phrases_for_trend_cleaned(trend_name, phrase_source_df, top_n=12)
        words = [_truncate_label(w) for w in counts.index.tolist()] if counts is not None and not counts.empty else []
        left_margin = _estimate_left_margin(words) if words else 30
        with col:
            fig = _lollipop_fig(trend_name.title(), counts, style)
            _render_single_lollipop_scrollable(fig, words, left_margin)

def _profile_card_html(
    row: pd.Series,
    phrases_source_df: pd.DataFrame,
    trend_units_df: pd.DataFrame,
    style_index: int = 0,
) -> str:
    trend_name = str(row["name"])
    peak = row["peak_period"]
    peak_str = peak.strftime("%B %Y") if pd.notna(peak) else "—"
    style = _get_compare_style(style_index)
    card_bg = style["card_bg"]
    card_font = style["card_font"]
    title_font = style["title_font"]
    profile_label_font = style.get("profile_label_font", card_font)
    profile_badge_bg = style.get("profile_badge_bg")
    profile_badge_font = style.get("profile_badge_font")
    status_badge_bg = style.get("status_badge_bg")
    status_badge_font = style.get("status_badge_font")


    related = []
    if trend_units_df is not None and isinstance(trend_units_df, pd.DataFrame) and not trend_units_df.empty and TREND_UNIT_TEXT_COL in trend_units_df.columns:
        hits = trend_units_df[
            trend_units_df[TREND_UNIT_TEXT_COL].astype(str).str.lower().str.contains(trend_name, na=False)
        ]
        related_words = set()
        for val in hits[TREND_UNIT_TEXT_COL].astype(str).str.lower():
            for w in val.split():
                if w != trend_name:
                    related_words.add(w)
        related = list(related_words)[:5]

    common_counts = _get_common_phrases_for_trend(trend_name, phrases_source_df, top_n=4)
    common_phrases = common_counts.index.tolist() if common_counts is not None and not common_counts.empty else []

    category_badge = _badge(row["category"], bg_color=profile_badge_bg, text_color=profile_badge_font)
    status_badge = _status_badge(row["status"], bg_color=status_badge_bg, text_color=status_badge_font)

    def _profile_row(label: str, value_html: str) -> str:
        return (
            f'<div class="te-profile-row" style="color:{card_font}; border-bottom:1px solid rgba(49, 36, 33, 0.18);">'
            f'<span style="color:{profile_label_font}; opacity:0.82; font-weight:900;">{label}</span>'
            f'<b style="color:{card_font}; font-weight:900;">{value_html}</b>'
            '</div>'
        )

    return f"""
    <div class="te-card" style="background-color:{card_bg}; color:{card_font}; border:1px solid rgba(49, 36, 33, 0.22);">
        <h4 style="color:{title_font}; opacity:1.0;">{_safe(row['name'].title())}</h4>
        {_profile_row('Category', category_badge)}
        {_profile_row('Total mentions', str(int(row['mentions'])))}
        {_profile_row(_tooltip_label('Average Sentiment', AVG_SENTIMENT_TOOLTIP), _fmt_score(row['avg_sentiment']))}
        {_profile_row(_tooltip_label('Weighted Sentiment', WEIGHTED_SENTIMENT_TOOLTIP), _fmt_score(row['weighted_sentiment']))}
        {_profile_row('Peak month', _safe(peak_str))}
        {_profile_row('Monthly change', _fmt_pct(row['monthly_change_pct']))}
        {_profile_row('Status', status_badge)}
        {_profile_row('Related trends', _safe(', '.join(related).title() if related else '—'))}
        {_profile_row('Common phrases', _safe(', '.join(common_phrases).title() if common_phrases else '—'))}
    </div>
    """

def _build_comparison_fig(trend_names: list[str], master_df: pd.DataFrame, predictions: dict | None, show_forecast: bool):
    if not trend_names or master_df is None or master_df.empty:
        return None

    d = master_df[master_df["name"].isin(trend_names)].copy()
    if d.empty:
        return None

    d["period"] = d[DATE_COL].dt.to_period("M").dt.to_timestamp()
    monthly = (
        d.groupby(["period", "name"])
        .agg(mentions=("name", "count"), sentiment=(SCORE_COL, "mean"))
        .reset_index()
        .sort_values("period")
    )

    if monthly.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, trend_name in enumerate(trend_names):
        sub = monthly[monthly["name"] == trend_name]
        style = _get_compare_style(i)
        colour = style["chart"]

        mentions_legend = (
            f"<span style='color:{colour}; font-weight:800;'>"
            f"{trend_name.title()} Mentions"
            f"</span>"
        )
        sentiment_legend = (
            f"<span style='color:{colour}; font-weight:800;'>"
            f"{trend_name.title()} Sentiment"
            f"</span>"
        )
        forecast_legend = (
            f"<span style='color:{colour}; font-weight:800;'>"
            f"{trend_name.title()} Forecast"
            f"</span>"
        )

        fig.add_trace(
            go.Bar(
                x=sub["period"],
                y=sub["mentions"],
                name=mentions_legend,
                marker_color=colour,
                opacity=0.83,
                offsetgroup=trend_name,
                hovertemplate="%{x|%b %Y}<br>Mentions: %{y}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["period"],
                y=sub["sentiment"],
                mode="lines+markers",
                name=sentiment_legend,
                line=dict(color=colour, width=3),
                marker=dict(size=8, color=colour),
                hovertemplate="%{x|%b %Y}<br>Sentiment: %{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

        if show_forecast and predictions:
            row_category = d[d["name"] == trend_name]["category"].mode()
            category = row_category.iloc[0] if not row_category.empty else ""
            source_type = _category_to_source_type(category)
            fc = _forecast_lookup(predictions, trend_name, source_type, "sentiment")
            if fc is not None and not fc.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fc["date"],
                        y=fc["forecast_value"],
                        mode="lines+markers",
                        name=forecast_legend,
                        line=dict(color=colour, width=3, dash="dash"),
                        marker=dict(size=8, color=colour),
                        hovertemplate="%{x|%b %Y}<br>Forecast: %{y:.2f}<extra></extra>",
                    ),
                    secondary_y=True,
                )

    fig.update_layout(
        title=dict(text="Trend Comparison — Mentions and Sentiment", x=0.5, xanchor="center", font=dict(size=20, color=SOFT_BLUE)),
        paper_bgcolor="#4b3d3a",
        plot_bgcolor="#4b3d3a",
        font=dict(color=OFF_WHITE, size=12),
        height=470,
        margin=dict(l=55, r=55, t=75, b=115),
        hovermode="x unified",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(245,243,245,0.25)",
            font=dict(color=SOFT_BLUE, size=10),
        ),
    )
    fig.update_xaxes(
        title_text="Month",
        tickformat="%b %Y",
        gridcolor="rgba(194,220,255,0.12)",
        linecolor="rgba(194,220,255,0.28)",
        zerolinecolor="rgba(194,220,255,0.18)",
        tickfont=dict(color="rgba(245,243,245,0.95)", size=12),
        title_font=dict(color="rgba(245,243,245,0.95)", size=13),
    )
    fig.update_yaxes(
        title_text="Mentions", 
        secondary_y=False, 
        gridcolor="rgba(194,220,255,0.12)",
        tickfont=dict(color="rgba(245,243,245,0.95)", size=12),
        title_font=dict(color="rgba(245,243,245,0.95)", size=13),
    )
    fig.update_yaxes(
        title_text="Sentiment", 
        secondary_y=True, 
        zeroline=True, 
        zerolinecolor="rgba(194,220,255,0.55)", 
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(color="rgba(245,243,245,0.95)", size=12),
        title_font=dict(color="rgba(245,243,245,0.95)", size=13),
    )
    return fig

def _render_profile(
    trend_names, agg, master_df, phrases_df, trend_units_df, predictions, show_forecast
):
    if isinstance(trend_names, str):
        trend_names = [trend_names]
    trend_names = [name for name in trend_names if name]
    if not trend_names:
        return

    rows = []
    for trend_name in trend_names:
        row = agg[agg["name"] == trend_name]
        if not row.empty:
            rows.append(row.iloc[0])

    if not rows:
        return

    if len(rows) == 1:
        left, right = st.columns([1, 1.4], gap="large")
        with left:
        
            st.markdown(_profile_card_html(rows[0], phrases_df, trend_units_df, style_index=0), unsafe_allow_html=True)
            if show_forecast:
                source_type = _category_to_source_type(rows[0]["category"])
                fc = _forecast_lookup(predictions, rows[0]["name"], source_type, "sentiment")
                if fc is not None and not fc.empty:
                    avg_forecast = fc["forecast_value"].mean()
                    st.markdown(
                        f'<div class="te-forecast-note">Forecasted sentiment (Feb–Mar 2026): <b>{avg_forecast:.2f}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="te-forecast-note">No forecast available for this trend.</div>',
                        unsafe_allow_html=True,
                    )
        with right:
            sub_df = master_df[master_df["name"] == rows[0]["name"]].copy()
            source_type = _category_to_source_type(rows[0]["category"])
            fc = _forecast_lookup(predictions, rows[0]["name"], source_type, "sentiment") if show_forecast else None
            fig = _build_profile_fig(rows[0]["name"], sub_df, fc)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            else:
                _note("No chart data is available for this selected trend.")
        _render_lollipop_comparison([rows[0]["name"]], phrases_df)
        return

    cols = st.columns(len(rows))
    for i, (col, row) in enumerate(zip(cols, rows)):
        with col:
            st.markdown(_profile_card_html(row, phrases_df, trend_units_df, style_index=i), unsafe_allow_html=True)

    fig = _build_comparison_fig([str(row["name"]) for row in rows], master_df, predictions, show_forecast)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        _note("No chart data is available for the selected trend comparison.")
        
    _render_lollipop_comparison([str(row["name"]) for row in rows], phrases_df)


def _render_emerging(
    agg: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    category_filter: str = "ALL",
):
    _section_title(
        "Emerging Trends",
        "Use the filters below to control how strict the emerging trend detection should be.",
    )
    if agg.empty:
        return
    
    if start_date is not None and end_date is not None:
        date_start = pd.Timestamp(start_date).strftime('%d %b %Y')
        date_end = pd.Timestamp(end_date).strftime('%d %b %Y')
        category_text = str(category_filter).replace("_", " ").title()
        
        st.markdown(
            f"""
            <div style="width:100%; color:{OFF_WHITE}; font-size:0.8rem; margin:0 0 12px 0; text-align:center;">
                <span style="font-weight:900;">Filters applied:</span>
                <span style="font-weight:650;"> Date range:</span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(date_start)}</span>
                <span style="font-weight:650;"> to </span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(date_end)}</span>
                <span style="font-weight:650;"> · Trend category:</span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(category_filter)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    emerging_pool = agg.copy()

    if "current_count" in emerging_pool.columns:
        emerging_pool["latest_month_mentions"] = pd.to_numeric(
            emerging_pool["current_count"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        emerging_pool["latest_month_mentions"] = pd.to_numeric(
            emerging_pool["mentions"], errors="coerce"
        ).fillna(0).astype(int)

    if "prev_count" in emerging_pool.columns:
        emerging_pool["previous_month_mentions"] = pd.to_numeric(
            emerging_pool["prev_count"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        emerging_pool["previous_month_mentions"] = 0

    emerging_pool["mention_increase"] = (
        emerging_pool["latest_month_mentions"] - emerging_pool["previous_month_mentions"]
    )

    max_total_mentions = max(1, int(emerging_pool["mentions"].max()))
    max_latest_mentions = max(1, int(emerging_pool["latest_month_mentions"].max()))
    max_mention_increase = max(0, int(emerging_pool["mention_increase"].max()))
    max_growth_pct = max(100, int(np.ceil(emerging_pool["monthly_change_pct"].max())))


    

    with st.container(key="trend_explorer_controls_emerging"):
        c1, c2, c3, c4 = st.columns(4, gap="large")

        with c1:
            min_total_mentions = st.number_input(
                "Min total mentions",
                min_value=1,
                max_value=max_total_mentions,
                value=min(10, max_total_mentions),
                step=1,
                key="te_emerging_min_total_mentions",
            )
            
        with c2:

            st.markdown(
                _tooltip_label(
                    "Min latest-month mentions",
                    "The minimum number of times a trend must appear in the latest month of the selected date range."
                    ),
                    unsafe_allow_html=True,
                )
            min_latest_mentions = st.number_input(
                "Min latest-month mentions",
                min_value=1,
                max_value=max_latest_mentions,
                value=min(5, max_latest_mentions),
                step=1,
                key="te_emerging_min_latest_mentions",
                label_visibility="collapsed",
                )
            

        #c3, c4 = st.columns(2, gap="medium")
        
        with c3:
            min_mention_increase = st.number_input(
                "Min mention increase",
                min_value=0,
                max_value=max_mention_increase,
                value=min(3, max_mention_increase),
                step=1,
                key="te_emerging_min_mention_increase",
            )

        with c4:
            min_growth_pct = st.number_input(
                "Min growth %",
                min_value=0,
                max_value=max_growth_pct,
                value=min(25, max_growth_pct),
                step=5,
                key="te_emerging_min_growth_pct",
            )

        c5, c6, c7 = st.columns([1.7, 0.9, 0.75], gap="medium")

        with c5:
            selected_statuses = st.multiselect(
                "Trend status",
                ["Emerging", "Rising", "Stable", "Declining", "Fading"],
                default=["Emerging", "Rising"],
                key="te_emerging_status_filter",
            )

        with c6:
            sentiment_filter = st.selectbox(
                "Sentiment filter",
                ["Positive only", "Any sentiment", "Negative only"],
                index=0,
                key="te_emerging_sentiment_filter",
            )

        with c7:
            rank_by = st.selectbox(
                "Rank by",
                ["Growth %", "Mention Increase", "Latest-Month Mentions", "Weighted Sentiment"],
                index=0,
                key="te_emerging_rank_by",
            )

    if not selected_statuses:
        _note("Select at least one trend status to view emerging trends.")
        return

    emerging_all = emerging_pool[
        (emerging_pool["status"].isin(selected_statuses))
        & (emerging_pool["mentions"] >= int(min_total_mentions))
        & (emerging_pool["latest_month_mentions"] >= int(min_latest_mentions))
        & (emerging_pool["mention_increase"] >= int(min_mention_increase))
        & (emerging_pool["monthly_change_pct"] >= float(min_growth_pct))
    ].copy()

    if sentiment_filter == "Positive only":
        emerging_all = emerging_all[emerging_all["weighted_sentiment"] > 0]
    elif sentiment_filter == "Negative only":
        emerging_all = emerging_all[emerging_all["weighted_sentiment"] < 0]

    sort_map = {
        "Growth %": "monthly_change_pct",
        "Mention Increase": "mention_increase",
        "Latest-Month Mentions": "latest_month_mentions",
        "Weighted Sentiment": "weighted_sentiment",
    }
    emerging_all = emerging_all.sort_values(
        [sort_map[rank_by], "mentions"], ascending=[False, False]
    )

    if emerging_all.empty:
        _note("No emerging trends match the current filters. Try lowering the mention or growth thresholds.")
        return

    visible_count = min(int(st.session_state.get("te_emerging_limit", 5)), len(emerging_all))
    emerging = emerging_all.head(visible_count)

    cols_per_row = 5
    for i in range(0, len(emerging), cols_per_row):
        row_items = emerging.iloc[i:i + cols_per_row]
        cols = st.columns(len(row_items)) if len(row_items) <= cols_per_row else st.columns(cols_per_row)
        for col, (_, r) in zip(cols, row_items.iterrows()):
            increase_text = f"+{int(r['mention_increase'])}" if int(r["mention_increase"]) > 0 else str(int(r["mention_increase"]))
            with col:
                st.markdown(
                    f"""
                    <div class="te-emerging-card">
                        <div style="font-weight:900; color:{OFF_WHITE};">{_safe(r['name'].title())}</div>
                        <div style="margin-top:6px;">{_badge(r['category'])}</div>
                        <div style="margin-top:8px; font-size:0.85rem; color:{OFF_WHITE}; opacity:0.95; font-weight:700; line-height:1.45;">
                        <span style="color:{SOFT_BLUE}; font-weight:900;">{int(r['mentions'])}</span> total mentions<br>
                        <span style="color:{SOFT_BLUE}; font-weight:900;">{int(r['latest_month_mentions'])}</span> latest-month mentions · 
                        <span style="color:{SOFT_BLUE}; font-weight:900;">{increase_text}</span> increase<br>
                        <span style="color:{SOFT_BLUE}; font-weight:900;">{_fmt_pct(r['monthly_change_pct'])}</span> growth<br>
                        {_tooltip_label('Weighted Sentiment', WEIGHTED_SENTIMENT_TOOLTIP)} 
                        <span style="color:{SOFT_BLUE}; font-weight:900;">{_fmt_score(r['weighted_sentiment'])}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if visible_count < len(emerging_all):
        st.button(
            f"View more ({min(visible_count, len(emerging_all))} of {len(emerging_all)})",
            key="te_view_more_emerging",
            on_click=_show_more_emerging,
        )


# ════════════════════════════════════════════════════════════════
# WORD CLOUD
# ════════════════════════════════════════════════════════════════

def render_wordcloud(df, label, bg_colour="#312421", profile_page_slug="Trend_Explorer"):
    subset = df[df["label"] == label].copy()
    subset["entity"] = subset["entity"].str.lower().str.strip()

    if subset.empty:
        return None

    safe_label = str(label).replace(" ", "_").replace("/", "_").replace("-", "_")

    freq_dict = subset["entity"].value_counts().head(80).to_dict()
    max_freq = max(freq_dict.values())
    total = sum(freq_dict.values())

    words_json = json.dumps([
        {
            "text": word,
            "size": (15 + (freq / max_freq) * 58) * 0.80,
            "freq": freq,
            "pct": round((freq / total) * 100, 2),
        }
        for word, freq in freq_dict.items()
    ])

    colours = [
        "#c2dcff", "#f5f3f5", "#87a7b3", "#e1f1dd",
        "#e1d0b3", "#a18d6d", "#b4cde6", "#cdc7be", "#628e90"
    ]
    colours_json = json.dumps(colours)

    html = f"""
    <div id="wc_{safe_label}" style="
        position:relative;
        width:100%;
        background-color:{bg_colour};
        border-radius:20px;
        padding:18px;
        box-sizing:border-box;
        overflow:hidden;
    ">
        <svg id="svg_{safe_label}" width="100%" height="450"></svg>

        <div id="tooltip_{safe_label}" style="
            display:none;
            position:absolute;
            background-color:#312421;
            color:#c2dcff;
            border:2px solid #c2dcff;
            border-radius:10px;
            padding:8px 14px;
            font-size:14px;
            font-weight:800;
            pointer-events:none;
            z-index:999;
        "></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3-cloud/1.2.5/d3.layout.cloud.min.js"></script>

    <script>
        (function() {{
            const words = {words_json};
            const colours = {colours_json};
            const profileSlug = {json.dumps(profile_page_slug)};

            const svgEl = document.getElementById("svg_{safe_label}");
            const tooltip = document.getElementById("tooltip_{safe_label}");
            const container = document.getElementById("wc_{safe_label}");

            const width = container.clientWidth || 900;
            const height = 450;

            svgEl.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

            const fill = (d, i) => colours[i % colours.length];

            d3.layout.cloud()
                .size([width, height])
                .words(words)
                .padding(7)
                .rotate(() => (Math.random() > 0.78 ? 90 : 0))
                .font("Arial")
                .fontWeight("900")
                .fontSize(d => d.size)
                .on("end", draw)
                .start();

            function draw(words) {{
                d3.select(svgEl)
                    .append("g")
                    .attr("transform", `translate(${{width / 2}},${{height / 2}})`)
                    .selectAll("text")
                    .data(words)
                    .enter()
                    .append("text")
                    .style("font-size", d => `${{d.size}}px`)
                    .style("font-family", "Arial")
                    .style("font-weight", "900")
                    .style("fill", fill)
                    .style("cursor", "pointer")
                    .attr("text-anchor", "middle")
                    .attr("transform", d => `translate(${{d.x}},${{d.y}}) rotate(${{d.rotate}})`)
                    .text(d => d.text)
                    .on("mouseover", function(event, d) {{
                        tooltip.style.display = "block";
                        tooltip.innerHTML = `
                            <div>${{d.text}}</div>
                            <div>Count: ${{d.freq}}</div>
                            <div>Share: ${{d.pct}}%</div>
                        `;
                        d3.select(this).style("opacity", 0.75);
                    }})
                    .on("mousemove", function(event) {{
                        const rect = container.getBoundingClientRect();
                        tooltip.style.left = (event.clientX - rect.left + 12) + "px";
                        tooltip.style.top = (event.clientY - rect.top + 12) + "px";
                    }})
                    .on("mouseout", function() {{
                        tooltip.style.display = "none";
                        d3.select(this).style("opacity", 1);
                    }})

                    .on("click", function(event, d) {{
                        const origin = window.top.location.origin + window.top.location.pathname;
                        const params = new URLSearchParams();
                        params.set("page", "TREND EXPLORER");
                        params.set("trend", d.text);
                        params.set("view", "profile_only");
                        window.open(origin + "?" + params.toString(), "_blank");
                    }});
            }}
        }})();
    </script>
    """

    return html

def _render_trend_spotlight_wordclouds(
    entities_df: pd.DataFrame,
    clean_ent_df: pd.DataFrame | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    category_filter: str = "ALL",
) -> None:

    if entities_df is None or entities_df.empty:
        st.markdown(
            f'<div style="color:{SOFT_BLUE};font-weight:700;text-align:center;margin-top:18px;">No word cloud data available.</div>',
            unsafe_allow_html=True,
        )
        return

    if start_date is not None and end_date is not None:
        date_start = pd.Timestamp(start_date).strftime('%d %b %Y')
        date_end = pd.Timestamp(end_date).strftime('%d %b %Y')
        category_text = str(category_filter).replace("_", " ").title()

        st.markdown(
            f"""
            <div style="width:100%; color:{OFF_WHITE}; font-size:0.8rem; margin:0 0 12px 0; text-align:center;">
                <span style="font-weight:900;">Filters applied:</span>
                <span style="font-weight:650;"> Date range:</span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(date_start)}</span>
                <span style="font-weight:650;"> to </span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(date_end)}</span>
                <span style="font-weight:650;"> · Trend category:</span>
                <span style="font-weight:900; color:{LIGHT_WHITE}; text-decoration: underline; text-underline-offset: 3px;">
                {_safe(category_text)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    filtered_entities_df = _filter_wordcloud_df(entities_df, start_date, end_date)
    filtered_clean_ent_df = _filter_wordcloud_df(clean_ent_df, start_date, end_date)

    if category_filter != "ALL":
        labels_to_render = [category_filter] if category_filter in WORDCLOUD_LABELS else []
    else:
        labels_to_render = WORDCLOUD_LABELS

    if not labels_to_render:
        st.markdown(
            f"""
            <div style="color:{SOFT_BLUE};font-weight:700;text-align:center;margin-top:18px;">
                No word cloud is available for {_safe(str(category_filter).replace("_", " ").title())}.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for label in labels_to_render:
        friendly_title = ENTITY_DISPLAY_NAMES.get(label, label.title())

        st.markdown(
            f"""
            <div style="color:{SOFT_BLUE};font-size:25px;font-weight:800;margin-top:25px;margin-bottom:4px;text-align:center;">
                {_safe(friendly_title)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_to_use = filtered_clean_ent_df if label == "ITEM" and not filtered_clean_ent_df.empty else filtered_entities_df

        html = _get_wordcloud_cached(df_to_use, label)

        if html:
            components.html(html, height=500)
        else:
            st.markdown(
                f"""
                <div style="color:{SOFT_BLUE};font-weight:700;margin-bottom:20px;text-align:center;">
                    No data available for {_safe(friendly_title)}.
                </div>
                """,
                unsafe_allow_html=True,
            )

def _render_trend_spotlight_wordclouds_ex(entities_df: pd.DataFrame, clean_ent_df: pd.DataFrame | None = None) -> None:
    _section_title(
        "Trend Spotlight",
        "Explore the most frequent fashion terms by category using word clouds."
    )

    if entities_df is None or entities_df.empty:
        st.markdown(
            f'<div style="color:{SOFT_BLUE};font-weight:700;text-align:center;margin-top:18px;">No word cloud data available.</div>',
            unsafe_allow_html=True,
        )
        return

    for label in WORDCLOUD_LABELS:
        friendly_title = ENTITY_DISPLAY_NAMES.get(label, label.title())

        st.markdown(
            f"""
            <div style="color:{SOFT_BLUE};font-size:25px;font-weight:800;margin-top:25px;margin-bottom:4px;text-align:center;">
                {_safe(friendly_title)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_to_use = clean_ent_df if label == "ITEM" and clean_ent_df is not None else entities_df

        html = _get_wordcloud_cached(df_to_use, label)

        if html:
            components.html(html, height=500)
        else:
            st.markdown(
                f"""
                <div style="color:{SOFT_BLUE};font-weight:700;margin-bottom:20px;text-align:center;">
                    No data available for {_safe(friendly_title)}.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def render_explorer(
    results: dict,
    predictions: dict | None = None,
    clean_ent_df: pd.DataFrame | None = None,
):

    _inject_css()

    entities_df = _pick(results, "entities_sentiment", "entities")
    phrases_df = _pick(results, "phrases_sentiment", "phrases")
    trend_units_df = _pick(results, "trend_units_sentiment", "trend_units")

    master_df = _build_master(entities_df, phrases_df, trend_units_df)

    if master_df.empty:
        _note(
            "No usable data found. Check that your dataframes contain the expected text and date columns, "
            "then adjust the CONFIG block at the top of trend_explorer.py if your column names differ."
        )
        return
    
    query_params = st.query_params
    is_profile_only = st.session_state.get("deep_link_view") == "profile_only"
    deep_link_trend = st.session_state.get("deep_link_trend")


    full_agg = _aggregate(master_df)

    if is_profile_only and deep_link_trend:
        trend_key = deep_link_trend.strip().lower()

        st.session_state["te_profile_choice"] = trend_key.title()
        st.session_state["te_compare_count"] = 1
        st.session_state.pop("te_profile_choice_2", None)
        st.session_state.pop("te_profile_choice_3", None)

        if trend_key not in full_agg["name"].values:
            st.warning(f"No trend profile found for “{deep_link_trend}”.")
            st.stop()

        with st.container(key="subpage_button_band_trend_profile_only"):
            st.markdown(
                f'<div class="te-static-subpage-header">TREND PROFILE — {_safe(deep_link_trend.upper())}</div>',
                unsafe_allow_html=True,
            )

        with st.container(key="subpage_description_panel"):
            st.markdown(
                f"""
                <div class="subpage-description">
                    A focused look at <b>{_safe(deep_link_trend.title())}</b> — its category, total mentions,
                    average and weighted sentiment, peak month, monthly movement, and related terms, alongside
                    its monthly trend chart.
                </div>
                <div class="description-bottom-band"></div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:26px;"></div>', unsafe_allow_html=True)

        gutter_l, content, gutter_r = st.columns([0.3, 9.4, 0.3])
        with content:
            _render_profile(
                [trend_key],
                full_agg,
                master_df,
                phrases_df,
                trend_units_df,
                predictions,
                False,
            )
        st.stop()



    _page_header()

    with st.container(key="trend_explorer_page_body"):
        start_date, end_date, category_filter, show_forecast = _render_filters(master_df)

        date_filtered = _filter_master(master_df, start_date, end_date, "ALL")
        filtered = _filter_master(master_df, start_date, end_date, category_filter)
        agg = _aggregate(filtered)
        agg_all = _aggregate(date_filtered)

        current_sub_page = st.session_state.get("te_sub_page", "Trend Summary")

        if current_sub_page == "Trend Summary":
            _render_kpis(agg)
            if category_filter == "ALL":
                _render_top_by_category(agg_all)
            _render_emerging(agg, start_date, end_date, category_filter)
        elif current_sub_page == "Trend Comparison":
            view, clicked_trend = _render_leaderboard(agg, start_date, end_date, category_filter)
            selected_trends = _render_profile_selectors(view, clicked_trend)
            _render_profile(
                selected_trends, agg, filtered, phrases_df, trend_units_df, predictions, show_forecast
            )
        elif current_sub_page == "Trend Spotlight":
            _render_trend_spotlight_wordclouds(
                entities_df,
                clean_ent_df,
                start_date,
                end_date,        
                category_filter,
            )
            #_render_trend_spotlight_wordclouds(entities_df, clean_ent_df)
            


# Backward-compatible alias if app.py calls trend_explorer.render(...)
def render(results: dict, predictions: dict | None = None):
    return render_explorer(results, predictions)
