import json
from itertools import combinations
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
from supabase import create_client, Client
import streamlit as st
import re

_thread_local = threading.local()


# Full set of 8 attribute-entity labels — used consistently across
# the frequency time series page AND the sentiment time series page.
# SEASON is intentionally excluded here: it is plotted separately via the
# optional Season overlay toggle on each chart.
ATTRIBUTE_LABELS = [
    "ITEM", "COLOR", "MATERIAL", "PATTERN", "STYLE",
     "DETAIL","BRAND", "PRODUCT"
]

WORDCLOUD_LABELS = [
    "ITEM", "COLOR", "MATERIAL", "PATTERN", "STYLE",
     "DETAIL","BRAND", "PRODUCT"
]

SENTIMENT_ATTR_LABELS = ATTRIBUTE_LABELS

SEASON_WORDS = ["winter", "spring", "summer", "fall"]
SEASON_COLOURS = {
    "winter": "#5b8fb9",
    "spring": "#6fbf73",
    "summer": "#f2a900",
    "fall": "#c1440e",
}

GRAPH_INFO = {
    "Phrases": (
        "Overall Fashion Terms",
        "Raw fashion terms as captured across articles, showing fine-grained trends — e.g. suede jacket.",
    ),
    "Trend Units": (
        "Fashion Terms Used in Combination",
        "Fashion descriptions combining an item with its attributes — e.g. 'wide-leg jeans'.",
    ),
    "Trigram Trend Units": (
        "Three-Word Fashion Term Combinations",
        "Three-word fashion combinations combining an item with its attributes — e.g. 'dark wash wide-leg jeans'.",
    ),
    "ITEM": (
        "Clothing & Accessories",
        "The most frequently mentioned clothing items, footwear, and accessories.",
    ),
    "COLOR": (
        "Colours",
        "Colours and shades most commonly referenced across fashion articles.",
    ),
    "MATERIAL": (
        "Materials & Fabrics",
        "Fabrics and materials most talked about across fashion articles.",
    ),
    "PATTERN": (
        "Prints & Patterns",
        "Surface designs and prints appearing most in fashion articles.",
    ),
    "STYLE": (
        "Style Aesthetics",
        "Broad fashion aesthetics and movements trending across fashion articles.",
    ),
    "BRAND": (
        "Brands",
        "Fashion brands most frequently mentioned in fashion articles.",
    ),
    "DETAIL": (
        "Design Details",
        "Garment details and construction features most discussed.",
    ),
    "SEASON": (
        "Seasons & Collections",
        "Seasonal references and collection periods mentioned most.",
    ),
    "PRODUCT": (
        "Signature Products",
        "Named or signature products appearing the most in fashion coverage.",
    ),
}

DEFAULT_TOP_N = 15

SUBPAGE_DESCRIPTIONS = {
    "Trend Frequency Over Time": (
        "See how often each fashion trend was mentioned across articles over time. "
        "Each chart shows the top trends for that category — hover over any point to see the exact count for that month, "
        "or use the input above a chart to change how many trends are shown and turn on the season toggle to see "
        "what is trending across seasons."
    ),
    "Trend Spotlight": (
        "The larger a word appears, the more frequently it was mentioned across fashion articles. "
        "Hover over any word to see its exact count and share of mentions and click on a word to open up "
        "its trend profile and view the trend in more detail."
    ),
    "Trend Perception Over Time": (
        "See whether each trend was covered positively or negatively over time. Each line combines how often a trend "
        "was mentioned with how positive or negative that coverage was — the higher the score, the more positively "
        "the trend was discussed in fashion articles while a score of zero indicates neutrality. "
        "Hover over any point for the exact score, or use the input "
        "below a chart to change how many trends are shown and turn on the season toggle to see "
        "what is trending across seasons."
    ),
}

DISCARD_ENTITIES = {
    "accessory", "accessories", "dress", "dresses",
    "bag", "bags", "pants", "shoe", "'", '"', "-"
}

SENTIMENT_MIN_FREQ = 10

DEFAULT_SENTIMENT_TOP_N = {
    "Phrases": 10,
    "Trend Units": 10,
    "Trigram Trend Units": 10,
    "ITEM": 10,
    "COLOR": 10,
    "MATERIAL": 10,
    "PATTERN": 10,
    "STYLE": 10,
    "BRAND": 10,
    "DETAIL": 10,
    "SEASON": 10,
    "PRODUCT": 10,
}

SUPABASE_PAGE_SIZE = 25000


# ════════════════════════════════════════════════════════════════
# GENERAL HELPERS
# ════════════════════════════════════════════════════════════════

def _empty_df(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or [])


def _looks_like_supabase_client(obj: Any) -> bool:
    return hasattr(obj, "table") and callable(getattr(obj, "table"))


def _ensure_datetime_period(
    df: pd.DataFrame,
    date_col: str = "date",
    period_col: str = "period",
    time_freq: str = "M",
) -> pd.DataFrame:
    """
    Supabase returns date columns as ISO strings. This helper normalises date and
    period columns before charting so Plotly receives real monthly Timestamps.
    """
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else _empty_df()

    out = df.copy()

    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if period_col in out.columns:
        out[period_col] = pd.to_datetime(out[period_col], errors="coerce")

    if date_col in out.columns:
        # Always recompute monthly period from date when possible. This prevents
        # string dates from showing as 00:00:00 or becoming NaT during grouping.
        valid_date_mask = out[date_col].notna()
        out.loc[valid_date_mask, period_col] = (
            out.loc[valid_date_mask, date_col]
            .dt.to_period(time_freq)
            .dt.to_timestamp()
        )

    return out


def _normalise_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise text/date columns after reading from Supabase or Excel."""
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else _empty_df()

    out = df.copy()

    if "label" in out.columns:
        out["label"] = out["label"].astype(str).str.upper().str.strip()

    for col in ["entity", "phrase", "trend_unit", "labels", "ngram_type"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
            out.loc[out[col].isin(["nan", "None", "NaT"]), col] = ""

    for col in ["roberta_score_signed", "roberta_score", "sentiment"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return _ensure_datetime_period(out)


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


def render_wordcloud_ex(df, label, bg_colour="#312421"):
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
            "size": (15 + (freq / max_freq) * 58) * 0.75,
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
                    }});
            }}
        }})();
    </script>
    """

    return html

# ════════════════════════════════════════════════════════════════
# SEASON OVERLAY
# ════════════════════════════════════════════════════════════════

def _season_pivot(entities_df, time_freq="M"):
    if entities_df is None or entities_df.empty:
        return None

    required_cols = {"label", "entity", "date"}
    if not required_cols.issubset(entities_df.columns):
        return None

    season_df = _ensure_datetime_period(entities_df, time_freq=time_freq)
    season_df = season_df[season_df["label"].astype(str).str.upper() == "SEASON"].copy()
    if season_df.empty:
        return None

    season_df["entity"] = season_df["entity"].astype(str).str.lower().str.strip()
    season_df = season_df[season_df["entity"].isin(SEASON_WORDS)]
    season_df = season_df[season_df["period"].notna()]
    if season_df.empty:
        return None

    counts = season_df.groupby(["period", "entity"]).size().reset_index(name="frequency")
    pivot = counts.pivot(index="period", columns="entity", values="frequency").fillna(0)
    return pivot


def add_season_overlay(fig, entities_df, time_freq="M"):
    pivot = _season_pivot(entities_df, time_freq=time_freq)

    if pivot is None:
        return fig

    for season in SEASON_WORDS:
        if season not in pivot.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[season],
                mode="lines+markers",
                name=f"Season: {season.title()}",
                line=dict(
                    dash="dot",
                    width=2.5,
                    color=SEASON_COLOURS[season],
                ),
                marker=dict(
                    size=5,
                    color=SEASON_COLOURS[season],
                ),
                hovertemplate=(
                    f"<b>Season: {season.title()}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Mentions: %{y:.0f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    # Keep the secondary scale, but hide its axis and title
    fig.update_yaxes(
        title_text=None,
        showticklabels=False,
        showline=False,
        ticks="",
        showgrid=False,
        zeroline=False,
        secondary_y=True,
    )

    return fig

# ════════════════════════════════════════════════════════════════
# TIME-SERIES CHARTS
# ════════════════════════════════════════════════════════════════

def create_time_series_chart(df, text_col, top_n, title, time_freq="M", season_entities_df=None):
    if df is None or df.empty or text_col not in df.columns:
        return None

    plot_df = _ensure_datetime_period(df, time_freq=time_freq)
    plot_df = plot_df[plot_df[text_col].notna() & (plot_df[text_col].astype(str).str.strip() != "")]
    plot_df = plot_df[plot_df["period"].notna()]
    if plot_df.empty:
        return None

    total_counts = plot_df[text_col].value_counts()
    valid_items = total_counts.head(int(top_n)).index
    plot_df = plot_df[plot_df[text_col].isin(valid_items)]
    if plot_df.empty:
        return None

    time_series = plot_df.groupby(["period", text_col]).size().reset_index(name="frequency")
    pivot_df = time_series.pivot(index="period", columns=text_col, values="frequency").fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for column in pivot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df[column],
                mode="lines+markers",
                name=column,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Frequency: %{y:.0f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if season_entities_df is not None:
        fig = add_season_overlay(fig, season_entities_df, time_freq=time_freq)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
        paper_bgcolor="#f5f3f5",
        plot_bgcolor="#f5f3f5",
        height=500,
        yaxis_title="Frequency of Mention in Fashion Articles",
        legend=dict(
            orientation="v",
            x=0.98,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            itemwidth=30,
            ),
        margin=dict(l=30, r=130, t=60, b=55),
        hovermode="closest",
        xaxis=dict(tickformat="%b %Y"),
        autosize=True,
    )
    
    return fig


def create_trend_units_time_series(df, ngram_type, top_n, title, time_freq="M", season_entities_df=None):
    if df is None or df.empty or "trend_unit" not in df.columns or "ngram_type" not in df.columns:
        return None

    filtered_df = _ensure_datetime_period(df, time_freq=time_freq)
    filtered_df = filtered_df[filtered_df["ngram_type"] == ngram_type].copy()
    filtered_df = filtered_df[filtered_df["trend_unit"].notna() & (filtered_df["trend_unit"].astype(str).str.strip() != "")]
    filtered_df = filtered_df[filtered_df["period"].notna()]
    if filtered_df.empty:
        return None

    total_counts = filtered_df["trend_unit"].value_counts()
    valid_units = total_counts.head(int(top_n)).index
    filtered_df = filtered_df[filtered_df["trend_unit"].isin(valid_units)]
    if filtered_df.empty:
        return None

    time_series = filtered_df.groupby(["period", "trend_unit"]).size().reset_index(name="frequency")
    pivot_df = time_series.pivot(index="period", columns="trend_unit", values="frequency").fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for column in pivot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df[column],
                mode="lines+markers",
                name=column,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Frequency: %{y:.0f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if season_entities_df is not None:
        fig = add_season_overlay(fig, season_entities_df, time_freq=time_freq)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
        paper_bgcolor="#f5f3f5",
        plot_bgcolor="#f5f3f5",
        height=500,
        yaxis_title="Frequency of Mention in Fashion Articles",
        legend=dict(
            orientation="v",
            x=0.96,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            itemwidth=30,
            ),
        margin=dict(l=30, r=130, t=60, b=55),
        hovermode="closest",
        xaxis=dict(tickformat="%b %Y"),
        autosize=True,
    )
    return fig


def create_attribute_time_series(entities_df, label, top_n, title=None, time_freq="M", season_entities_df=None):
    if entities_df is None or entities_df.empty or not {"label", "entity"}.issubset(entities_df.columns):
        return None

    filtered_df = _ensure_datetime_period(entities_df, time_freq=time_freq)
    filtered_df = filtered_df[filtered_df["label"].astype(str).str.upper() == label].copy()
    if filtered_df.empty:
        return None

    filtered_df["entity"] = filtered_df["entity"].astype(str).str.lower().str.strip()
    filtered_df = filtered_df[filtered_df["entity"] != ""]
    filtered_df = filtered_df[filtered_df["period"].notna()]
    if filtered_df.empty:
        return None

    total_counts = filtered_df["entity"].value_counts()
    valid_entities = total_counts.head(int(top_n)).index
    filtered_df = filtered_df[filtered_df["entity"].isin(valid_entities)]
    if filtered_df.empty:
        return None

    time_series = filtered_df.groupby(["period", "entity"]).size().reset_index(name="frequency")
    pivot_df = time_series.pivot(index="period", columns="entity", values="frequency").fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for column in pivot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df[column],
                mode="lines+markers",
                name=column,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Frequency: %{y:.0f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if season_entities_df is not None:
        fig = add_season_overlay(fig, season_entities_df, time_freq=time_freq)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
        paper_bgcolor="#f5f3f5",
        plot_bgcolor="#f5f3f5",
        height=500,
        yaxis_title="Frequency of Mention in Fashion Articles",
        legend=dict(
            orientation="v",
            x=0.98,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            itemwidth=30,
            ),
        margin=dict(l=30, r=130, t=72, b=55),
        hovermode="closest",
        xaxis=dict(tickformat="%b %Y"),
        autosize=True,
    )
    return fig


# ════════════════════════════════════════════════════════════════
# SENTIMENT-WEIGHTED CHARTS
# ════════════════════════════════════════════════════════════════

def create_weighted_sentiment_fig(
    df,
    text_col,
    top_n,
    title,
    score_col="roberta_score_signed",
    date_col="date",
    time_freq="M",
    season_entities_df=None,
):
    required_cols = {text_col, score_col, date_col}
    if df is None or df.empty or not required_cols.issubset(df.columns):
        return None

    plot_df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")].copy()
    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
    plot_df = plot_df[plot_df[score_col].notna()]
    if plot_df.empty:
        return None

    plot_df = _ensure_datetime_period(plot_df, date_col=date_col, time_freq=time_freq)
    plot_df = plot_df[plot_df[date_col].notna() & plot_df["period"].notna()]
    if plot_df.empty:
        return None

    total_counts = plot_df[text_col].value_counts()
    valid_items = total_counts[total_counts >= SENTIMENT_MIN_FREQ].head(int(top_n)).index
    plot_df = plot_df[plot_df[text_col].isin(valid_items)]
    if plot_df.empty:
        return None

    grp = plot_df.groupby(["period", text_col])
    agg = grp[score_col].agg(avg_sent="mean", freq="count").reset_index()
    agg["weighted"] = agg["avg_sent"] * agg["freq"]

    agg["weighted"] = agg.groupby(text_col)["weighted"].transform(
        lambda x: x / x.abs().max() if x.abs().max() > 0 else x
    )

    pivot = (
        agg.pivot(index="period", columns=text_col, values="weighted")
        .fillna(0)
        #.rolling(window=3, min_periods=1)
        #.mean()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for column in pivot.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[column],
                mode="lines+markers",
                name=column,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Score: %{y:.3f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if season_entities_df is not None:
        fig = add_season_overlay(fig, season_entities_df, time_freq=time_freq)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
        paper_bgcolor="#f5f3f5",
        plot_bgcolor="#f5f3f5",
        height=500,
        yaxis_title="Frequency * Associated Sentiment/Perception Score",
        legend=dict(
            orientation="v",
            x=0.98,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            itemwidth=30,
            ),
        margin=dict(l=30, r=130, t=72, b=55),
        hovermode="closest",
        xaxis=dict(tickformat="%b %Y"),
        autosize=True,
    )
    return fig


# ════════════════════════════════════════════════════════════════
# TREND UNIT BUILDING
# ════════════════════════════════════════════════════════════════

def extract_trend_units_from_phrase_row(row):
    phrase = row.get("phrase", "")
    labels_raw = row.get("labels", "")
    units = []

    if pd.isna(phrase) or pd.isna(labels_raw):
        return units

    words = str(phrase).lower().split()
    labels = [label.strip() for label in str(labels_raw).split(" + ")]

    if len(words) != len(labels):
        return units

    item_positions = [i for i, label in enumerate(labels) if label == "ITEM"]
    carry_columns = [
        "date",
        "period",
        "roberta_score_signed",
        "roberta_label",
        "roberta_score",
        "sentiment",
        "sentiment_label",
        "is_base_data",
        "user_id",
        "uploaded_at",
        "import_job_id",
    ]

    def build_unit(trend_unit, ngram_type):
        unit = {
            "trend_unit": trend_unit,
            "ngram_type": ngram_type,
        }
        for col in carry_columns:
            if col in row.index:
                unit[col] = row[col]
        return unit

    for item_pos in item_positions:
        item = words[item_pos]

        for i, label in enumerate(labels):
            if i != item_pos:
                units.append(build_unit(f"{words[i]} {item}", "bigram"))

        attribute_positions = [i for i in range(len(words)) if i != item_pos]
        for attr1, attr2 in combinations(attribute_positions, 2):
            units.append(build_unit(f"{words[attr1]} {words[attr2]} {item}", "trigram"))

    return units


def build_trend_units_from_phrases(phrases_df):
    columns = [
        "date",
        "period",
        "trend_unit",
        "ngram_type",
        "roberta_score_signed",
        "roberta_label",
        "roberta_score",
        "sentiment",
        "sentiment_label",
        "is_base_data",
        "user_id",
        "uploaded_at",
        "import_job_id",
    ]

    if phrases_df is None or phrases_df.empty or "phrase" not in phrases_df.columns or "labels" not in phrases_df.columns:
        return pd.DataFrame(columns=columns)

    trend_rows = []
    for _, row in phrases_df.iterrows():
        trend_rows.extend(extract_trend_units_from_phrase_row(row))

    if not trend_rows:
        return pd.DataFrame(columns=columns)

    trend_units_df = pd.DataFrame(trend_rows)
    trend_units_df = _normalise_basic_columns(trend_units_df)
    return trend_units_df


# ════════════════════════════════════════════════════════════════
# SUPABASE DATA LOADING
# ════════════════════════════════════════════════════════════════



def get_thread_client() -> Client:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    return _thread_local.client



def _query_table(client, table: str, user_id: str | None = None, page_size: int = SUPABASE_PAGE_SIZE) -> pd.DataFrame:
    thread_client = get_thread_client()   # <-- own client per thread, ignore passed-in client

    all_rows: list[dict] = []
    start = 0
    while True:
        end = start + page_size - 1
        query = thread_client.table(table).select("*")
        if user_id:
            query = query.or_(f"is_base_data.eq.true,user_id.eq.{user_id}")
        else:
            query = query.eq("is_base_data", True)
        response = query.range(start, end).execute()
        batch = response.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return pd.DataFrame(all_rows)


""" def _load_trend_data_from_supabase(client, user_id: str | None = None):
    tables = ["entities", "phrases", "trend_units",
              "entities_sentiment", "phrases_sentiment", "trend_units_sentiment"]

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = dict(zip(tables, executor.map(
            lambda t: _normalise_basic_columns(_query_table(client, t, user_id)), tables
        )))

    entities_df = results["entities"]
    phrases_df = results["phrases"]
    trend_units_df = results["trend_units"]
    sent_entities_df = results["entities_sentiment"]
    sent_phrases_df = results["phrases_sentiment"]
    sent_trend_units_df = results["trend_units_sentiment"]

    trend_units_from_phrases_df = build_trend_units_from_phrases(phrases_df)
    sent_trend_units_from_phrases_df = build_trend_units_from_phrases(sent_phrases_df)

    return (entities_df, phrases_df, trend_units_df, trend_units_from_phrases_df, 
            sent_entities_df, sent_phrases_df, sent_trend_units_df, sent_trend_units_from_phrases_df) """

def _load_trend_data_from_supabase(client, user_id: str | None = None):
    tables = ["entities", "phrases", "trend_units"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = dict(zip(tables, executor.map(
            lambda t: _normalise_basic_columns(_query_table(client, t, user_id)), tables
        )))

    entities_df = results["entities"]
    phrases_df = results["phrases"]
    trend_units_df = results["trend_units"]

    # sentiment dataframes reference the same underlying data,
    # since roberta_score_signed lives in the same table
    sent_entities_df = entities_df.copy()
    sent_phrases_df = phrases_df.copy()
    sent_trend_units_df = trend_units_df.copy()

    trend_units_from_phrases_df = build_trend_units_from_phrases(phrases_df)
    sent_trend_units_from_phrases_df = build_trend_units_from_phrases(sent_phrases_df)

    return (entities_df, phrases_df, trend_units_df, trend_units_from_phrases_df,
            sent_entities_df, sent_phrases_df, sent_trend_units_df, sent_trend_units_from_phrases_df)



# ════════════════════════════════════════════════════════════════
# LEGACY EXCEL LOADING FALLBACK
# ════════════════════════════════════════════════════════════════

def _load_trend_data_from_excel(filepath, sentiment_filepath):
    entities_df = pd.read_excel(filepath, sheet_name="entities")
    phrases_df = pd.read_excel(filepath, sheet_name="phrases")
    trend_units_df = pd.read_excel(filepath, sheet_name="trend_units")

    entities_df = _normalise_basic_columns(entities_df)
    phrases_df = _normalise_basic_columns(phrases_df)
    trend_units_df = _normalise_basic_columns(trend_units_df)

    trend_units_from_phrases_df = build_trend_units_from_phrases(phrases_df)

    sent_entities_df = pd.read_excel(sentiment_filepath, sheet_name="Entities")
    sent_phrases_df = pd.read_excel(sentiment_filepath, sheet_name="Phrases")
    sent_trend_units_df = pd.read_excel(sentiment_filepath, sheet_name="Trend Units")

    sent_entities_df = _normalise_basic_columns(sent_entities_df)
    sent_phrases_df = _normalise_basic_columns(sent_phrases_df)
    sent_trend_units_df = _normalise_basic_columns(sent_trend_units_df)

    sent_trend_units_from_phrases_df = build_trend_units_from_phrases(sent_phrases_df)

    return (
        entities_df,
        phrases_df,
        trend_units_df,
        trend_units_from_phrases_df,
        sent_entities_df,
        sent_phrases_df,
        sent_trend_units_df,
        sent_trend_units_from_phrases_df,
    )


def load_trend_data(source, user_id=None, sentiment_filepath=None):
    """
    Load trend data.

    New Supabase usage:
        load_trend_data(supabase_client, user_id=None)

    Temporary Excel fallback, useful while migrating:
        load_trend_data("data/entities_final.xlsx", "data/entities_with_sentiment_final.xlsx")
    """
    if _looks_like_supabase_client(source):
        return _load_trend_data_from_supabase(source, user_id=user_id)

    # Backward compatibility with the old call style:
    # load_trend_data(filepath, sentiment_filepath)
    filepath = source
    if sentiment_filepath is None:
        sentiment_filepath = user_id

    if sentiment_filepath is None:
        raise ValueError(
            "For Excel loading, provide both filepath and sentiment_filepath. "
            "For Supabase loading, call load_trend_data(supabase_client, user_id)."
        )

    return _load_trend_data_from_excel(filepath, sentiment_filepath)


# ════════════════════════════════════════════════════════════════
# USER DATA DELETE HELPER
# ════════════════════════════════════════════════════════════════

def delete_user_data(client, user_id: str):
    tables = [
        "entities",
        "phrases",
        "trend_units",
        "entities_sentiment",
        "phrases_sentiment",
        "trend_units_sentiment",
        "trend_predictions",
    ]

    for table in tables:
        (
            client.table(table)
            .delete()
            .eq("user_id", user_id)
            .eq("is_base_data", False)
            .execute()
        )


# ════════════════════════════════════════════════════════════════
# CLEANING
# ════════════════════════════════════════════════════════════════

def clean_entities_dataframe(entities_df):
    if entities_df is None or entities_df.empty or "entity" not in entities_df.columns:
        return entities_df.copy() if isinstance(entities_df, pd.DataFrame) else _empty_df()

    clean_df = entities_df.copy()
    clean_df["entity_clean"] = clean_df["entity"].astype(str).str.lower().str.strip()
    clean_df = clean_df[~clean_df["entity_clean"].isin(DISCARD_ENTITIES)]
    clean_df = clean_df[~clean_df["entity_clean"].str.fullmatch(r"[\W_]+", na=False)]
    clean_df = clean_df[clean_df["entity_clean"] != ""]
    clean_df["entity"] = clean_df["entity_clean"]
    clean_df = clean_df.drop(columns=["entity_clean"])
    return clean_df
