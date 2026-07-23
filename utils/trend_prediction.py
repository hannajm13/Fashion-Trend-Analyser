"""
Trend Prediction page helpers for the Fashion Trend Analyser.

Save this file as: utils/trend_prediction.py

It reads the Excel forecast file `data/all_future_forecasts.xlsx` and combines
those February/March forecasts with the already-loaded historical data used by
Trend Analysis.
"""

from __future__ import annotations

import re
from typing import Optional
from typing import Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative
import threading
from supabase import create_client, Client
_thread_local = threading.local()

try:
    from utils.trend_evolution import (
        ATTRIBUTE_LABELS,
        SENTIMENT_ATTR_LABELS,
        DEFAULT_TOP_N,
        DEFAULT_SENTIMENT_TOP_N,
        SENTIMENT_MIN_FREQ,
        clean_entities_dataframe,
    )
except Exception:
    ATTRIBUTE_LABELS = [
        "ITEM", "COLOR", "MATERIAL", "PATTERN", "STYLE",
        "BRAND", "DETAIL", "PRODUCT",
    ]
    SENTIMENT_ATTR_LABELS = ATTRIBUTE_LABELS
    DEFAULT_TOP_N = 15
    DEFAULT_SENTIMENT_TOP_N = {
        "Phrases": 15,
        "Trend Units": 15,
        "ITEM": 15,
        "COLOR": 15,
        "MATERIAL": 15,
        "PATTERN": 15,
        "STYLE": 15,
        "BRAND": 15,
        "DETAIL": 15,
        "PRODUCT": 15,
    }
    SENTIMENT_MIN_FREQ = 15

    def clean_entities_dataframe(entities_df: pd.DataFrame) -> pd.DataFrame:
        return entities_df.copy()


FORECAST_FILE = "data/all_future_forecasts_6sheets_ex.xlsx"
FORECAST_MONTH_COLUMNS_PATTERN = re.compile(r"^\d{4}-\d{2}$")

# Keep prediction page aligned with Trend Analysis labels. Season is not included
# because your Trend Analysis treats Season as an overlay rather than a main entity chart.
PREDICTION_ATTRIBUTE_LABELS = [
    label for label in ATTRIBUTE_LABELS
    if label in {"ITEM", "COLOR", "MATERIAL", "PATTERN", "STYLE", "BRAND", "DETAIL", "PRODUCT"}
]
if "ITEM" not in PREDICTION_ATTRIBUTE_LABELS:
    PREDICTION_ATTRIBUTE_LABELS = ["ITEM"] + PREDICTION_ATTRIBUTE_LABELS

PREDICTION_SENTIMENT_LABELS = [
    label for label in SENTIMENT_ATTR_LABELS
    if label in {"ITEM", "COLOR", "MATERIAL", "PATTERN", "STYLE", "BRAND", "DETAIL", "PRODUCT"}
]
if "ITEM" not in PREDICTION_SENTIMENT_LABELS:
    PREDICTION_SENTIMENT_LABELS = ["ITEM"] + PREDICTION_SENTIMENT_LABELS


SUPABASE_PAGE_SIZE = 50000

GRAPH_INFO = {
    "Phrases": (
        "Overall Fashion Terms",
        "Raw fashion terms as captured across articles, showing fine-grained trends — e.g. suede jacket.",
    ),
    "Trend Units": (
        "Fashion Terms Used in Combination",
        "Fashion descriptions combining an item with its attributes — e.g. 'wide-leg jeans'.",
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

SUBPAGE_DESCRIPTIONS = {
    "Trend Frequency Forecast": (
        "See historical trend frequency together with statistical-based forecasts for the next 2 (two) months. "
        "Solid lines show actual article mentions, dashed lines show predicted values, and the vertical line marks the end of the actual data period."
    ),
    "Trend Perception Forecast": (
        "See historical frequency * perception scores trend movement together with statistical-based forecasts for the next 2 (two) months. "
        "Solid lines show actual frequency * perception scores, dashed lines show predicted values, and the vertical line marks where forecasting begins."
    ),
}

def get_current_user_id():
    if st.session_state.get("auth_status") == "authenticated":
        return st.session_state.get("user_id")
    return None


def get_supabase() -> Client:
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    return st.session_state.supabase_client

@st.cache_data(show_spinner="Getting the data...")
def get_prediction_data_cached(user_id, _auth_status):
    supabase = get_supabase()
    return load_prediction_data(supabase, user_id=user_id)

def get_prediction_data():
    user_id = get_current_user_id()
    auth_status = st.session_state.get("auth_status", "guest")
    return get_prediction_data_cached(user_id, auth_status)


# ════════════════════════════════════════════════════════════════
# Forecast Excel loading
# ════════════════════════════════════════════════════════════════

def _looks_like_forecast_month(col) -> bool:
    """Return True for Excel forecast month columns such as '2026-02'."""
    if isinstance(col, pd.Timestamp):
        return True
    col_str = str(col).strip()
    if FORECAST_MONTH_COLUMNS_PATTERN.match(col_str):
        return True
    parsed = pd.to_datetime(col_str, errors="coerce")
    return pd.notna(parsed) and col_str[:4].isdigit()


def _column_to_period(col) -> Optional[pd.Timestamp]:
    """Convert a forecast column name such as '2026-02' into a monthly timestamp."""
    if isinstance(col, pd.Timestamp):
        return col.to_period("M").to_timestamp()

    col_str = str(col).strip()
    if FORECAST_MONTH_COLUMNS_PATTERN.match(col_str):
        return pd.to_datetime(f"{col_str}-01", errors="coerce")

    parsed = pd.to_datetime(col_str, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_period("M").to_timestamp()


def _standardise_term(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _normalise_forecast_sheet(
    df: pd.DataFrame,
    term_col: str,
    source_type: str,
    metric_type: str,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert one forecast sheet from wide format:
        term | label | order | 2026-02 | 2026-03
    into long format:
        term | label | source_type | metric_type | period | forecast_value
    """
    output_cols = [
        "term", "label", "source_type", "metric_type",
        "period", "date", "forecast_value",
    ]

    if df is None or df.empty or term_col not in df.columns:
        return pd.DataFrame(columns=output_cols)

    forecast_cols = [col for col in df.columns if _looks_like_forecast_month(col)]
    if not forecast_cols:
        return pd.DataFrame(columns=output_cols)

    work_df = df.copy()
    work_df[term_col] = _standardise_term(work_df[term_col])
    work_df = work_df[work_df[term_col].notna() & (work_df[term_col] != "")]

    rows = []
    for _, row in work_df.iterrows():
        term = row.get(term_col)
        label = row.get(label_col) if label_col and label_col in work_df.columns else None
        label = str(label).strip().upper() if pd.notna(label) else None

        for month_col in forecast_cols:
            period = _column_to_period(month_col)
            if period is None or pd.isna(period):
                continue

            value = pd.to_numeric(row.get(month_col), errors="coerce")
            if pd.isna(value):
                continue

            rows.append({
                "term": term,
                "label": label,
                "source_type": source_type,
                "metric_type": metric_type,
                "period": period,
                "date": period,
                "forecast_value": float(value),
            })

    return pd.DataFrame(rows, columns=output_cols)


@st.cache_data(show_spinner=False)
def load_prediction_data_excel(filepath: str = FORECAST_FILE) -> dict[str, pd.DataFrame]:
    """Load all six forecast sheets from the Auto-ARIMA Excel file."""
    sheets = pd.read_excel(
        filepath,
        sheet_name=[
            "entity_frequency",
            "entity_sentiment",
            "phrase_frequency",
            "phrase_sentiment",
            "trend_frequency",
            "trend_sentiment",
        ],
    )

    return {
        "entity_frequency": _normalise_forecast_sheet(
            sheets.get("entity_frequency"),
            term_col="entity",
            label_col="label",
            source_type="entity",
            metric_type="frequency",
        ),
        "entity_sentiment": _normalise_forecast_sheet(
            sheets.get("entity_sentiment"),
            term_col="entity",
            label_col="label",
            source_type="entity",
            metric_type="sentiment",
        ),
        "phrase_frequency": _normalise_forecast_sheet(
            sheets.get("phrase_frequency"),
            term_col="phrase",
            source_type="phrase",
            metric_type="frequency",
        ),
        "phrase_sentiment": _normalise_forecast_sheet(
            sheets.get("phrase_sentiment"),
            term_col="phrase",
            source_type="phrase",
            metric_type="sentiment",
        ),
        "trend_frequency": _normalise_forecast_sheet(
            sheets.get("trend_frequency"),
            term_col="trend_unit",
            source_type="trend_unit",
            metric_type="frequency",
        ),
        "trend_sentiment": _normalise_forecast_sheet(
            sheets.get("trend_sentiment"),
            term_col="trend_unit",
            source_type="trend_unit",
            metric_type="sentiment",
        ),
    }


def get_thread_client() -> Client:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    return _thread_local.client


def _fetch_predictions_table(client, user_id: str | None = None, page_size: int = SUPABASE_PAGE_SIZE) -> pd.DataFrame:
    """Fetch all visible rows from trend_predictions, paginated, respecting user/base-data rules."""
    thread_client = get_thread_client()

    all_rows: list[dict] = []
    start = 0
    while True:
        end = start + page_size - 1
        query = thread_client.table("trend_predictions").select("*")

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


def _load_prediction_data_from_supabase(client, user_id: str | None = None) -> dict[str, pd.DataFrame]:
    """Load and split forecast data from Supabase into the same 6-key dict shape as the Excel loader."""
    output_cols = ["term", "label", "source_type", "metric_type", "period", "date", "forecast_value"]

    df = _fetch_predictions_table(client, user_id=user_id)

    combos = {
        "entity_frequency": ("entity", "frequency"),
        "entity_sentiment": ("entity", "sentiment"),
        "phrase_frequency": ("phrase", "frequency"),
        "phrase_sentiment": ("phrase", "sentiment"),
        "trend_frequency": ("trend_unit", "frequency"),
        "trend_sentiment": ("trend_unit", "sentiment"),
    }

    if df.empty:
        return {key: pd.DataFrame(columns=output_cols) for key in combos}

    # Normalise term text the same way the Excel path did
    df["term"] = _standardise_term(df["term"])
    df = df[df["term"].notna() & (df["term"] != "")]

    # label: uppercase/stripped, same treatment as Excel path
    df["label"] = df["label"].apply(
        lambda v: str(v).strip().upper() if pd.notna(v) else None
    )

    # forecast_month -> period/date, matching _column_to_period's output type
    df["period"] = df["forecast_month"].apply(_column_to_period)
    df["date"] = df["period"]
    df = df[df["period"].notna()]

    df["forecast_value"] = pd.to_numeric(df["forecast_value"], errors="coerce")
    df = df[df["forecast_value"].notna()]

    result = {}
    for key, (source_type, metric_type) in combos.items():
        subset = df[
            (df["source_type"] == source_type) & (df["metric_type"] == metric_type)
        ][output_cols].reset_index(drop=True)
        result[key] = subset

    return result

def _looks_like_supabase_client(obj: Any) -> bool:
    return hasattr(obj, "table") and callable(getattr(obj, "table"))

def load_prediction_data(source=None, filepath: str = FORECAST_FILE, user_id: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load prediction/forecast data.

    New Supabase usage:
        load_prediction_data(supabase_client, user_id=user_id)

    Legacy Excel fallback:
        load_prediction_data(filepath="path/to/forecast.xlsx")
    """
    if _looks_like_supabase_client(source):
        return _load_prediction_data_from_supabase(source, user_id=user_id)
    load_prediction_data_excel(filepath)
    




# ════════════════════════════════════════════════════════════════
# Actual data preparation
# ════════════════════════════════════════════════════════════════

def _ensure_period(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure period is a clean monthly Timestamp. This prevents Streamlit/Plotly
    from showing dates as midnight timestamp strings.
    """
    work_df = df.copy()

    if date_col in work_df.columns:
        work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
        work_df["period"] = work_df[date_col].dt.to_period("M").dt.to_timestamp()
    elif "period" in work_df.columns:
        work_df["period"] = pd.to_datetime(work_df["period"], errors="coerce")
        work_df["period"] = work_df["period"].dt.to_period("M").dt.to_timestamp()
    else:
        work_df["period"] = pd.NaT

    return work_df[work_df["period"].notna()].copy()


def _actual_frequency_long(
    df: pd.DataFrame,
    text_col: str,
    label: Optional[str] = None,
    label_col: str = "label",
) -> pd.DataFrame:
    output_cols = ["period", "term", "actual_value"]
    if df is None or df.empty or text_col not in df.columns:
        return pd.DataFrame(columns=output_cols)

    plot_df = _ensure_period(df)
    if label is not None and label_col in plot_df.columns:
        plot_df[label_col] = plot_df[label_col].astype(str).str.upper().str.strip()
        plot_df = plot_df[plot_df[label_col] == label]

    plot_df[text_col] = _standardise_term(plot_df[text_col])
    plot_df = plot_df[plot_df[text_col].notna() & (plot_df[text_col] != "")]
    if plot_df.empty:
        return pd.DataFrame(columns=output_cols)

    actual = (
        plot_df.groupby(["period", text_col])
        .size()
        .reset_index(name="actual_value")
        .rename(columns={text_col: "term"})
    )
    return actual[output_cols]


def _actual_sentiment_long(
    df: pd.DataFrame,
    text_col: str,
    top_n: int,
    label: Optional[str] = None,
    label_col: str = "label",
    score_col: str = "roberta_score_signed",
) -> tuple[pd.DataFrame, list[str]]:
    output_cols = ["period", "term", "actual_value"]
    if df is None or df.empty or text_col not in df.columns or score_col not in df.columns:
        return pd.DataFrame(columns=output_cols), []

    plot_df = _ensure_period(df)
    if label is not None and label_col in plot_df.columns:
        plot_df[label_col] = plot_df[label_col].astype(str).str.upper().str.strip()
        plot_df = plot_df[plot_df[label_col] == label]

    plot_df[text_col] = _standardise_term(plot_df[text_col])
    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
    plot_df = plot_df[
        plot_df[text_col].notna()
        & (plot_df[text_col] != "")
        & plot_df[score_col].notna()
    ].copy()
    if plot_df.empty:
        return pd.DataFrame(columns=output_cols), []

    total_counts = plot_df[text_col].value_counts()
    valid_terms = total_counts[total_counts >= SENTIMENT_MIN_FREQ].head(top_n).index.tolist()
    if not valid_terms:
        return pd.DataFrame(columns=output_cols), []

    plot_df = plot_df[plot_df[text_col].isin(valid_terms)]
    grp = plot_df.groupby(["period", text_col])
    agg = grp[score_col].agg(avg_sent="mean", freq="count").reset_index()
    agg["actual_value"] = agg["avg_sent"] * agg["freq"]

    # Match the same sentiment-weighted logic used in Trend Analysis:
    # normalize each item to its own max absolute value, then smooth monthly values.
    agg["actual_value"] = agg.groupby(text_col)["actual_value"].transform(
        lambda x: x / x.abs().max() if x.abs().max() > 0 else x
    )

    pivot = (
        agg.pivot(index="period", columns=text_col, values="actual_value")
        .fillna(0)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    actual = (
        pivot.reset_index()
        .melt(id_vars="period", var_name="term", value_name="actual_value")
    )
    return actual[output_cols], valid_terms


def _filter_forecast(
    forecast_df: pd.DataFrame,
    label: Optional[str] = None,
) -> pd.DataFrame:
    output_cols = ["period", "term", "forecast_value"]
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame(columns=output_cols)

    work_df = forecast_df.copy()
    if label is not None and "label" in work_df.columns:
        work_df["label"] = work_df["label"].astype(str).str.upper().str.strip()
        work_df = work_df[work_df["label"] == label]

    work_df["term"] = _standardise_term(work_df["term"])
    work_df["period"] = pd.to_datetime(work_df["period"], errors="coerce")
    work_df["forecast_value"] = pd.to_numeric(work_df["forecast_value"], errors="coerce")
    work_df = work_df[
        work_df["term"].notna()
        & (work_df["term"] != "")
        & work_df["period"].notna()
        & work_df["forecast_value"].notna()
    ].copy()
    if work_df.empty:
        return pd.DataFrame(columns=output_cols)

    return work_df[output_cols]


def _filter_trigrams(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only three-word trend units where possible."""
    if df is None or df.empty or "term" not in df.columns:
        return df
    work_df = df.copy()
    word_count = work_df["term"].astype(str).str.split().str.len()
    return work_df[word_count == 3].copy()


# ════════════════════════════════════════════════════════════════
# Plotting helpers
# ════════════════════════════════════════════════════════════════

def _select_frequency_terms(actual_long: pd.DataFrame, forecast_long: pd.DataFrame, top_n: int) -> list[str]:
    if actual_long is not None and not actual_long.empty:
        return (
            actual_long.groupby("term")["actual_value"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

    if forecast_long is not None and not forecast_long.empty:
        return (
            forecast_long.groupby("term")["forecast_value"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

    return []


def _append_forecast_anchor(
    actual_term_df: pd.DataFrame,
    forecast_term_df: pd.DataFrame,
) -> pd.DataFrame:
    """Start the dashed forecast line from the last actual point for visual continuity."""
    if forecast_term_df.empty:
        return forecast_term_df
    if actual_term_df.empty:
        return forecast_term_df.sort_values("period")

    last_actual = actual_term_df.sort_values("period").tail(1)
    anchor = pd.DataFrame({
        "period": last_actual["period"].values,
        "term": forecast_term_df["term"].iloc[0],
        "forecast_value": last_actual["actual_value"].values,
    })

    return pd.concat([anchor, forecast_term_df], ignore_index=True).sort_values("period")


def _add_actual_and_forecast_traces(
    fig: go.Figure,
    actual_long: pd.DataFrame,
    forecast_long: pd.DataFrame,
    terms: list[str],
) -> None:
    colours = qualitative.Plotly

    for idx, term in enumerate(terms):
        colour = colours[idx % len(colours)]
        actual_term = actual_long[actual_long["term"] == term].sort_values("period") if not actual_long.empty else pd.DataFrame()
        forecast_term = forecast_long[forecast_long["term"] == term].sort_values("period") if not forecast_long.empty else pd.DataFrame()

        if not actual_term.empty:
            fig.add_trace(go.Scatter(
                x=actual_term["period"],
                y=actual_term["actual_value"],
                mode="lines+markers",
                name=term,
                line=dict(color=colour, dash="solid"),
                marker=dict(color=colour),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Actual: %{y:.3f}"
                    "<extra></extra>"
                ),
            ))

        if not forecast_term.empty:
            forecast_plot = _append_forecast_anchor(actual_term, forecast_term)
            fig.add_trace(go.Scatter(
                x=forecast_plot["period"],
                y=forecast_plot["forecast_value"],
                mode="lines+markers",
                name=f"{term} forecast",
                line=dict(color=colour, dash="dash"),
                marker=dict(color=colour, symbol="diamond"),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Forecast: %{y:.3f}"
                    "<extra></extra>"
                ),
                showlegend=True,
            ))


def _build_forecast_figure(
    actual_long: pd.DataFrame,
    forecast_long: pd.DataFrame,
    terms: list[str],
    title: str,
    yaxis_title: str,
    value_format: str = ".3f",
) -> Optional[go.Figure]:

    if not terms:
        return None

    actual_long = (
        actual_long[actual_long["term"].isin(terms)].copy()
        if actual_long is not None
        else pd.DataFrame()
    )

    forecast_long = (
        forecast_long[forecast_long["term"].isin(terms)].copy()
        if forecast_long is not None
        else pd.DataFrame()
    )

    if actual_long.empty and forecast_long.empty:
        return None

    fig = go.Figure()

    _add_actual_and_forecast_traces(
        fig,
        actual_long,
        forecast_long,
        terms,
    )

    # ── Add forecast-start line ───────────────────────────────────────────────
    if not actual_long.empty:
        last_actual_period = pd.to_datetime(
            actual_long["period"].max()
        )

        fig.add_shape(
            type="line",
            x0=last_actual_period,
            x1=last_actual_period,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color="#312421",
                width=2,
                dash="dash",
            ),
            opacity=0.75,
        )

        fig.add_annotation(
            x=last_actual_period,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast starts",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(
                color="#312421",
                size=12,
            ),
            bgcolor="rgba(245,243,245,0.85)",
        )

    # ── Identify sentiment charts ─────────────────────────────────────────────
    is_sentiment = (
        "sentiment" in yaxis_title.lower()
        or "score" in yaxis_title.lower()
        or "perception" in yaxis_title.lower()
    )

    yaxis_config = {
        "title": {
            "text": yaxis_title
        }
    }

    # ── Dynamic sentiment y-axis ──────────────────────────────────────────────
    if is_sentiment:
        plotted_values = []

        if (
            not actual_long.empty
            and "actual_value" in actual_long.columns
        ):
            actual_values = pd.to_numeric(
                actual_long["actual_value"],
                errors="coerce",
            ).dropna()

            plotted_values.extend(actual_values.tolist())

        if (
            not forecast_long.empty
            and "forecast_value" in forecast_long.columns
        ):
            forecast_values = pd.to_numeric(
                forecast_long["forecast_value"],
                errors="coerce",
            ).dropna()

            plotted_values.extend(forecast_values.tolist())

        if plotted_values:
            minimum_value = min(plotted_values)
            maximum_value = max(plotted_values)

            # Remain at 0–1 when all values are non-negative
            if minimum_value >= 0:
                lower_limit = 0

                # Normally the upper limit remains 1.
                # Expand only if a forecast exceeds 1.
                if maximum_value <= 1:
                    upper_limit = 1
                else:
                    upper_padding = max(
                        0.02,
                        maximum_value * 0.05,
                    )

                    upper_limit = maximum_value + upper_padding

            # Extend below zero only as much as necessary
            else:
                lower_padding = max(
                    0.02,
                    abs(minimum_value) * 0.08,
                )

                lower_limit = minimum_value - lower_padding

                if maximum_value <= 1:
                    upper_limit = 1
                else:
                    upper_padding = max(
                        0.02,
                        maximum_value * 0.05,
                    )

                    upper_limit = maximum_value + upper_padding

            yaxis_config["range"] = [
                lower_limit,
                upper_limit,
            ]

        else:
            yaxis_config["range"] = [0, 1]

        # Zero reference line
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            opacity=0.5,
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        paper_bgcolor="#f5f3f5",
        plot_bgcolor="#f5f3f5",
        height=650,

        # Use the actual argument instead of hard-coded frequency text
        yaxis=yaxis_config,

        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            itemwidth=30,
            tracegroupgap=0,
        ),
        margin=dict(
            l=30,
            r=170,
            t=72,
            b=55,
        ),
        hovermode="closest",
        xaxis=dict(
            tickformat="%b %Y"
        ),
        autosize=True,
    )

    return fig


def create_frequency_prediction_fig(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    text_col: str,
    top_n: int,
    title: str,
    label: Optional[str] = None,
    forecast_is_trigram: bool = False,
) -> Optional[go.Figure]:
    actual_long = _actual_frequency_long(actual_df, text_col=text_col, label=label)
    forecast_long = _filter_forecast(forecast_df, label=label)

    if forecast_is_trigram:
        forecast_long = _filter_trigrams(forecast_long)

    terms = _select_frequency_terms(actual_long, forecast_long, top_n)

    return _build_forecast_figure(
        actual_long=actual_long,
        forecast_long=forecast_long,
        terms=terms,
        title=title,
        yaxis_title="Frequency of Mention in Fashion Articles",
    )


def create_sentiment_prediction_fig(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    text_col: str,
    top_n: int,
    title: str,
    label: Optional[str] = None,
    forecast_is_trigram: bool = False,
) -> Optional[go.Figure]:
    actual_long, valid_terms = _actual_sentiment_long(
        actual_df,
        text_col=text_col,
        top_n=top_n,
        label=label,
    )
    forecast_long = _filter_forecast(forecast_df, label=label)

    if forecast_is_trigram:
        forecast_long = _filter_trigrams(forecast_long)

    if valid_terms:
        terms = valid_terms
    else:
        terms = _select_frequency_terms(
            actual_long=pd.DataFrame(),
            forecast_long=forecast_long.rename(columns={"forecast_value": "actual_value"}) if not forecast_long.empty else forecast_long,
            top_n=top_n,
        )

    return _build_forecast_figure(
        actual_long=actual_long,
        forecast_long=forecast_long,
        terms=terms,
        title=title,
        yaxis_title="Frequency * Associated Sentiment/Perception Score",
    )


# ════════════════════════════════════════════════════════════════
# Streamlit rendering TREND PREDICTION
# ════════════════════════════════════════════════════════════════

def _graph_header(key: str) -> str:
    friendly_title, description = GRAPH_INFO.get(key, (key, ""))
    st.markdown(
        f'<div class="graph-header-title">{friendly_title}</div>'
        f'<div class="graph-header-desc">{description}</div>',
        unsafe_allow_html=True,
    )
    return friendly_title


def _prediction_controls(key: str, default_value: int = DEFAULT_TOP_N, max_value: int = 100) -> int:

    with st.container(key=f"forecast_topn_wide_{key}"):
        top_n = st.number_input(
            "Top number of trends to display",
            min_value=1,
            max_value=max_value,
            value=int(default_value),
            key=key,
        )

    return int(top_n)




def _plot_or_message(fig: Optional[go.Figure], message: str) -> None:
    if fig:
        st.plotly_chart(fig,  width="stretch", config={"responsive": True})
    else:
        st.markdown(
            f'<div style="color:#c2dcff;font-weight:700;margin-bottom:20px;">{message}</div>',
            unsafe_allow_html=True,
        )


def _has_trigram_data(actual_trigram_df: Optional[pd.DataFrame], forecast_df: pd.DataFrame) -> bool:
    if actual_trigram_df is not None and not actual_trigram_df.empty:
        return True
    forecast_long = _filter_forecast(forecast_df)
    return not _filter_trigrams(forecast_long).empty


def render_trend_prediction_page(
    entities_df: pd.DataFrame,
    phrases_df: pd.DataFrame,
    trend_units_df: pd.DataFrame,
    trend_units_from_phrases_df: pd.DataFrame,
    sent_entities_df: pd.DataFrame,
    sent_phrases_df: pd.DataFrame,
    sent_trend_units_df: pd.DataFrame,
    forecast_file: str = FORECAST_FILE,
) -> None:
    """
    Render the full Trend Prediction page.

    Pass in the same historical dataframes already used by Trend Analysis.
    Forecast values are read from `forecast_file`.
    """

    page_slug = "forecast"

    if "prediction_sub_page" not in st.session_state:
        st.session_state.prediction_sub_page = "Trend Frequency Forecast"

    with st.container(key="subpage_button_band_trend_forecast"):
        spacer_l, sub_col1, sub_col2, spacer_r = st.columns([0.5, 1.7, 1.7, 0.5])
        with sub_col1:
            if st.button(
                "Trend Frequency Forecast", key="btn_prediction_frequency", use_container_width=True,
                type="primary" if st.session_state.prediction_sub_page == "Trend Frequency Forecast" else "secondary",
            ):
                st.session_state.prediction_sub_page = "Trend Frequency Forecast"
                st.rerun()
        with sub_col2:
            if st.button(
                "Trend Perception Forecast", key="btn_prediction_sentiment", use_container_width=True,
                type="primary" if st.session_state.prediction_sub_page == "Trend Perception Forecast" else "secondary",
            ):
                st.session_state.prediction_sub_page = "Trend Perception Forecast"
                st.rerun()

    with st.container(key="subpage_description_panel"):
        st.markdown(
            f'<div class="subpage-description">{SUBPAGE_DESCRIPTIONS[st.session_state.prediction_sub_page]}</div>'
            '<div class="description-bottom-band"></div>',
            unsafe_allow_html=True,
        )
    
    try:
        forecasts = get_prediction_data() 
        #forecasts = load_prediction_data_excel(forecast_file)
    except FileNotFoundError:
        st.error(
            f"Prediction file not found: {forecast_file}. "
            "Place all_future_forecasts.xlsx inside your data folder or update the forecast_file path."
        )
        return
    except Exception as exc:
        st.error(f"Could not load prediction data: {exc}")
        return

    if forecasts is None:
        st.error("Prediction data failed to load — forecasts came back empty.")
        return

    # Use the same cleaning treatment as Trend Analysis for ITEM charts.
    clean_ent_df = clean_entities_dataframe(entities_df)
    sent_clean_ent_df = clean_entities_dataframe(sent_entities_df)

    if st.session_state.prediction_sub_page == "Trend Frequency Forecast":
        title = _graph_header("Phrases")
        top_n = _prediction_controls("pred_topn_phrase_frequency", DEFAULT_TOP_N, max_value=100)
        fig = create_frequency_prediction_fig(
            actual_df=phrases_df,
            forecast_df=forecasts["phrase_frequency"],
            text_col="phrase",
            top_n=top_n,
            title=f"{title} — Actual and Forecast",
        )
        _plot_or_message(fig, f"No prediction data available for {title}.")

        title = _graph_header("Trend Units")
        top_n = _prediction_controls("pred_topn_trend_frequency", DEFAULT_TOP_N, max_value=100)
        fig = create_frequency_prediction_fig(
            actual_df=trend_units_df,
            forecast_df=forecasts["trend_frequency"],
            text_col="trend_unit",
            top_n=top_n,
            title=f"{title} — Actual and Forecast",
        )
        _plot_or_message(fig, f"No prediction data available for {title}.")


        for label in PREDICTION_ATTRIBUTE_LABELS:
            title = _graph_header(label)
            top_n = _prediction_controls(f"pred_topn_entity_frequency_{label}", DEFAULT_TOP_N, max_value=100)
            actual_df = clean_ent_df if label == "ITEM" else entities_df
            fig = create_frequency_prediction_fig(
                actual_df=actual_df,
                forecast_df=forecasts["entity_frequency"],
                text_col="entity",
                top_n=top_n,
                title=f"{title} — Actual and Forecast",
                label=label,
            )
            _plot_or_message(fig, f"No prediction data available for {title}.")

    elif st.session_state.prediction_sub_page == "Trend Perception Forecast":
        title = _graph_header("Phrases")
        top_n = _prediction_controls(
            "pred_topn_phrase_sentiment",
            DEFAULT_SENTIMENT_TOP_N.get("Phrases", DEFAULT_TOP_N),
            max_value=100,
        )
        fig = create_sentiment_prediction_fig(
            actual_df=sent_phrases_df,
            forecast_df=forecasts["phrase_sentiment"],
            text_col="phrase",
            top_n=top_n,
            title=f"Frequency * Perception of {title} — Actual and Forecast",
        )
        _plot_or_message(fig, f"No sentiment prediction data available for {title}.")

        title = _graph_header("Trend Units")
        top_n = _prediction_controls(
            "pred_topn_trend_sentiment",
            DEFAULT_SENTIMENT_TOP_N.get("Trend Units", DEFAULT_TOP_N),
            max_value=100,
        )
        fig = create_sentiment_prediction_fig(
            actual_df=sent_trend_units_df,
            forecast_df=forecasts["trend_sentiment"],
            text_col="trend_unit",
            top_n=top_n,
            title=f"Frequency * Perception of {title} — Actual and Forecast",
        )
        _plot_or_message(fig, f"No sentiment prediction data available for {title}.")

    
        for label in PREDICTION_SENTIMENT_LABELS:
            title = _graph_header(label)
            top_n = _prediction_controls(
                f"pred_topn_entity_sentiment_{label}",
                DEFAULT_SENTIMENT_TOP_N.get(label, DEFAULT_TOP_N),
                max_value=100,
            )
            actual_df = sent_clean_ent_df if label == "ITEM" else sent_entities_df
            fig = create_sentiment_prediction_fig(
                actual_df=actual_df,
                forecast_df=forecasts["entity_sentiment"],
                text_col="entity",
                top_n=top_n,
                title=f"Frequency * Perception of {title} — Actual and Forecast",
                label=label,
            )
            _plot_or_message(fig, f"No sentiment prediction data available for {title}.")
