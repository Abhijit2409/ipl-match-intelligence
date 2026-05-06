"""app.py
========================================================================
IPL Match Intelligence — Win Probability & Phase Impact

Streamlit dashboard wrapping the trained 134-feature XGBoost model from
`feature_builder.build_features()`. Lets users filter the historical
innings universe (year, venue, teams, innings, predicted-probability
band) and inspect any single innings end-to-end: phase metrics, context
features, win-probability gauge, distributions, team comparison, model
feature importance, and methodology.

This is a HISTORICAL INNINGS EXPLORER — every prediction is generated
from features the pipeline rebuilt from `matches_*.csv` and
`deliveries_*.csv`. No manual feature entry, no hypothetical match
construction.

Run:
    py -m streamlit run app.py
========================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_builder import build_features, load_assets

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="IPL Match Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Theme — navy / royal blue / gold / electric cyan
# ----------------------------------------------------------------------

PALETTE = {
    "bg_deep":        "#060B17",   # page background
    "bg_panel":       "#0F1B33",   # card / panel base
    "bg_panel_light": "#16243F",   # card hover / accent panels
    "navy":           "#1E3A8A",   # primary navy
    "royal":          "#2563EB",   # royal blue
    "cyan":           "#22D3EE",   # electric cyan accent
    "gold":           "#FBBF24",   # primary gold
    "gold_soft":      "#FDE68A",   # secondary gold
    "text":           "#E2E8F0",   # primary text
    "text_dim":       "#94A3B8",   # secondary text
    "text_muted":     "#64748B",   # tertiary text
    "border":         "rgba(34, 211, 238, 0.18)",  # cyan-tinted borders
    "good":           "#34D399",   # win / above expected
    "warn":           "#FBBF24",   # balanced
    "bad":            "#F87171",   # below expected
}

# Custom CSS — gradient header, glass-effect cards, premium typography
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* App-wide */
.stApp {{
    background: {PALETTE["bg_deep"]};
    font-family: 'Sora', sans-serif;
    color: {PALETTE["text"]};
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #08111F 0%, #0B1A33 100%);
    border-right: 1px solid {PALETTE["border"]};
}}

/* Hero header */
.hero {{
    background:
      radial-gradient(circle at 0% 0%, rgba(37, 99, 235, 0.45) 0%, transparent 55%),
      radial-gradient(circle at 100% 0%, rgba(34, 211, 238, 0.30) 0%, transparent 55%),
      linear-gradient(135deg, #061026 0%, #0F1B33 50%, #0B274C 100%);
    padding: 2.6rem 2.4rem;
    border-radius: 18px;
    border: 1px solid {PALETTE["border"]};
    box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: "";
    position: absolute; right: -120px; top: -120px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(251,191,36,0.10) 0%, transparent 70%);
    pointer-events: none;
}}
.hero h1 {{
    color: {PALETTE["text"]};
    font-size: 2.55rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}}
.hero h1 .accent-gold {{ color: {PALETTE["gold"]}; }}
.hero h1 .accent-cyan {{ color: {PALETTE["cyan"]}; }}
.hero .subtitle {{
    color: {PALETTE["text_dim"]};
    font-size: 1.02rem;
    font-weight: 300;
    line-height: 1.55;
    max-width: 880px;
    margin: 0;
}}
.hero .badges {{
    margin-top: 1.2rem;
    display: flex; gap: 0.6rem; flex-wrap: wrap;
}}
.hero .badge {{
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: rgba(34, 211, 238, 0.08);
    color: {PALETTE["cyan"]};
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(34, 211, 238, 0.30);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}}
.hero .badge.gold {{
    background: rgba(251, 191, 36, 0.10);
    color: {PALETTE["gold"]};
    border-color: rgba(251, 191, 36, 0.35);
}}

/* Section heading */
.section-heading {{
    color: {PALETTE["text"]};
    font-size: 1.08rem;
    font-weight: 700;
    margin: 1.6rem 0 0.7rem 0;
    padding-left: 0.85rem;
    border-left: 3px solid {PALETTE["gold"]};
    letter-spacing: 0.01em;
}}
.section-sub {{
    color: {PALETTE["text_dim"]};
    font-size: 0.92rem;
    margin: -0.4rem 0 1.1rem 0.85rem;
    line-height: 1.5;
}}

/* KPI cards */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin: 0.4rem 0 1.4rem 0;
}}
.kpi {{
    background: linear-gradient(155deg, {PALETTE["bg_panel"]} 0%, {PALETTE["bg_panel_light"]} 100%);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    position: relative;
    transition: transform 0.18s ease, border-color 0.18s ease;
}}
.kpi:hover {{
    transform: translateY(-3px);
    border-color: {PALETTE["border"]};
}}
.kpi .kpi-label {{
    color: {PALETTE["text_muted"]};
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}
.kpi .kpi-value {{
    color: {PALETTE["text"]};
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.05;
    margin-top: 0.3rem;
}}
.kpi .kpi-value.cyan  {{ color: {PALETTE["cyan"]}; }}
.kpi .kpi-value.gold  {{ color: {PALETTE["gold"]}; }}
.kpi .kpi-sub {{
    color: {PALETTE["text_dim"]};
    font-size: 0.78rem;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}}

/* Insight / context cards */
.icard {{
    background: linear-gradient(155deg, {PALETTE["bg_panel"]} 0%, {PALETTE["bg_panel_light"]} 100%);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    height: 100%;
}}
.icard .icard-label {{
    color: {PALETTE["cyan"]};
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}}
.icard .icard-value {{
    color: {PALETTE["text"]};
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 0.25rem;
}}
.icard .icard-note {{
    color: {PALETTE["text_dim"]};
    font-size: 0.82rem;
    margin-top: 0.35rem;
    line-height: 1.45;
}}

/* Confidence pill */
.conf-pill {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid;
}}

/* Methodology / explanation */
.method-box {{
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.18), rgba(34, 211, 238, 0.08));
    border-left: 3px solid {PALETTE["cyan"]};
    padding: 1.1rem 1.4rem;
    border-radius: 0 14px 14px 0;
    margin: 0.7rem 0;
}}
.method-box h4 {{
    color: {PALETTE["gold"]};
    margin: 0 0 0.5rem 0;
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.method-box p, .method-box li {{
    color: {PALETTE["text_dim"]};
    font-size: 0.92rem;
    line-height: 1.65;
    margin: 0.25rem 0;
}}

/* Plotly tweaks */
.js-plotly-plot .plotly text {{ font-family: 'Sora', sans-serif !important; }}

/* Streamlit native widgets */
.stSlider > div > div > div {{ background: {PALETTE["royal"]}; }}
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
    background: rgba(34, 211, 238, 0.18);
    color: {PALETTE["cyan"]};
    border: 1px solid rgba(34, 211, 238, 0.35);
}}

/* Hide Streamlit chrome */
#MainMenu, header, footer {{ visibility: hidden; }}

/* Tighten block padding */
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# File-presence check (fail loudly with the expected file list)
# ----------------------------------------------------------------------
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR

REQUIRED_FILES = [
    BASE_DIR / "src" / "feature_builder.py",
    BASE_DIR / "artifacts" / "ipl_model.pkl",
    BASE_DIR / "artifacts" / "feature_columns.pkl",
    BASE_DIR / "data" / "matches_updated_ipl_upto_2025.csv",
    BASE_DIR / "data" / "deliveries_updated_ipl_upto_2025.csv",
]

missing = [str(f) for f in REQUIRED_FILES if not f.exists()]
if missing:
    st.error(
        "Required files missing from the app directory. Expected each of "
        "the following next to `app.py`:\n\n"
        + "\n".join(f"- `{f}`" for f in REQUIRED_FILES)
        + "\n\n**Not found:** "
        + ", ".join(f"`{f}`" for f in missing)
    )
    st.stop()

# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading XGBoost model and feature schema…")
def _load_model_and_schema():
    """Cache the trained model and its 134-feature column schema."""
    return load_assets()

@st.cache_data(show_spinner="Building 134 features from match + ball-by-ball data…")
def _build_prediction_frame():
    """Cache the full prediction-ready feature matrix and metadata join.

    Returns
    -------
    final_X : aligned feature matrix (n_innings × 134)
    mdl_meta : identifying columns for UI display
    proba   : np.ndarray of model.predict_proba()[:, 1] aligned to both
    """
    model, feature_columns = _load_model_and_schema()
    final_X, mdl_meta = build_features()

    # Validate the 3 invariants the spec calls out
    assert final_X.shape[1] == len(feature_columns), (
        f"Feature count mismatch: final_X has {final_X.shape[1]} cols vs "
        f"feature_columns has {len(feature_columns)}")
    assert list(final_X.columns) == list(feature_columns), \
        "Feature column ORDER does not match feature_columns.pkl"
    assert len(final_X) == len(mdl_meta), \
        "final_X and mdl_meta row counts differ — index alignment broken"

    proba = model.predict_proba(final_X)[:, 1]
    return final_X, mdl_meta, proba

try:
    model, feature_columns = _load_model_and_schema()
    final_X, mdl_meta, proba_all = _build_prediction_frame()
except Exception as exc:
    st.error(f"Feature build / prediction failed:\n\n```\n{exc}\n```")
    st.stop()

# Build a single display dataframe carrying both meta and prediction, with
# a stable row index (`row_id`) we use to look up features in `final_X`.
display_df = mdl_meta.copy()
display_df["predicted_win_probability"] = proba_all
display_df["row_id"] = np.arange(len(display_df))

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{PALETTE['gold']}; font-weight:700; font-size:1.05rem; "
        f"text-transform:uppercase; letter-spacing:0.10em; margin-top:0.5rem;'>"
        f"Filter the universe</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Narrow the 2,292 historical innings.")

    # Defaults exposed for reset
    YEARS_ALL    = sorted(display_df["year"].dropna().astype(int).unique().tolist())
    VENUES_ALL   = sorted(display_df["venue"].dropna().unique().tolist())
    BAT_ALL      = sorted(display_df["batting_team"].dropna().unique().tolist())
    BOWL_ALL     = sorted(display_df["bowling_team"].dropna().unique().tolist())
    INNINGS_ALL  = sorted(display_df["innings"].dropna().astype(int).unique().tolist())

    # Reset handling — store / restore widget state under known keys
    KEYS = ["flt_years", "flt_venues", "flt_bat", "flt_bowl", "flt_inn", "flt_proba"]
    if st.button("↺ Reset filters", use_container_width=True):
        for k in KEYS:
            st.session_state.pop(k, None)
        st.rerun()

    flt_years = st.multiselect("Season (year)", YEARS_ALL,
                               default=st.session_state.get("flt_years", YEARS_ALL),
                               key="flt_years")
    flt_venues = st.multiselect("Venue", VENUES_ALL,
                                default=st.session_state.get("flt_venues", []),
                                key="flt_venues",
                                help="Empty = all venues")
    flt_bat = st.multiselect("Batting team", BAT_ALL,
                             default=st.session_state.get("flt_bat", []),
                             key="flt_bat",
                             help="Empty = all teams")
    flt_bowl = st.multiselect("Bowling team", BOWL_ALL,
                              default=st.session_state.get("flt_bowl", []),
                              key="flt_bowl",
                              help="Empty = all teams")
    flt_inn = st.multiselect("Innings", INNINGS_ALL,
                             default=st.session_state.get("flt_inn", INNINGS_ALL),
                             key="flt_inn")
    flt_proba = st.slider("Predicted win-probability range",
                          0.0, 1.0,
                          value=st.session_state.get("flt_proba", (0.0, 1.0)),
                          step=0.05, key="flt_proba")

    st.markdown("---")
    st.caption(
        f"<span style='color:{PALETTE['text_muted']};font-size:0.78rem;'>"
        "Tip: leave a multiselect empty to disable that filter.</span>",
        unsafe_allow_html=True,
    )

# Apply filters — empty multiselects mean no constraint on that column
mask = (
    display_df["year"].astype(int).isin(flt_years if flt_years else YEARS_ALL)
    & display_df["innings"].astype(int).isin(flt_inn if flt_inn else INNINGS_ALL)
    & display_df["predicted_win_probability"].between(*flt_proba)
)
if flt_venues:
    mask &= display_df["venue"].isin(flt_venues)
if flt_bat:
    mask &= display_df["batting_team"].isin(flt_bat)
if flt_bowl:
    mask &= display_df["bowling_team"].isin(flt_bowl)

filtered = display_df[mask].copy()

# ----------------------------------------------------------------------
# 1. Hero header
# ----------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
        <h1>IPL <span class="accent-cyan">Match</span> Intelligence —
            <span class="accent-gold">Win Probability</span> &amp; Phase Impact</h1>
        <p class="subtitle">
            A 134-feature XGBoost model analyzing phase performance, wickets, chase
            pressure, venue context, team form, head-to-head, toss, par score, and
            Impact Player era. Pick any historical innings and see exactly what the
            model thinks — and why.
        </p>
        <div class="badges">
            <span class="badge">XGBoost · Strategic</span>
            <span class="badge">{len(feature_columns)} Engineered Features</span>
            <span class="badge gold">Hold-out ROC-AUC&nbsp;0.911</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 2. KPI strip (filtered)
# ----------------------------------------------------------------------

if filtered.empty:
    st.warning("No innings match the current filter combination. Widen a filter or hit Reset.")
    st.stop()

avg_p = filtered["predicted_win_probability"].mean()
max_p = filtered["predicted_win_probability"].max()
n_inn = len(filtered)

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-label">Filtered Innings</div>
            <div class="kpi-value cyan">{n_inn:,}</div>
            <div class="kpi-sub">of {len(display_df):,} total</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Avg P(batting wins)</div>
            <div class="kpi-value">{avg_p*100:.1f}%</div>
            <div class="kpi-sub">across selection</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Highest P(win)</div>
            <div class="kpi-value gold">{max_p*100:.1f}%</div>
            <div class="kpi-sub">single-innings peak</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Model Features</div>
            <div class="kpi-value">{len(feature_columns)}</div>
            <div class="kpi-sub">per innings</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 3. Selected innings explorer
# ----------------------------------------------------------------------

st.markdown('<div class="section-heading">Innings explorer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Pick any innings from the filtered set. The gauge '
    'shows the model\'s probability that the batting team wins this match.</div>',
    unsafe_allow_html=True,
)

# Readable selectbox label per row
def _row_label(r):
    return (f"Match {int(r['match_id'])}  ·  {int(r['year'])}  ·  Inn {int(r['innings'])}  "
            f"·  {r['batting_team']}  vs  {r['bowling_team']}  ·  {r['venue'][:40]}")

filtered["__label__"] = filtered.apply(_row_label, axis=1)
selected_label = st.selectbox(
    "Select innings",
    filtered["__label__"].tolist(),
    key="innings_selector",
    label_visibility="collapsed",
)
selected_row    = filtered[filtered["__label__"] == selected_label].iloc[0]
selected_index  = int(selected_row["row_id"])    # row index into final_X

# Re-predict for the selected innings using the exact column order, as a
# DataFrame (NOT NumPy) — keeps XGBoost happy with feature names.
selected_X    = final_X.loc[[selected_index], feature_columns]
selected_prob = float(model.predict_proba(selected_X)[0, 1])
selected_pct  = selected_prob * 100

# Confidence label
def _confidence_band(p):
    pct = p * 100
    if pct < 40:    return "Low control",    PALETTE["bad"]
    if pct < 55:    return "Balanced",       PALETTE["warn"]
    if pct < 70:    return "Advantage",      PALETTE["cyan"]
    return                  "Strong control", PALETTE["good"]

conf_label, conf_color = _confidence_band(selected_prob)

col_meta, col_gauge = st.columns([1.05, 1.0])

with col_meta:
    # Match identity card grid
    sr = selected_row
    target_actual = "Yes" if sr.get("target", 0) == 1 else "No"
    st.markdown(
        f"""
        <div class="icard" style="margin-bottom:0.7rem;">
            <div class="icard-label">Match · Innings</div>
            <div class="icard-value">#{int(sr['match_id'])} · Innings {int(sr['innings'])}</div>
            <div class="icard-note">Season {int(sr['year'])}
              &nbsp;|&nbsp; Date {pd.to_datetime(sr['date']).date()
                                  if pd.notna(sr['date']) else 'n/a'}</div>
        </div>
        <div class="icard" style="margin-bottom:0.7rem;">
            <div class="icard-label">Matchup</div>
            <div class="icard-value">{sr['batting_team']}
              <span style="color:{PALETTE['text_muted']};font-weight:400">  vs  </span>
              {sr['bowling_team']}</div>
            <div class="icard-note">Toss won by {sr.get('toss_winner','—')} &nbsp;·&nbsp;
              chose to {sr.get('toss_decision','—')}</div>
        </div>
        <div class="icard">
            <div class="icard-label">Venue</div>
            <div class="icard-value" style="font-size:1.1rem;">{sr['venue']}</div>
            <div class="icard-note">Outcome on record: batting team
              <strong style="color:{PALETTE['gold']}">won? {target_actual}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_gauge:
    # Plotly gauge — navy/cyan/gold theme
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=selected_pct,
        number={"suffix": "%", "font": {"size": 56,
                                          "color": conf_color, "family": "Sora"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PALETTE["text_muted"],
                      "tickwidth": 1, "tickfont": {"color": PALETTE["text_muted"]}},
            "bar": {"color": conf_color, "thickness": 0.30},
            "bgcolor": PALETTE["bg_panel"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": "rgba(248, 113, 113, 0.15)"},
                {"range": [40, 55],  "color": "rgba(251, 191, 36, 0.15)"},
                {"range": [55, 70],  "color": "rgba(34, 211, 238, 0.18)"},
                {"range": [70, 100], "color": "rgba(52, 211, 153, 0.20)"},
            ],
            "threshold": {"line": {"color": PALETTE["gold"], "width": 3},
                           "thickness": 0.85, "value": 50},
        },
    ))
    gauge.update_layout(
        paper_bgcolor=PALETTE["bg_deep"],
        font={"color": PALETTE["text_dim"], "family": "Sora"},
        height=320,
        margin=dict(l=20, r=20, t=10, b=10),
    )
    st.plotly_chart(gauge, use_container_width=True,
                     config={"displayModeBar": False})
    st.markdown(
        f"<div style='text-align:center; margin-top:-0.5rem;'>"
        f"<span class='conf-pill' style='color:{conf_color}; "
        f"border-color:{conf_color}; background:rgba(255,255,255,0.02)'>"
        f"{conf_label}  ·  P(batting wins) = {selected_pct:.1f}%</span></div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# 4. Phase summary — pull from selected_X (one row, exact feature columns)
# ----------------------------------------------------------------------

st.markdown('<div class="section-heading">Phase performance — this innings</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Powerplay (overs 1–6), Middle (7–15), Death (16–20). '
    'Run rate is runs per over within the phase.</div>',
    unsafe_allow_html=True,
)

PHASE_FIELDS = [
    ("Powerplay runs",     "powerplay_runs",     "{:.0f}",   PALETTE["cyan"]),
    ("Middle runs",        "middle_runs",        "{:.0f}",   PALETTE["cyan"]),
    ("Death runs",         "death_runs",         "{:.0f}",   PALETTE["gold"]),
    ("Powerplay wickets",  "powerplay_wickets",  "{:.0f}",   PALETTE["text"]),
    ("Middle wickets",     "middle_wickets",     "{:.0f}",   PALETTE["text"]),
    ("Death wickets",      "death_wickets",      "{:.0f}",   PALETTE["text"]),
    ("Powerplay RR",       "powerplay_rr",       "{:.2f}",   PALETTE["cyan"]),
    ("Middle RR",          "middle_rr",          "{:.2f}",   PALETTE["cyan"]),
    ("Death RR",           "death_rr",           "{:.2f}",   PALETTE["gold"]),
]

phase_cols = st.columns(3)
for i, (label, col, fmt, color) in enumerate(PHASE_FIELDS):
    target_col = phase_cols[i % 3]
    val_str = "—"
    if col in selected_X.columns:
        try:
            val_str = fmt.format(float(selected_X.iloc[0][col]))
        except Exception:
            val_str = "—"
    target_col.markdown(
        f"""
        <div class="icard" style="margin-bottom:0.7rem;">
            <div class="icard-label">{label}</div>
            <div class="icard-value" style="color:{color}">{val_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# 5. Context insight cards — plain-English readings of strategic features
# ----------------------------------------------------------------------

st.markdown('<div class="section-heading">Strategic context</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">How the model reads chase pressure, venue par, '
    'head-to-head form, toss, and Impact Player era for this row.</div>',
    unsafe_allow_html=True,
)

def _safe(col, default=None):
    """Return the selected row's value for a feature, or `default` if missing."""
    return float(selected_X.iloc[0][col]) if col in selected_X.columns else default

cp_pp     = _safe("chase_pressure_pp")
cp_15     = _safe("chase_pressure_15")
v_par     = _safe("venue_par_total")
score_par = _safe("score_vs_venue_par")
h2h       = _safe("bat_h2h_winrate")
won_toss  = _safe("batting_won_toss")
era_after = _safe("impact_era_after")
innings_n = int(selected_row["innings"])

context_cards = []

# Chase pressure (only meaningful for innings 2)
if innings_n == 2 and cp_15 is not None:
    if cp_15 < 0:
        cp_note = "Chase ahead of pace — required RR is below current RR."
    elif cp_15 < 2:
        cp_note = "Chase tracking par — required RR roughly matches the pace so far."
    elif cp_15 < 5:
        cp_note = "Pressure building — required RR is climbing above the achieved pace."
    else:
        cp_note = "Heavy late pressure — required RR is well above what the chase has managed."
    context_cards.append((
        "Chase pressure (after over 15)",
        f"{cp_15:+.2f} rpo",
        cp_note,
        PALETTE["gold"] if cp_15 >= 5 else (PALETTE["cyan"] if cp_15 >= 0 else PALETTE["good"]),
    ))
elif innings_n == 2 and cp_pp is not None:
    context_cards.append((
        "Chase pressure (after powerplay)",
        f"{cp_pp:+.2f} rpo",
        "Required RR vs current RR after the powerplay.",
        PALETTE["cyan"],
    ))
else:
    context_cards.append((
        "Chase pressure",
        "Not applicable",
        "First-innings batting — no target exists yet, so chase features default to 0.",
        PALETTE["text_muted"],
    ))

# Venue par + score-vs-par
if v_par is not None:
    if score_par is not None:
        if score_par >=  20:  par_note = "Innings well above this venue's historical par."
        elif score_par >=  5: par_note = "Innings slightly above venue par."
        elif score_par > -5:  par_note = "Innings landed near venue par."
        elif score_par > -20: par_note = "Innings slightly below venue par."
        else:                 par_note = "Innings well below venue par."
    else:
        par_note = "Training-only venue benchmark."
    context_cards.append((
        "Venue par (training-era mean)",
        f"{v_par:.0f} runs",
        par_note + (f"  Δ = {score_par:+.0f}" if score_par is not None else ""),
        PALETTE["gold"],
    ))

# Head-to-head form
if h2h is not None:
    if h2h >= 0.60:   h2h_note = "Strong historical edge for the batting team."
    elif h2h >= 0.55: h2h_note = "Slight historical edge for the batting team."
    elif h2h >= 0.45: h2h_note = "Roughly even history between these two."
    else:             h2h_note = "Bowling team historically dominates this matchup."
    context_cards.append((
        "Head-to-head winrate",
        f"{h2h*100:.1f}%",
        h2h_note + " Computed on matches strictly before this one.",
        PALETTE["cyan"],
    ))

# Toss
if won_toss is not None:
    toss_text = "Yes" if won_toss >= 0.5 else "No"
    toss_note = ("Batting team won the toss at this match."
                 if won_toss >= 0.5 else
                 "Batting team did not win the toss at this match.")
    context_cards.append((
        "Won the toss?",
        toss_text,
        toss_note,
        PALETTE["good"] if won_toss >= 0.5 else PALETTE["text_dim"],
    ))

# Impact Player era
if era_after is not None:
    era_text = "After 2023" if era_after >= 0.5 else "Before 2023"
    era_note = (
        "Impact Player rule active — extra batter, deeper death-overs scoring patterns."
        if era_after >= 0.5 else
        "Pre-rule era — no Impact Player substitution available.")
    context_cards.append((
        "Impact Player era",
        era_text,
        era_note,
        PALETTE["gold"] if era_after >= 0.5 else PALETTE["text_dim"],
    ))

# Render in a 3-up grid (with row wrap)
ctx_cols = st.columns(3)
for i, (lbl, val, note, color) in enumerate(context_cards):
    ctx_cols[i % 3].markdown(
        f"""
        <div class="icard" style="margin-bottom:0.8rem; border-left:3px solid {color};">
            <div class="icard-label">{lbl}</div>
            <div class="icard-value" style="color:{color}">{val}</div>
            <div class="icard-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# 6. Prediction distribution + 7. Team comparison (side-by-side)
# ----------------------------------------------------------------------

st.markdown('<div class="section-heading">Population view</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Where the selected innings sits across the filtered '
    'universe of innings, and how predictions average out by batting team.</div>',
    unsafe_allow_html=True,
)

dist_col, team_col = st.columns([1, 1])

with dist_col:
    hist = px.histogram(
        filtered,
        x="predicted_win_probability",
        nbins=30,
        title="Distribution of predicted win probabilities",
    )
    hist.update_traces(marker_color=PALETTE["royal"],
                        marker_line_color=PALETTE["cyan"], marker_line_width=0.6,
                        opacity=0.9,
                        hovertemplate="P(win) %{x:.2f}<br>Innings %{y}<extra></extra>")
    hist.add_vline(x=selected_prob, line_color=PALETTE["gold"], line_width=2,
                    annotation_text="this innings",
                    annotation_position="top",
                    annotation_font_color=PALETTE["gold"])
    hist.update_layout(
        plot_bgcolor=PALETTE["bg_deep"], paper_bgcolor=PALETTE["bg_deep"],
        font={"color": PALETTE["text_dim"], "family": "Sora"},
        title_font_color=PALETTE["text"],
        xaxis=dict(title="P(batting team wins)",
                    color=PALETTE["text_dim"], gridcolor="rgba(148, 163, 184, 0.10)"),
        yaxis=dict(title="Innings count",
                    color=PALETTE["text_dim"], gridcolor="rgba(148, 163, 184, 0.10)"),
        height=380, margin=dict(l=40, r=20, t=50, b=40),
        bargap=0.05,
    )
    st.plotly_chart(hist, use_container_width=True,
                     config={"displayModeBar": False})

with team_col:
    team_avg = (filtered.groupby("batting_team")["predicted_win_probability"]
                         .agg(["mean", "size"])
                         .rename(columns={"mean": "avg_p", "size": "n"})
                         .reset_index()
                         .sort_values("avg_p", ascending=True))
    bar = go.Figure(go.Bar(
        x=team_avg["avg_p"] * 100,
        y=team_avg["batting_team"],
        orientation="h",
        marker=dict(
            color=team_avg["avg_p"],
            colorscale=[[0, PALETTE["bad"]], [0.5, PALETTE["warn"]],
                         [0.75, PALETTE["cyan"]], [1.0, PALETTE["gold"]]],
            showscale=False,
            line=dict(width=0),
        ),
        text=[f"{v*100:.1f}% (n={n})" for v, n in zip(team_avg["avg_p"], team_avg["n"])],
        textposition="outside",
        textfont={"color": PALETTE["text_dim"], "family": "JetBrains Mono", "size": 10},
        hovertemplate="<b>%{y}</b><br>Avg P(win) %{x:.1f}%<extra></extra>",
    ))
    bar.update_layout(
        title="Average predicted P(win) by batting team",
        title_font_color=PALETTE["text"],
        plot_bgcolor=PALETTE["bg_deep"], paper_bgcolor=PALETTE["bg_deep"],
        font={"color": PALETTE["text_dim"], "family": "Sora"},
        xaxis=dict(title="Avg P(batting wins) %", range=[0, 105],
                    color=PALETTE["text_dim"], gridcolor="rgba(148, 163, 184, 0.10)"),
        yaxis=dict(title="", color=PALETTE["text_dim"]),
        height=380, margin=dict(l=20, r=70, t=50, b=40),
    )
    st.plotly_chart(bar, use_container_width=True,
                     config={"displayModeBar": False})

# ----------------------------------------------------------------------
# 8. Feature importance
# ----------------------------------------------------------------------

st.markdown('<div class="section-heading">What the model leans on</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Top 10 XGBoost feature importances. Importance reflects '
    "how often a feature splits the model's trees and improves the loss — it's a "
    'signal of <strong>model influence, not direct causation</strong>.</div>',
    unsafe_allow_html=True,
)

fi_df = (pd.DataFrame({"feature": feature_columns,
                        "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(10)
            .iloc[::-1])

fi_fig = go.Figure(go.Bar(
    x=fi_df["importance"], y=fi_df["feature"],
    orientation="h",
    marker=dict(
        color=fi_df["importance"],
        colorscale=[[0, PALETTE["royal"]], [0.5, PALETTE["cyan"]], [1.0, PALETTE["gold"]]],
        showscale=False,
    ),
    text=[f"{v:.3f}" for v in fi_df["importance"]],
    textposition="outside",
    textfont={"color": PALETTE["text_dim"], "family": "JetBrains Mono"},
    hovertemplate="<b>%{y}</b><br>importance %{x:.3f}<extra></extra>",
))
fi_fig.update_layout(
    plot_bgcolor=PALETTE["bg_deep"], paper_bgcolor=PALETTE["bg_deep"],
    font={"color": PALETTE["text_dim"], "family": "Sora"},
    xaxis=dict(title="Importance",
                color=PALETTE["text_dim"], gridcolor="rgba(148, 163, 184, 0.10)"),
    yaxis=dict(title="", color=PALETTE["text_dim"]),
    height=440, margin=dict(l=30, r=80, t=10, b=40),
)
st.plotly_chart(fi_fig, use_container_width=True,
                 config={"displayModeBar": False})

# ----------------------------------------------------------------------
# 9. Data table
# ----------------------------------------------------------------------

st.markdown('<div class="section-heading">Filtered innings table</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="section-sub">{n_inn:,} rows. Sortable; predictions match the gauge above.</div>',
    unsafe_allow_html=True,
)
table_df = (filtered[["match_id", "year", "innings", "batting_team",
                       "bowling_team", "venue", "predicted_win_probability"]]
                .sort_values("predicted_win_probability", ascending=False)
                .reset_index(drop=True))
# Use Streamlit column_config (works without jinja2, formats client-side)
st.dataframe(
    table_df,
    use_container_width=True,
    height=420,
    hide_index=True,
    column_config={
        "match_id":     st.column_config.NumberColumn("Match",   format="%d"),
        "year":         st.column_config.NumberColumn("Year",    format="%d"),
        "innings":      st.column_config.NumberColumn("Inn",     format="%d"),
        "batting_team": st.column_config.TextColumn("Batting team"),
        "bowling_team": st.column_config.TextColumn("Bowling team"),
        "venue":        st.column_config.TextColumn("Venue"),
        "predicted_win_probability": st.column_config.ProgressColumn(
            "P(batting team wins)",
            format="%.1f%%",
            min_value=0.0, max_value=1.0,
        ),
    },
)

# ----------------------------------------------------------------------
# 10. Methodology
# ----------------------------------------------------------------------

st.markdown('<div class="section-heading">How this app works</div>',
            unsafe_allow_html=True)
METHOD_HTML = """
<div class="method-box">
    <h4>Pipeline</h4>
    <p>The app rebuilds the full 134-feature vector for every historical innings
        directly from the matches and ball-by-ball CSVs &mdash; phase metrics, run rates,
        wickets in hand, cumulative resources, venue par scores, chase pressure,
        rolling team form, head-to-head winrate, toss flags, and Impact Player era
        indicators. The same XGBoost classifier saved to <code>ipl_model.pkl</code>
        scores each row.</p>
</div>
<div class="method-box">
    <h4>What the prediction means</h4>
    <p>The model outputs the probability that the <strong>batting team in this
        innings wins the match</strong>. For first-innings rows that is the team
        posting the total; for second-innings rows that is the team chasing.</p>
</div>
<div class="method-box">
    <h4>Validation strategy</h4>
    <p>Training used a chronological split with seasons through 2023 in the training
        window and 2024&ndash;2025 held out. Impact Player era is included as a feature
        (<code>impact_era_after</code>), so the model sees both pre- and post-rule
        innings during training and explicitly accounts for the regime shift &mdash;
        it is <em>not</em> trained only on the pre-rule period. Hold-out
        <strong>ROC-AUC&nbsp;=&nbsp;0.911</strong>.</p>
</div>
<div class="method-box">
    <h4>Scope of this view</h4>
    <p>This is a <strong>historical innings explorer</strong>. Every prediction
        comes from features built from real historical match and ball-by-ball
        records. Manual construction of a hypothetical future-match feature
        vector would require rebuilding all 134 features from a synthetic match
        state &mdash; that is a different surface, deliberately out of scope here.</p>
</div>
"""
st.markdown(METHOD_HTML, unsafe_allow_html=True)

st.markdown(
    f"<div style='text-align:center; color:{PALETTE['text_muted']}; "
    f"font-size:0.78rem; margin-top:1.6rem; padding-top:1rem; "
    f"border-top:1px solid rgba(148,163,184,0.08);'>"
    f"IPL Match Intelligence · {len(feature_columns)}-feature XGBoost · "
    f"chronological hold-out ROC-AUC 0.911"
    f"</div>",
    unsafe_allow_html=True,
)
