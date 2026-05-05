"""feature_builder.py
========================================================================
Reproduces the exact feature engineering used during model training and
returns a prediction-ready feature matrix aligned to feature_columns.pkl.

Designed to be imported by a Streamlit app (or any inference script):

    from feature_builder import build_features, load_assets

    model, feature_columns = load_assets()
    final_X, mdl_meta = build_features()
    proba = model.predict_proba(final_X)[:, 1]

Run this file directly (`python feature_builder.py`) to validate the
pipeline end-to-end before wiring it into a UI.

Key guarantees
--------------
- Same leakage-safe baselines as training (training-only venue / par /
  toss-bias means computed from rows with year <= TRAIN_END).
- Final feature matrix has *exactly* the columns listed in
  feature_columns.pkl, in the same order.
- Missing one-hot dummy columns are added as 0.
- Extra columns produced by feature engineering are dropped.
- Row count = number of innings in the historical data; identifying
  metadata is returned in `mdl_meta` for UI display.
========================================================================
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Configuration — must match train_model.py / train_model.ipynb
# --------------------------------------------------------------------

# Default file locations (override via build_features arguments if needed)
DELIVERIES_PATH    = "deliveries_updated_ipl_upto_2025.csv"
MATCHES_PATH       = "matches_updated_ipl_upto_2025.csv"
FEATURE_COLS_PATH  = "feature_columns.pkl"
MODEL_PATH         = "ipl_model.pkl"

# Chronological cut-off used at training time (inclusive)
TRAIN_END = 2023

# T20 phase definitions
PP_OVERS, M_OVERS, D_OVERS = 6, 9, 5
PHASE_BINS   = [-1, 5, 14, 19]
PHASE_LABELS = ["Powerplay", "Middle", "Death"]

# Cricket dismissals counted as a wicket against the batting team
WICKET_KINDS = {
    "caught", "bowled", "run out", "lbw", "stumped",
    "caught and bowled", "hit wicket", "obstructing the field",
    "retired hurt", "retired out",
}

# Neutral defaults when a team has no prior history
NEUTRAL_TEAM_STATS = {
    "form_to_date":         0.5,
    "form_last_10":         0.5,
    "avg_runs_scored":      160.0,
    "avg_runs_conceded":    160.0,
    "recent_runs_scored":   160.0,
    "recent_runs_conceded": 160.0,
}
TEAM_COLS = list(NEUTRAL_TEAM_STATS.keys())

# Column-name aliases for varying IPL CSV dump conventions
DELIVERY_ALIASES = {
    "match_id":         ["match_id", "matchId", "id"],
    "innings":          ["innings", "inning"],
    "over":             ["over"],
    "ball":             ["ball"],
    "batting_team":     ["batting_team"],
    "bowling_team":     ["bowling_team"],
    "runs_off_bat":     ["runs_off_bat", "batsman_runs", "batter_runs"],
    "extras":           ["extras", "extra_runs"],
    "wides":            ["wides", "isWide", "wide_runs"],
    "noballs":          ["noballs", "isNoBall", "noball_runs"],
    "byes":             ["byes", "Byes", "bye_runs"],
    "legbyes":          ["legbyes", "LegByes", "legbye_runs"],
    "penalty":          ["penalty", "Penalty", "penalty_runs"],
    "wicket_type":      ["wicket_type", "dismissal_kind"],
    "player_dismissed": ["player_dismissed"],
    "date":             ["date", "start_date"],
}
MATCH_ALIASES = {
    "match_id":      ["match_id", "matchId", "id"],
    "date":          ["date", "start_date"],
    "venue":         ["venue"],
    "winner":        ["winner"],
    "team1":         ["team1"],
    "team2":         ["team2"],
    "toss_winner":   ["toss_winner"],
    "toss_decision": ["toss_decision"],
}

# Scalar features whose names start with "venue_" but are NOT one-hot dummies
VENUE_SCALAR_FEATURES = {"venue_par_total", "venue_year_par",
                          "venue_bat_first_winrate"}

# Identifying / metadata columns retained in `mdl_meta` (not features)
META_COLS = ["match_id", "innings", "batting_team", "bowling_team",
             "venue", "year", "date", "toss_winner", "toss_decision",
             "target"]


# --------------------------------------------------------------------
# 1. Loading + cleaning
# --------------------------------------------------------------------

def _standardize(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Rename any alias columns to their canonical name."""
    rename = {}
    for canonical, options in aliases.items():
        for opt in options:
            if opt in df.columns and opt != canonical:
                rename[opt] = canonical
                break
    return df.rename(columns=rename)


def _load_and_clean(deliveries_path: str, matches_path: str
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs, normalize column names, coerce types, drop super-overs
    and abandoned matches."""
    deliveries = _standardize(pd.read_csv(deliveries_path), DELIVERY_ALIASES)
    matches    = _standardize(pd.read_csv(matches_path),    MATCH_ALIASES)

    # Coerce numeric extras (some dumps store flags as blanks)
    for col in ("runs_off_bat", "extras", "wides", "noballs",
                "byes", "legbyes", "penalty"):
        if col in deliveries.columns:
            deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce").fillna(0)
        else:
            deliveries[col] = 0

    # Drop super-overs and abandoned matches
    deliveries = deliveries[deliveries["innings"].isin([1, 2])].copy()
    deliveries["date"] = pd.to_datetime(deliveries["date"], errors="coerce")
    matches["date"]    = pd.to_datetime(matches["date"],    errors="coerce")
    matches            = matches[matches["winner"].notna()].copy()
    return deliveries, matches


# --------------------------------------------------------------------
# 2. Innings-level base table
# --------------------------------------------------------------------

def _build_innings_table(deliveries: pd.DataFrame,
                         matches: pd.DataFrame
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot ball-by-ball to one row per (match, innings) with phase-level
    runs/wickets and innings metadata. Returns (mdl, df) where `df` is the
    enriched ball-level frame used by team-strength feature builders."""
    # Phase classification
    deliveries["phase"] = pd.Categorical(
        pd.cut(deliveries["over"].astype(int),
               bins=PHASE_BINS, labels=PHASE_LABELS),
        categories=PHASE_LABELS, ordered=True,
    )
    deliveries["total_runs"] = deliveries["runs_off_bat"] + deliveries["extras"]
    deliveries["is_wicket"]  = (deliveries["wicket_type"].astype(str)
                                                          .str.lower()
                                                          .isin(WICKET_KINDS))

    # Merge winner + derive outcome / era
    df = deliveries.merge(matches[["match_id", "winner"]], on="match_id", how="inner")
    df["batting_team_won_match"] = df["batting_team"] == df["winner"]
    df["year"]              = df["date"].dt.year
    df["impact_era_after"]  = (df["year"] >= 2023).astype(int)

    # Phase pivots — one column per phase
    runs = df.pivot_table(index=["match_id", "innings"], columns="phase",
                          values="total_runs", aggfunc="sum",
                          observed=True).fillna(0)
    runs.columns = [f"{str(c).lower()}_runs" for c in runs.columns]

    wkts = df.pivot_table(index=["match_id", "innings"], columns="phase",
                          values="is_wicket", aggfunc="sum",
                          observed=True).fillna(0).astype(int)
    wkts.columns = [f"{str(c).lower()}_wickets" for c in wkts.columns]

    # Innings metadata
    meta = (df.groupby(["match_id", "innings"])
              .agg(batting_team=("batting_team", "first"),
                   bowling_team=("bowling_team", "first"),
                   year=("year", "first"),
                   impact_era_after=("impact_era_after", "first"),
                   batting_team_won_match=("batting_team_won_match", "first"))
              .reset_index())

    mdl = (runs.join(wkts).reset_index()
                .merge(meta, on=["match_id", "innings"])
                .merge(matches[["match_id", "venue", "date",
                                "toss_winner", "toss_decision"]],
                       on="match_id", how="left"))
    mdl["batting_first_or_second"] = mdl["innings"]
    mdl["target"] = mdl["batting_team_won_match"].astype(int)
    return mdl, df


# --------------------------------------------------------------------
# 3. Phase features (run rates, cumulatives, RPW, collapses, momentum)
# --------------------------------------------------------------------

def _add_advanced_phase_features(mdl: pd.DataFrame) -> pd.DataFrame:
    """Phase run rates, cumulatives, runs-per-wicket, collapse flags,
    acceleration deltas, boundary-pressure proxies."""
    # Tempo: phase run rates
    mdl["powerplay_rr"] = mdl["powerplay_runs"] / PP_OVERS
    mdl["middle_rr"]    = mdl["middle_runs"]    / M_OVERS
    mdl["death_rr"]     = mdl["death_runs"]     / D_OVERS

    # Resource: cumulative runs
    mdl["cum_runs_after_pp"] = mdl["powerplay_runs"]
    mdl["cum_runs_after_15"] = mdl["powerplay_runs"] + mdl["middle_runs"]
    mdl["cum_runs_total"]    = mdl["cum_runs_after_15"] + mdl["death_runs"]

    # Resource: cumulative wickets / wickets remaining
    mdl["wkts_after_pp"]      = mdl["powerplay_wickets"]
    mdl["wkts_after_15"]      = mdl["powerplay_wickets"] + mdl["middle_wickets"]
    mdl["wkts_total"]         = mdl["wkts_after_15"] + mdl["death_wickets"]
    mdl["wkts_rem_after_pp"]  = (10 - mdl["wkts_after_pp"]).clip(lower=0)
    mdl["wkts_rem_after_15"]  = (10 - mdl["wkts_after_15"]).clip(lower=0)

    # Stress: runs per wicket lost (denominator capped at 1)
    mdl["pp_rpw"]     = mdl["powerplay_runs"] / mdl["powerplay_wickets"].clip(lower=1)
    mdl["middle_rpw"] = mdl["middle_runs"]    / mdl["middle_wickets"].clip(lower=1)
    mdl["death_rpw"]  = mdl["death_runs"]     / mdl["death_wickets"].clip(lower=1)

    # Stress: collapse flags (3+ wickets in a single phase)
    mdl["pp_collapse"]     = (mdl["powerplay_wickets"] >= 3).astype(int)
    mdl["middle_collapse"] = (mdl["middle_wickets"]    >= 3).astype(int)
    mdl["death_collapse"]  = (mdl["death_wickets"]     >= 3).astype(int)

    # Tempo: acceleration deltas between phases
    mdl["accel_pp_to_middle"]    = mdl["middle_rr"] - mdl["powerplay_rr"]
    mdl["accel_middle_to_death"] = mdl["death_rr"]  - mdl["middle_rr"]

    # Innings shape
    mdl["death_share_of_innings"] = mdl["death_runs"] / mdl["cum_runs_total"].clip(lower=1)

    # Boundary-pressure proxies (excess scoring above a baseline RR)
    mdl["pp_aggression"]     = (mdl["powerplay_runs"] - 6 * PP_OVERS).clip(lower=0)
    mdl["middle_aggression"] = (mdl["middle_runs"]    - 7 * M_OVERS).clip(lower=0)
    mdl["death_aggression"]  = (mdl["death_runs"]     - 9 * D_OVERS).clip(lower=0)
    return mdl


# --------------------------------------------------------------------
# 4. Venue baselines (TRAINING-ONLY)
# --------------------------------------------------------------------

def _add_venue_baselines(mdl: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Training-only venue means + per-row deltas. Test-era / unseen venues
    fall back to the global training mean."""
    train_only = mdl.loc[train_mask]
    v_pp = train_only.groupby("venue")["powerplay_runs"].mean()
    v_m  = train_only.groupby("venue")["middle_runs"].mean()
    v_d  = train_only.groupby("venue")["death_runs"].mean()
    g_pp = train_only["powerplay_runs"].mean()
    g_m  = train_only["middle_runs"].mean()
    g_d  = train_only["death_runs"].mean()

    mdl["v_pp_avg"]     = mdl["venue"].map(v_pp).fillna(g_pp)
    mdl["v_middle_avg"] = mdl["venue"].map(v_m).fillna(g_m)
    mdl["v_death_avg"]  = mdl["venue"].map(v_d).fillna(g_d)
    mdl["v_pp_diff"]     = mdl["powerplay_runs"] - mdl["v_pp_avg"]
    mdl["v_middle_diff"] = mdl["middle_runs"]    - mdl["v_middle_avg"]
    mdl["v_death_diff"]  = mdl["death_runs"]     - mdl["v_death_avg"]
    return mdl


# --------------------------------------------------------------------
# 5. Team strength + head-to-head
# --------------------------------------------------------------------

def _add_team_strength_features(mdl: pd.DataFrame, df: pd.DataFrame,
                                matches: pd.DataFrame) -> pd.DataFrame:
    """Rolling team form + per-team batting/bowling profile + head-to-head.
    Leakage-safe via groupby(team).shift(1).expanding/rolling — every stat
    reflects only matches strictly before the current one."""
    # Long-form (match × team) frame
    ml = pd.concat([
        matches[["match_id", "date", "team1", "winner"]].rename(columns={"team1": "team"}),
        matches[["match_id", "date", "team2", "winner"]].rename(columns={"team2": "team"}),
    ], ignore_index=True)
    ml["won"] = (ml["team"] == ml["winner"]).astype(int)

    # Per-(match, team) runs scored / conceded
    inn_pair = (df.groupby(["match_id", "batting_team", "bowling_team"])["total_runs"]
                  .sum().reset_index())
    runs_scored = (inn_pair.groupby(["match_id", "batting_team"])["total_runs"]
                            .sum().reset_index()
                            .rename(columns={"batting_team": "team",
                                             "total_runs":   "runs_scored"}))
    runs_conceded = (inn_pair.groupby(["match_id", "bowling_team"])["total_runs"]
                              .sum().reset_index()
                              .rename(columns={"bowling_team": "team",
                                               "total_runs":   "runs_conceded"}))
    ml = (ml.merge(runs_scored,   on=["match_id", "team"], how="left")
            .merge(runs_conceded, on=["match_id", "team"], how="left"))
    ml = ml.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

    # Expanding / rolling team metrics (shift(1) so current match is excluded)
    ml["form_to_date"]         = ml.groupby("team")["won"].transform(
        lambda s: s.shift(1).expanding().mean())
    ml["form_last_10"]         = ml.groupby("team")["won"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    ml["avg_runs_scored"]      = ml.groupby("team")["runs_scored"].transform(
        lambda s: s.shift(1).expanding().mean())
    ml["avg_runs_conceded"]    = ml.groupby("team")["runs_conceded"].transform(
        lambda s: s.shift(1).expanding().mean())
    ml["recent_runs_scored"]   = ml.groupby("team")["runs_scored"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    ml["recent_runs_conceded"] = ml.groupby("team")["runs_conceded"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean())

    # Fill leading NaNs (first appearances) with neutral defaults
    for col, val in NEUTRAL_TEAM_STATS.items():
        ml[col] = ml[col].fillna(val)

    # Attach as batting-side and bowling-side variants
    bat_feats  = ml[["match_id", "team"] + TEAM_COLS].rename(
        columns={"team": "batting_team", **{c: f"bat_{c}"  for c in TEAM_COLS}})
    bowl_feats = ml[["match_id", "team"] + TEAM_COLS].rename(
        columns={"team": "bowling_team", **{c: f"bowl_{c}" for c in TEAM_COLS}})
    mdl = mdl.merge(bat_feats,  on=["match_id", "batting_team"],  how="left")
    mdl = mdl.merge(bowl_feats, on=["match_id", "bowling_team"], how="left")

    # Head-to-head — canonicalized pair, expanding win rate
    mh = matches.copy().sort_values(["date", "match_id"]).reset_index(drop=True)
    mh["pair_a"] = np.where(mh["team1"] < mh["team2"], mh["team1"], mh["team2"])
    mh["pair_b"] = np.where(mh["team1"] < mh["team2"], mh["team2"], mh["team1"])
    mh["pair_a_won"] = (mh["winner"] == mh["pair_a"]).astype(int)
    mh = mh.sort_values(["pair_a", "pair_b", "date", "match_id"]).reset_index(drop=True)
    mh["pair_a_h2h_winrate"] = (mh.groupby(["pair_a", "pair_b"])["pair_a_won"]
                                  .transform(lambda s: s.shift(1).expanding().mean()))
    mh["pair_a_h2h_winrate"] = mh["pair_a_h2h_winrate"].fillna(0.5)
    mh["team1_h2h_winrate"] = np.where(mh["pair_a"] == mh["team1"],
                                        mh["pair_a_h2h_winrate"],
                                        1 - mh["pair_a_h2h_winrate"])
    mdl = mdl.merge(mh[["match_id", "team1", "team2", "team1_h2h_winrate"]],
                    on="match_id", how="left")
    mdl["bat_h2h_winrate"] = np.where(mdl["batting_team"] == mdl["team1"],
                                       mdl["team1_h2h_winrate"],
                                       1 - mdl["team1_h2h_winrate"])
    mdl = mdl.drop(columns=["team1", "team2", "team1_h2h_winrate"])
    return mdl


# --------------------------------------------------------------------
# 6. Toss + chase pressure
# --------------------------------------------------------------------

def _add_toss_chase_features(mdl: pd.DataFrame, df: pd.DataFrame,
                             matches: pd.DataFrame) -> pd.DataFrame:
    """Toss flags + training-only venue bat-first winrate + chase-pressure
    stack. Chase features default to 0 for innings 1 (no target exists)."""
    # Toss flags
    mdl["batting_won_toss"] = (mdl["batting_team"] == mdl["toss_winner"]).astype(int)
    mdl["toss_chose_bat"]   = (mdl["toss_decision"] == "bat").astype(int)

    # Venue bat-first winrate — TRAINING-ERA ONLY
    first_inn_team = df[df["innings"] == 1].groupby("match_id")["batting_team"].first()
    mfm = matches.set_index("match_id")
    outcome = pd.DataFrame({
        "venue":          mfm["venue"],
        "first_inn_team": first_inn_team,
        "winner":         mfm["winner"],
    }).dropna()
    outcome["bat_first_won"] = (outcome["first_inn_team"] == outcome["winner"]).astype(int)
    outcome["year"] = mfm["date"].dt.year
    train_outcome  = outcome[outcome["year"] <= TRAIN_END]
    venue_bfw      = train_outcome.groupby("venue")["bat_first_won"].mean()
    g_bfw          = train_outcome["bat_first_won"].mean()
    mdl["venue_bat_first_winrate"] = mdl["venue"].map(venue_bfw).fillna(g_bfw)

    # Chase features — innings 2 only; innings 1 stays at neutral zeros
    inn1_totals = mdl[mdl["innings"] == 1].set_index("match_id")["cum_runs_total"]
    is_chase    = (mdl["innings"] == 2)
    mdl["target_score"]      = mdl["match_id"].map(inn1_totals).where(is_chase, 0).fillna(0)
    safe_target              = mdl["target_score"].clip(lower=1)
    mdl["chase_progress_pp"] = np.where(is_chase, mdl["cum_runs_after_pp"] / safe_target, 0.0)
    mdl["chase_progress_15"] = np.where(is_chase, mdl["cum_runs_after_15"] / safe_target, 0.0)
    mdl["req_runs_after_pp"] = np.where(is_chase, mdl["target_score"] - mdl["cum_runs_after_pp"], 0)
    mdl["req_runs_after_15"] = np.where(is_chase, mdl["target_score"] - mdl["cum_runs_after_15"], 0)
    mdl["req_rr_after_pp"]   = np.where(is_chase, mdl["req_runs_after_pp"] / 14.0, 0)
    mdl["req_rr_after_15"]   = np.where(is_chase, mdl["req_runs_after_15"] /  5.0, 0)
    mdl["chase_pressure_pp"] = np.where(is_chase, mdl["req_rr_after_pp"]  - mdl["powerplay_rr"], 0)
    mdl["chase_pressure_15"] = np.where(is_chase,
                                         mdl["req_rr_after_15"] - (mdl["cum_runs_after_15"] / 15.0),
                                         0)
    return mdl


# --------------------------------------------------------------------
# 7. Par scores (TRAINING-ONLY)
# --------------------------------------------------------------------

def _add_par_features(mdl: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Venue and venue × year par scores (training-only innings-1 means)
    plus score-vs-par deltas and overperform / underperform flags."""
    train_inn1 = mdl[(mdl["innings"] == 1) & train_mask]
    venue_par_total = train_inn1.groupby("venue")["cum_runs_total"].mean()
    venue_year_par  = train_inn1.groupby(["venue", "year"])["cum_runs_total"].mean()
    g_par = train_inn1["cum_runs_total"].mean()

    mdl["venue_par_total"] = mdl["venue"].map(venue_par_total).fillna(g_par)
    vy_idx = pd.MultiIndex.from_arrays([mdl["venue"], mdl["year"]])
    mdl["venue_year_par"] = pd.Series(vy_idx.map(venue_year_par).to_numpy(),
                                       index=mdl.index)
    mdl["venue_year_par"] = mdl["venue_year_par"].fillna(mdl["venue_par_total"])

    venue_par_pp = train_inn1.groupby("venue")["cum_runs_after_pp"].mean()
    venue_par_15 = train_inn1.groupby("venue")["cum_runs_after_15"].mean()
    mdl["par_after_pp"] = mdl["venue"].map(venue_par_pp).fillna(train_inn1["cum_runs_after_pp"].mean())
    mdl["par_after_15"] = mdl["venue"].map(venue_par_15).fillna(train_inn1["cum_runs_after_15"].mean())

    mdl["score_vs_venue_par"] = mdl["cum_runs_total"]    - mdl["venue_par_total"]
    mdl["score_vs_year_par"]  = mdl["cum_runs_total"]    - mdl["venue_year_par"]
    mdl["pp_vs_par"]          = mdl["cum_runs_after_pp"] - mdl["par_after_pp"]
    mdl["mid_vs_par"]         = mdl["cum_runs_after_15"] - mdl["par_after_15"]
    mdl["overperform_par"]    = (mdl["score_vs_venue_par"] >=  15).astype(int)
    mdl["underperform_par"]   = (mdl["score_vs_venue_par"] <= -15).astype(int)
    return mdl


# --------------------------------------------------------------------
# 8. Momentum + composite indicators
# --------------------------------------------------------------------

def _add_momentum_features(mdl: pd.DataFrame) -> pd.DataFrame:
    """Composite shape indicators built off earlier features."""
    mdl["wicket_intensity"]   = (mdl["wkts_after_pp"]    / 6.0
                                  + mdl["middle_wickets"]   / 9.0
                                  + mdl["death_wickets"]    / 5.0)
    mdl["acceleration_score"] = mdl["accel_pp_to_middle"] + mdl["accel_middle_to_death"]
    mdl["momentum_breakdown"] = (mdl["accel_middle_to_death"] < -1.5).astype(int)
    mdl["recovery_innings"]   = ((mdl["pp_collapse"] == 1) &
                                  (mdl["death_aggression"] > 10)).astype(int)
    mdl["stable_innings"]     = ((mdl["wkts_total"] <= 4) &
                                  (mdl["cum_runs_total"] >= mdl["venue_par_total"])).astype(int)
    return mdl


# --------------------------------------------------------------------
# 9. Encode + align to feature_columns.pkl
# --------------------------------------------------------------------

def _align_to_feature_columns(mdl: pd.DataFrame,
                              feature_columns: list[str]) -> pd.DataFrame:
    """One-hot encode venue, then align the matrix to feature_columns:
    add missing dummy columns as 0, drop extras, lock the order."""
    # One-hot encode venue
    mdl_enc = pd.get_dummies(mdl, columns=["venue"], prefix="venue",
                              drop_first=True)

    # Drop string helpers the model can't consume directly
    drop_cols = ["batting_team", "bowling_team", "toss_winner",
                 "toss_decision", "date", "target",
                 "batting_team_won_match"]
    mdl_enc = mdl_enc.drop(columns=[c for c in drop_cols if c in mdl_enc.columns])

    # Add any feature_columns that are missing (typically venue dummies for
    # venues that don't appear in the current data) as zero columns.
    missing = [c for c in feature_columns if c not in mdl_enc.columns]
    for col in missing:
        mdl_enc[col] = 0

    # Drop columns the model never saw at training time (extra venues,
    # raw helpers like wkts_after_pp / wkts_after_15 that aren't features).
    extras = [c for c in mdl_enc.columns if c not in feature_columns]
    mdl_enc = mdl_enc.drop(columns=extras)

    # Final ordering to exactly match feature_columns.pkl
    final_X = mdl_enc[feature_columns].copy()
    return final_X


# --------------------------------------------------------------------
# 10. Public API
# --------------------------------------------------------------------

def load_assets(model_path: str = MODEL_PATH,
                feature_cols_path: str = FEATURE_COLS_PATH):
    """Load the fitted model and its training-time feature column list."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feature_cols_path, "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns


def build_features(deliveries_path: str = DELIVERIES_PATH,
                   matches_path: str   = MATCHES_PATH,
                   feature_cols_path: str = FEATURE_COLS_PATH
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the prediction-ready feature matrix from the historical CSVs.

    Returns
    -------
    final_X : pd.DataFrame
        Shape (n_innings, len(feature_columns)). Columns are in the
        exact order saved in feature_columns.pkl. Ready for
        `model.predict_proba(final_X)`.
    mdl_meta : pd.DataFrame
        Identifier columns (match_id, innings, batting_team, bowling_team,
        venue, year, date, toss_winner, toss_decision, target). Same
        index as final_X — useful for showing match details in a UI.
    """
    # 1. Load + clean
    deliveries, matches = _load_and_clean(deliveries_path, matches_path)

    # 2. Innings-level base table
    mdl, df = _build_innings_table(deliveries, matches)

    # 3. Chronological training mask (for training-only feature builders)
    train_mask = mdl["year"] <= TRAIN_END

    # 4. Feature engineering — same order and same logic as training
    mdl = _add_advanced_phase_features(mdl)
    mdl = _add_venue_baselines(mdl, train_mask)
    mdl = _add_team_strength_features(mdl, df, matches)
    mdl = _add_toss_chase_features(mdl, df, matches)
    mdl = _add_par_features(mdl, train_mask)
    mdl = _add_momentum_features(mdl)

    # 5. Snapshot identifying columns BEFORE encoding / dropping
    mdl_meta = mdl[[c for c in META_COLS if c in mdl.columns]].copy().reset_index(drop=True)

    # 6. Load expected feature order and align
    with open(feature_cols_path, "rb") as f:
        feature_columns = pickle.load(f)
    final_X = _align_to_feature_columns(mdl, feature_columns).reset_index(drop=True)

    # 7. Validate before returning
    assert final_X.shape[1] == len(feature_columns), (
        f"Column count mismatch: final_X has {final_X.shape[1]} cols, "
        f"feature_columns has {len(feature_columns)}")
    assert list(final_X.columns) == list(feature_columns), (
        "Column order does not match feature_columns")
    assert len(final_X) == len(mdl_meta), (
        f"Row count mismatch between final_X ({len(final_X)}) "
        f"and mdl_meta ({len(mdl_meta)})")
    return final_X, mdl_meta


# --------------------------------------------------------------------
# 11. Standalone test — run `python feature_builder.py`
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Build features
    final_X, mdl_meta = build_features()
    print(f"final_X.shape         = {final_X.shape}")
    print(f"mdl_meta.shape        = {mdl_meta.shape}")

    # Re-load expected feature list and confirm alignment
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    assert final_X.shape[1] == len(feature_columns), "shape mismatch"
    assert list(final_X.columns) == list(feature_columns), "order mismatch"
    print(f"feature column count  = {len(feature_columns)} (matches feature_columns.pkl)")

    # Sanity-check by scoring with the trained model
    if Path(MODEL_PATH).exists():
        model, _ = load_assets()
        proba = model.predict_proba(final_X)[:, 1]
        print(f"P(win) mean / min / max = "
              f"{proba.mean():.3f} / {proba.min():.3f} / {proba.max():.3f}")

        # If targets are present, report a quick AUC for the test era
        if "target" in mdl_meta.columns and "year" in mdl_meta.columns:
            from sklearn.metrics import roc_auc_score
            test_idx = mdl_meta["year"] > TRAIN_END
            if test_idx.sum() > 0:
                auc = roc_auc_score(mdl_meta.loc[test_idx, "target"],
                                    proba[test_idx.values])
                print(f"Hold-out (year > {TRAIN_END}) ROC-AUC = {auc:.3f}")

    print("feature_builder OK — Streamlit-ready")
