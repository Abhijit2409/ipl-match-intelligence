<!--
  README.md  ·  IPL Match Outcome & Phase Analysis
  Author: Abhijit Mishra
-->

<div align="center">

# IPL Match Outcome & Phase Analysis

### Where IPL matches are actually won — and how the Impact Player rule rewired the math.

A descriptive-to-predictive analytics project across **18 IPL seasons** (2008&ndash;2025): phase-level scoring intelligence, era-shift quantification, and a **134-feature XGBoost win-probability model** validated on a chronological 2024&ndash;2025 hold-out at **ROC-AUC 0.911**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-Strategic_Model-F26A2E?logo=xgboost&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-LR_·_RF-F7931E?logo=scikit-learn&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?logo=powerbi&logoColor=black)](#)
[![Status](https://img.shields.io/badge/status-portfolio-1E88E5)](#)

</div>

> **TL;DR** &nbsp;Most IPL matches are won by the team that takes **2 of 3 phases** of the innings. Since the **Impact Player rule (2023)**, the **death-overs run rate has lifted by ~1 run per over** and **200+ totals have grown 4&times;**. A 134-feature XGBoost model trained chronologically (≤2023) and tested on 2024&ndash;2025 hits **0.911 ROC-AUC** with **Brier 0.132** — a probability surface clean enough to put on a broadcast.

---

## 1 · Problem Statement

T20 commentary keeps reaching for the same one-liners — *"the powerplay sets the platform"*, *"finishers win matches"* — but the underlying question rarely gets answered with numbers:

> **At which phase of an IPL innings is the match actually decided, and how has the 2023 Impact Player rule changed that calculus?**

This project answers that with ball-by-ball data, leakage-safe feature engineering, three model classes, and an interactive product layer.

---

## 2 · Why this project matters

- **For analytics teams** &mdash; the same phase-decomposition logic generalizes to any T20 league or franchise's playbook.
- **For broadcasters & sportsbooks** &mdash; the `predict_live` surface and chase-pressure features power a real-time win-probability widget without rebuilding the pipeline.
- **For coaches & strategists** &mdash; the wickets-vs-runs trade-off and venue-adjusted scoring deltas turn vague "build a partnership" instincts into measurable targets.
- **For me, as a portfolio piece** &mdash; it demonstrates business framing &rarr; data engineering &rarr; modeling &rarr; productization end to end.

---

## 3 · Key questions answered

| # | Question | Where it's answered |
|:--|:---------|:--------------------|
| 1 | Which phase carries the strongest signal of who wins? | §7 Insights · §8 Feature importance |
| 2 | How many phases does the eventual winner usually take? | §7 Insight #2 |
| 3 | Did the Impact Player rule actually change scoring patterns? | §7 Insights #3&ndash;5 |
| 4 | Are wickets or runs more predictive at each phase? | §8 Feature importance |
| 5 | At what point in the innings does the result become &ldquo;reliable&rdquo;? | §8 Stage progression |
| 6 | Is post-2023 IPL more or less predictable than pre-2023? | §8 Era comparison |
| 7 | Can we put a calibrated probability on a chase situation? | §8 Calibration · §9 Streamlit app |

---

## 4 · Project workflow

```
   Raw CSVs              Feature pipeline           Validation                  Product layer
 ┌───────────┐         ┌──────────────────┐      ┌────────────────┐          ┌──────────────────┐
 │ matches   │         │  Phase pivots    │      │  Chronological │          │  Power BI        │
 │ deliveries│  ─────▶ │  Phase-level RR  │ ───▶ │  train ≤ 2023  │ ──────▶ │  5-page report   │
 │ 2008-2025 │         │  Venue par (T1)  │      │  test  > 2023  │          │                  │
 │           │         │  Team rolling    │      │                │          │  Streamlit app   │
 │           │         │  Chase pressure  │      │  ROC-AUC 0.911 │          │  Innings explorer│
 └───────────┘         └──────────────────┘      └────────────────┘          └──────────────────┘
       │                       │                          │                          │
   reproducible             134 leakage-safe         Brier 0.132              Live `predict_live()`
   loader, alias map        features, T1-only        log-loss 0.403           +  `whatif_chase()`
                            baselines
```

The pipeline is split into three reusable modules: **`feature_builder.py`** (the leakage-safe feature pipeline), **`train_model.py` / `train_model.ipynb`** (training + persistence), and **`app.py`** (Streamlit UI). The Power BI report consumes the analytic CSVs exported from the notebook.

---

## 5 · Dataset

| Attribute | Value |
|:----------|:------|
| Source format | Two CSVs &mdash; match-level + ball-by-ball |
| Coverage | IPL seasons 2008 &rarr; 2025 |
| Matches | **1,169** (after dropping abandoned) |
| Innings used | **2,292** (innings 1 &amp; 2 only; super-overs excluded) |
| Ball-level rows | ~278K |
| Time span | ~17 calendar years |
| Source files | `matches_updated_ipl_upto_2025.csv`, `deliveries_updated_ipl_upto_2025.csv` |

Column-name aliases are normalized at load time (`inning` &rarr; `innings`, `batsman_runs` &rarr; `runs_off_bat`, `isWide` &rarr; `wides`, etc.) so the pipeline is dump-agnostic.

---

## 6 · Methodology

### 6.1 &nbsp;Cleaning

- Drop super-overs (innings 3+) and abandoned matches (no winner recorded).
- Coerce extras (`wides`, `noballs`, `byes`, `legbyes`, `penalty`) to numeric, fillna 0.
- Phase-classify every ball: **Powerplay** (overs 1&ndash;6), **Middle** (7&ndash;15), **Death** (16&ndash;20).

### 6.2 &nbsp;Feature engineering &mdash; 134 leakage-safe features

Built in dependency order so each step has the inputs the next needs:

| Family | Examples | Leakage prevention |
|:-------|:---------|:-------------------|
| **Phase metrics** | runs / wickets / RR per phase, RPW, collapse flags | Computed within an innings only |
| **Resource & tempo** | cumulative runs, wickets remaining, acceleration deltas | Derived from observable phase totals |
| **Venue context** | `v_pp_avg`, `v_*_diff`, `venue_par_total`, `venue_year_par` | **Training-era-only baselines** (year ≤ 2023) |
| **Team strength** | rolling form (career, last 10), batting/bowling avg | `groupby(team).shift(1).expanding/rolling` &mdash; never sees current match |
| **Head-to-head** | `bat_h2h_winrate` (canonicalized pair key) | Same shift-and-expand pattern |
| **Toss & chase** | `batting_won_toss`, `venue_bat_first_winrate`, full chase-pressure stack | Venue toss bias from training era only |
| **Momentum** | wicket intensity, recovery / stable / breakdown indicators | Composites of upstream leakage-safe features |

### 6.3 &nbsp;Validation strategy

A **chronological hold-out** &mdash; train on seasons through **2023**, test on **2024&ndash;2025**. This respects the temporal structure (rule changes, evolving strategy) and ensures the test set is genuinely *future*. Hyperparameter tuning uses `TimeSeriesSplit` *inside* the training window so the hold-out is never seen during model selection.

> The Impact Player rule (2023) is included as a feature (`impact_era_after`), so the model trains on both pre- and post-rule innings and explicitly accounts for the regime shift &mdash; it is **not** trained only on the pre-rule period.

---

## 7 · Key insights

### Insight 1 &mdash; The middle overs are where matches are quietly won

Across 2,292 innings, winning teams out-score losing teams in **every phase**, but the gap is widest in the middle (overs 7&ndash;15):

| Phase | Lost (avg runs) | Won (avg runs) | **Diff** |
|:------|---:|---:|---:|
| Powerplay | 44.9 | 50.7 | **+5.8** |
| Middle | 66.0 | 73.9 | **+7.9** |
| Death | 43.2 | 47.3 | **+4.2** |

Wickets tell the same story in reverse &mdash; winners lose **0.65 fewer wickets** in the powerplay, **0.88 fewer** in the middle, and **1.03 fewer** in the death overs.

### Insight 2 &mdash; You need to win 2 of 3 phases to win the match

Of all completed matches in the 2008&ndash;2025 era:

- **59.2%** were won by the team that took **2 of 3 phases**
- **11.5%** of matches were won by a team that **dominated all three**
- Only **0.2%** of matches were won by a team that **didn't win a single phase**

Translation: **lose two phases and you almost certainly lose the match.**

### Insight 3 &mdash; The Impact Player rule shifted scoring forward by ~10 runs per innings

| Phase | Before 2023 | After 2023 | **&Delta;** |
|:------|---:|---:|---:|
| Powerplay (avg runs) | 46.0 | 55.7 | **+9.7** |
| Middle (avg runs) | 68.1 | 78.1 | **+10.0** |
| Death (avg runs) | 44.4 | 48.3 | **+3.9** |

The middle and powerplay carry the largest jump &mdash; deeper batting lineups attack earlier rather than waiting for the death.

### Insight 4 &mdash; Death-overs run rate climbed by &gt;1 RPO

Death-overs run rate **9.81 &rarr; 10.87 RPO** (Before vs. After 2023). Every season since 2023 has cleared the pre-2023 league average.

### Insight 5 &mdash; 200+ innings went from a rarity to the new normal

| Era | Innings | 200+ scores | **Share** |
|:----|---:|---:|---:|
| Before 2023 | 1,864 | 130 | **7.0%** |
| After 2023 | 428 | 129 | **30.1%** |

A **4.3&times; multiplier** on the share of innings crossing 200.

### Insight 6 &mdash; Wickets in hand outweigh raw runs at the death

Across the strategic XGBoost model, **wickets-related features dominate**:

1. `wkts_total` (importance 0.097)
2. `wicket_intensity` (0.069)
3. `chase_progress_15` (0.050)
4. `chase_pressure_15` (0.040)
5. `req_runs_after_15` (0.027)

Settled wickets at over 15 are the single strongest tell of the match outcome &mdash; ahead of cumulative runs scored.

### Insight 7 &mdash; Post-2023 IPL is *slightly less* predictable than pre-2023

Same model class, same 134 features &mdash; trained and tested *within each era separately*:

- **Before 2023:** ROC-AUC ~0.80
- **After 2023:** ROC-AUC ~0.77

The Impact Player rule introduced strategic flexibility (extra batter, late hitters, deeper chases) that adds genuine *strategic noise*. The model still works &mdash; the game changed quantitatively, not qualitatively.

---

## 8 · Predictive modeling

Three model classes were benchmarked across three nested feature stages. **Stage A** sees only the powerplay; **Stage B** adds the middle overs; **Stage C** adds the death overs.

### 8.1 &nbsp;Stage progression (chronological hold-out: train ≤ 2023, test = 2024&ndash;2025)

| Stage | Logistic Regression | Random Forest | XGBoost |
|:------|:---:|:---:|:---:|
| **A** &mdash; Powerplay only | 0.673 | 0.613 | 0.653 |
| **B** &mdash; + Middle overs | 0.760 | 0.732 | 0.735 |
| **C** &mdash; Full innings | 0.811 | 0.842 | **0.850** |

> **Match becomes "reliable" (AUC ≥ 0.75) at the end of the middle overs**, not in the powerplay. Linear models lead at early stages; tree ensembles take over at full innings.

### 8.2 &nbsp;Strategic Stage C+ (with team form, toss, par, momentum) &mdash; the final model

Adding **39 strategic features** (rolling team form, head-to-head, toss flags, par-score deltas, full chase-pressure stack, momentum composites) lifts the best model substantially:

| Metric | Stage C baseline | **Stage C+ Strategic** |
|:-------|:---:|:---:|
| ROC-AUC | 0.850 | **0.911** |
| Accuracy | 74.5% | **81.6%** |
| Precision | 0.707 | **0.773** |
| Recall | 0.823 | **0.894** |
| Brier | &mdash; | **0.132** |
| Log-loss | &mdash; | **0.403** |

### 8.3 &nbsp;Top-10 feature importance &mdash; strategic XGBoost

```
wkts_total              ████████████████████  0.097
wicket_intensity        ██████████████        0.069
chase_progress_15       ██████████            0.050
chase_pressure_15       ████████              0.040
req_runs_after_15       █████                 0.027
score_vs_year_par       █████                 0.025
req_rr_after_15         █████                 0.025
score_vs_venue_par      █████                 0.023
cum_runs_total          ████                  0.022
death_collapse          ████                  0.021
```

Phase-3 strategic features (chase pressure, par-score deltas, wicket intensity) crowd the top-10 above raw runs/wickets &mdash; **how the team navigates pressure matters more than how much they scored in absolute terms.**

### 8.4 &nbsp;Methodological integrity

- Single chronological hold-out used for all final test scores.
- Hyperparameter tuning uses `TimeSeriesSplit` on the training window only.
- Venue baselines, par scores, and toss biases are computed from training-era rows only.
- Stage feature lists are strict supersets &mdash; no information from later phases bleeds into earlier-stage models.
- Calibration is solid at extremes (predicted &lt; 0.05 &rarr; observed ~0%; predicted &gt; 0.95 &rarr; observed ~100%) and slightly overconfident in the 0.30&ndash;0.70 mid-range.

---

## 9 · Dashboard &amp; App

### 9.1 &nbsp;Power BI &mdash; *5-page executive narrative*

| Page | Content |
|:-----|:--------|
| **1 · Executive Summary** | Headline KPIs (matches analyzed, % won by 2+ phases, death-RR lift, 200+ growth), donut of phases-won-by-winner, era scoring lift |
| **2 · Where Matches Are Won** | Phase-wise runs &amp; wickets &mdash; winners vs losers, with sentence-style takeaways |
| **3 · Impact Player Effect** | Runs &amp; wickets by phase &times; era, hero callout for the +9.7 powerplay shift |
| **4 · Death Overs &amp; 200+ Explosion** | Season-level death RR with era reference lines, 200+ rate side-by-side cards |
| **5 · Strategic Takeaways** | Five sharp claim-cards plus a closing line judges remember |

Built in a deliberately understated **navy / saffron / indigo** palette &mdash; one font family (Segoe UI), one chart-title voice (sentence-style insight), one number per page that "pops".

### 9.2 &nbsp;Streamlit &mdash; *interactive innings explorer*

`app.py` wraps the trained 134-feature XGBoost behind a navy / royal-blue / electric-cyan / gold theme:

- **Sidebar filters** &mdash; year, venue, batting team, bowling team, innings, predicted-probability range, single-click reset.
- **KPI strip** &mdash; filtered innings count, mean / peak P(win), feature count.
- **Innings explorer** &mdash; selectbox over 2,292 historical innings, Plotly gauge with confidence band (Low control / Balanced / Advantage / Strong control).
- **Phase summary** &mdash; nine cards pulling phase runs / wickets / run rates direct from `final_X`.
- **Strategic context cards** &mdash; chase pressure (innings-2 only), venue par, head-to-head winrate, toss flag, era flag &mdash; each with a plain-English reading.
- **Population view** &mdash; histogram of P(win) across the filter, average-by-team horizontal bar.
- **Top-10 feature importance** + filtered data table + methodology cards.

> The app is a **historical innings explorer**, not a manual future-match simulator. Manual feature entry would require rebuilding all 134 features from a synthetic state &mdash; deliberately out of scope.

---

## 10 · Tech stack

| Layer | Tools |
|:------|:------|
| **Language** | Python 3.10+ |
| **Data wrangling** | `pandas`, `numpy` |
| **Modeling** | `scikit-learn` (LogisticRegression, RandomForest, GradientBoosting), `xgboost` |
| **Validation** | `TimeSeriesSplit`, `roc_auc_score`, `brier_score_loss`, calibration curves |
| **Visualization (notebook)** | `matplotlib` |
| **Visualization (app)** | `plotly` |
| **App framework** | `streamlit` |
| **Dashboard** | Power BI Desktop |
| **Persistence** | `pickle` (model + feature schema) |
| **Versioning** | Git, GitHub |

---

## 11 · Folder structure

```
ipl-match-intelligence/
│
├── data/
│   ├── matches_updated_ipl_upto_2025.csv          # match-level
│   └── deliveries_updated_ipl_upto_2025.csv       # ball-by-ball
│
├── notebooks/
│   ├── IPL_Analytics.ipynb                        # full descriptive + predictive analysis
│   └── train_model.ipynb                          # clean training pipeline (notebook form)
│
├── src/
│   ├── train_model.py                             # CLI training script
│   ├── feature_builder.py                         # leakage-safe feature pipeline
│   └── app.py                                     # Streamlit UI
│
├── artifacts/
│   ├── ipl_model.pkl                              # fitted XGBoost (Stage C+ Strategic)
│   └── feature_columns.pkl                        # exact 134-column schema
│
├── output_tables/
│   ├── a1_phase_runs_winners_vs_losers.csv
│   ├── a2_phase_wickets_winners_vs_losers.csv
│   ├── a3_phases_won_by_match_winner.csv
│   ├── a4_impact_era_runs_by_phase.csv
│   ├── a5_death_runrate_by_era.csv
│   ├── a5_death_runrate_by_season.csv
│   ├── a6_200_plus_by_era.csv
│   ├── model_comparison.csv
│   ├── model_phase3_metrics.csv
│   ├── model_phase3_feature_importance.csv
│   └── ... (all analysis tables exported as CSV)
│
├── powerbi/
│   └── IPL_Match_Intelligence.pbix                # 5-page Power BI report
│
├── assets/
│   ├── dashboard_overview.png
│   ├── streamlit_app.png
│   ├── feature_importance.png
│   └── era_comparison.png
│
├── requirements.txt
└── README.md
```

---

## 12 · How to run

### 12.1 &nbsp;Install

```bash
git clone https://github.com/<your-handle>/ipl-match-intelligence.git
cd ipl-match-intelligence
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 12.2 &nbsp;Reproduce the analysis (notebook)

```bash
jupyter notebook notebooks/IPL_Analytics.ipynb
```

The notebook is sectioned: descriptive analysis (§1&ndash;6), Phase-1 takeaways (§7), seasonal team wins (§8), Phase-2 modeling (§9), advanced features and tuning (§10), and Phase-3 strategic intelligence (§11). All result tables export to `output_tables/` as CSV.

### 12.3 &nbsp;Re-train the model

```bash
python src/train_model.py
# produces:  artifacts/ipl_model.pkl  +  artifacts/feature_columns.pkl
```

### 12.4 &nbsp;Launch the Streamlit app

```bash
py -m streamlit run src/app.py
# opens at http://localhost:8501
```

### 12.5 &nbsp;Open the Power BI report

Open `powerbi/IPL_Match_Intelligence.pbix` in Power BI Desktop. The report points at the CSVs in `output_tables/`.

---

## 13 · Screenshots

> Replace the placeholders with your actual exports / screenshots.

**Power BI &mdash; Executive Summary page**

![Power BI &mdash; Executive Summary](assets/dashboard_overview.png)

**Streamlit app &mdash; innings explorer**

![Streamlit innings explorer](assets/streamlit_app.png)

**Top-10 strategic feature importance**

![Feature importance](assets/feature_importance.png)

**Era comparison &mdash; before vs after Impact Player**

![Era comparison](assets/era_comparison.png)

---

## 14 · Future improvements

- **Player-level features** &mdash; the `bat_*` / `bowl_*` namespace already cleanly nests; adding individual batting and bowling rolling stats is a natural extension.
- **Isotonic / Platt calibration** &mdash; tightens the mid-range overconfidence; expected to push Brier from 0.132 closer to 0.10.
- **Live `predict_live()` integration** &mdash; the prediction surface is already factored as a stateless function. Wrapping it in a tiny FastAPI endpoint would unlock real-time scoring during matches.
- **Bookmark-driven Power BI navigation** &mdash; a "story tour" mode that walks judges through the five pages with a single click.
- **Ball-level boundary features** &mdash; current pipeline uses phase-aggregated proxies; ingesting per-ball boundary flags would sharpen aggression and momentum signals.
- **Weather and toss-time context** &mdash; dew-prone evening venues like Chinnaswamy and Eden Gardens drive part of the residual chase variance.

---

## 15 · Author

<div align="left">

### Abhijit Mishra
**Data Analyst &middot; Insight-first &middot; Open to opportunities in Canada**

I work at the intersection of business framing, careful data engineering, and product storytelling. This project is the long-form version of how I approach problems &mdash; start with the question, build the pipeline that genuinely answers it, validate without leakage, and ship a product layer that lets a non-technical stakeholder play with the result.

- **Email**  &nbsp;[abhijitmishra0103@gmail.com](mailto:abhijitmishra0103@gmail.com)
- **LinkedIn** &nbsp;`<add your LinkedIn URL>`
- **GitHub** &nbsp;`<add your GitHub URL>`
- **Portfolio** &nbsp;`<add your portfolio URL>`

</div>

---

<div align="center">

*Built with curiosity. Validated with rigor. Designed for the stakeholder, not the coder.*

</div>
