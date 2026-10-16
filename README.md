# HHS Unaccompanied Children Care Load & Placement Demand Forecasting
**Developer:** Kandregula Uday Teja

## Overview
Multi-horizon forecasting pipeline (h = 1..30 days) and interactive risk dashboard for the HHS Unaccompanied Children Program.

## Key Result
Damped-momentum models (`DampedDrift`, `Ensemble_Damped`) beat a persistence (Naive) baseline at 3 to 30 day horizons in walk-forward validation. Tree-based and ridge models did not (RelMAE above 1.0), so they were not used.

## Methodology
- **No future leakage:** features at day t use only information known by the end of day t.
- **Walk-forward validation:** 5 expanding folds (60 days each, gap = h), scored only on observed days (720 observed, 355 interpolated days excluded from scoring).
- **Metric:** `RelMAE` = out-of-fold MAE divided by same-horizon Naive MAE (below 1.00 beats persistence).
- **Intervals:** calibrated on Folds 1 to 4 and tested on Fold 5.

## Limitations
- Weekend (Fri/Sat) values are interpolated, so weekend forecasts are unvalidated.
- The test window is a calm, low-load period; behavior during a surge is untested.
- Model selection reused the same folds, so reported skill may be slightly optimistic.
- The capacity threshold is an adjustable assumption, not an official limit.

## Files
- `hhs_forecast_pipeline_final.py`: cleaning, EDA, validation, forecasting
- `app.py`: Streamlit dashboard
- `*.csv`, `meta.json`: pre-computed pipeline outputs
- `requirements.txt`: pinned dependencies

## Run
```bash
pip install -r requirements.txt
python hhs_forecast_pipeline_final.py   # optional: regenerate outputs
streamlit run app.py
```
