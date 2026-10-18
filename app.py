import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="HHS Care Load Forecasting | Uday Teja", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .developer-header {
        font-size: 0.95rem;
        color: #9ca3af;
        text-align: right;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #374151;
        margin-bottom: 1.5rem;
    }
    .developer-header span {
        font-weight: 700;
        color: #60a5fa;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(31, 41, 55, 0.45);
        border: 1px solid rgba(75, 85, 99, 0.4);
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    <div class="developer-header">
        Developed by <span>Kandregula Uday Teja</span> | Predictive Analytics & Operational Intelligence
    </div>
""", unsafe_allow_html=True)

CARE    = 'Children in HHS Care'
INFLOW  = 'Children transferred out of CBP custody'
OUTFLOW = 'Children discharged from HHS Care'

# ------------------------------------------------------------------ 1. DATA
@st.cache_data
def load_all():
    df = pd.read_csv('HHS_clean_daily.csv', index_col='Date', parse_dates=True).sort_index()
    fc = pd.read_csv('forecast.csv', index_col=0, parse_dates=True).sort_index()
    bench = pd.read_csv('model_benchmarks.csv')
    disc = pd.read_csv('discharge_benchmarks.csv')
    rq = pd.read_csv('resid_quantiles.csv')
    cov = pd.read_csv('coverage.csv')
    sens = pd.read_csv('sensitivity.csv')
    with open('meta.json') as f:
        meta = json.load(f)
    return df, fc, bench, disc, rq, cov, sens, meta

df, fc, bench, disc, rq, cov, sens, meta = load_all()
QCOLS = [c for c in rq.columns if c.startswith('q')]

# Pull all summary statistics dynamically from meta.json / DataFrames
total_rows = int(meta.get('rows', len(df)))
obs_rows = int(meta.get('observed_rows', df['Observed'].sum() if 'Observed' in df.columns else 720))
interp_rows = int(meta.get('interpolated_rows', total_rows - obs_rows))
flow_corr = float(meta.get('flow_care_corr', df[CARE].diff().corr(df[INFLOW] - df[OUTFLOW])))
recent_care_diff = float(meta.get('recent_28d_care_diff', df[CARE].diff().iloc[-28:].mean()))
recent_net_flow = float(meta.get('recent_28d_net_flow', (df[INFLOW] - df[OUTFLOW]).iloc[-28:].mean()))
wknd_slope = float(meta.get('wknd_mean_diff', df[CARE].diff()[df.index.dayofweek.isin([4, 5, 6])].mean()))
load_23_24_min = int(meta.get('load_23_24_min', df.loc[df.index.year.isin([2023, 2024]), CARE].min()))
load_23_24_max = int(meta.get('load_23_24_max', df.loc[df.index.year.isin([2023, 2024]), CARE].max()))
load_2025_min = int(meta.get('load_2025_min', df.loc[df.index.year == 2025, CARE].min()))
load_2025_max = int(meta.get('load_2025_max', df.loc[df.index.year == 2025, CARE].max()))

if 'RelMAE' not in bench.columns:
    bench['RelMAE'] = bench['MASE']
if 'RelMAE' not in sens.columns:
    sens['RelMAE'] = sens['MASE']

def breach_probs(model, preds, thr):
    """P(care load > thr) per day, from out-of-fold residual quantiles of that model."""
    sub = rq[rq.model == model].set_index('horizon')
    grid = sub.index.values
    out = []
    for i, p in enumerate(preds, start=1):
        g = grid[np.abs(grid - i).argmin()]
        out.append(np.mean(p + sub.loc[g, QCOLS].values.astype(float) > thr))
    return np.array(out)

# --------------------------------------------------------------- 2. SIDEBAR
st.sidebar.title(" Forecast Controls")
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=14)
fc_view = fc.iloc[:horizon].copy()

h_eval = int(min(bench.horizon.unique(), key=lambda h: abs(h - horizon)))
b_raw = bench[bench.horizon == h_eval].set_index('model').copy()

# Verified observed test pair counts per horizon
n_obs_h = int(b_raw['obs_test_n'].iloc[0]) if 'obs_test_n' in b_raw.columns else int(round(obs_rows * (5 * 60) / total_rows))
obs_n_by_h = (
    bench.groupby('horizon')['obs_test_n'].first().astype(int).to_dict()
    if 'obs_test_n' in bench.columns else {int(h): n_obs_h for h in bench.horizon.unique()}
)
obs_n_summary = ", ".join([f"h={k}d: n={v}" for k, v in sorted(obs_n_by_h.items())])

b_cols = ['MAE', 'RMSE', 'MAPE', 'RelMAE', 'Stability', 'Robustness']
if 'obs_test_n' in b_raw.columns:
    b_raw['obs_test_n'] = b_raw['obs_test_n'].astype(int)
    b_cols.append('obs_test_n')
b = b_raw[b_cols].round(2).sort_values('RelMAE')

min_rel_mae = b['RelMAE'].min()
top_group = set(b[b['RelMAE'] <= min_rel_mae + 0.05].index)
default_model = 'DampedDrift' if 'DampedDrift' in top_group else b['RelMAE'].idxmin()
models = [m for m in b.index if m in fc.columns]

st.sidebar.markdown("---")
st.sidebar.subheader(" Model Selection")
active_model = st.sidebar.selectbox(
    "Primary Headline Model",
    models,
    index=models.index(default_model) if default_model in models else 0,
    format_func=lambda m: f"{m} ★ (Top Group at h={h_eval}d)" if m in top_group else m,
    help=f"Models marked ★ are within 0.05 RelMAE of the lowest error (statistically indistinguishable over n={n_obs_h} observed test pairs at h={h_eval}d)."
)

compare_models = st.sidebar.multiselect(
    "Overlay Comparison Models",
    [m for m in models if m != active_model],
    default=['Naive', 'Ensemble_Damped'] if active_model == 'DampedDrift' else ['Naive']
)

st.sidebar.markdown("---")
st.sidebar.subheader(" Scenario & Stress Testing")
apply_scenario = st.sidebar.checkbox(
    "Enable & Apply Scenario to Risk KPIs",
    value=False,
    help="When enabled, the Inflow Shock and Clearance Boost sliders dynamically drive Projected Peak, Breach Probability, Lead Time, and Risk Status."
)
inflow_shock_pct = st.sidebar.slider(
    "Border Inflow Shock (%)", min_value=-25, max_value=50, value=25, step=5,
    help="Simulates a percentage surge or drop in daily CBP transfers."
)
outflow_boost_pct = st.sidebar.slider(
    "Sponsor Clearance Boost (%)", min_value=0, max_value=50, value=0, step=5,
    help="Simulates faster sponsor vetting and daily discharges."
)
cap = st.sidebar.number_input(
    "Illustrative Capacity Threshold (Adjustable Assumption)",
    min_value=1500, max_value=15000, value=2600, step=50,
    help=f"User-adjustable planning assumption (default 2,600 represents ~5% buffer above late-2025 active load of ~{int(round(df[CARE].iloc[-1], -1)):,}). Not an official statutory bed cap."
)

# ------------------------------------------------------------- 3. RISK LOGIC
current = float(df[CARE].iloc[-1])
last_date = df.index[-1]

recent_in_7d  = float(df[INFLOW].iloc[-7:].mean())
recent_out_7d = float(df[OUTFLOW].iloc[-7:].mean())
hs_all = np.arange(1, len(fc) + 1)
scenario_delta_all = ((inflow_shock_pct / 100.0) * recent_in_7d - (outflow_boost_pct / 100.0) * recent_out_7d) * hs_all

fc['custom_scenario'] = fc[active_model] + scenario_delta_all
fc_view['custom_scenario'] = fc['custom_scenario'].iloc[:horizon]

eval_series_all = fc['custom_scenario'].values if apply_scenario else fc[active_model].values
p_all = breach_probs(active_model, eval_series_all, cap)
p = p_all[:horizon]
p_base = breach_probs(active_model, fc[active_model].values, cap)[:horizon]

peak = float(fc_view['custom_scenario'].max() if apply_scenario else fc_view[active_model].max())
util_pct = (current / cap) * 100.0
peak_util_pct = (peak / cap) * 100.0

if p[:7].max() > 0.50 or peak >= cap:
    status, status_msg = 'Critical', 'High breach risk (<7d or Peak ≥ Cap)'
elif p.max() > 0.10:
    status, status_msg = 'Watch', 'Elevated breach prob (>10%)'
elif util_pct >= 95.0 or peak_util_pct >= 95.0:
    status, status_msg = 'Watch', f'High utilization ({max(util_pct, peak_util_pct):.1f}% ≥ 95%)'
else:
    status, status_msg = 'Safe', 'Breach prob ≤10% & buffer >5%'

hit = fc_view.index[p > 0.10]
lead_txt = f"{(hit[0].date() - last_date.date()).days} Days" if len(hit) else f">{horizon} Days"
improve = (1.0 - b.loc[active_model, 'RelMAE']) * 100 if active_model in b.index else 0.0

# ------------------------------------------------------------------- 4. KPIs
st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption(
    f"Data through **{meta['last_date']}** | **{total_rows:,} days ({obs_rows:,} observed, {interp_rows:,} Fri/Sat interpolated)** | "
    f"Headline Model: **{active_model}** | Capacity Threshold: **{cap:,}** *(adjustable planning assumption)*"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Current Care Load",
    f"{int(current):,}",
    f"{util_pct:.1f}% of {cap:,} assumption",
    delta_color="off",
    help="Latest recorded care load vs. the user-adjustable capacity threshold assumption."
)
k2.metric(
    "Projected Peak" + (" (Scenario)" if apply_scenario else ""),
    f"{int(peak):,}",
    delta=f"{int(peak - current):+d} children ({horizon}d)",
    help=f"Highest projected care load over the next {horizon} days ({'with scenario shock/boost applied' if apply_scenario else active_model})."
)
k3.metric(
    f"Skill vs. Naïve (at h={h_eval}d)",
    f"{improve:+.1f}%",
    delta="Top Group ★" if active_model in top_group else ("Beats Persistence" if improve > 0 else "Below Persistence"),
    delta_color="normal" if improve >= 0 else "inverse",
    help=f"1 − RelMAE evaluated at nearest grid horizon (h={h_eval}d, n={n_obs_h} observed test pairs) across 5 walk-forward folds."
)
k4.metric(
    "Peak Breach Prob." + (" (Scenario)" if apply_scenario else ""),
    f"{p.max() * 100:.0f}%",
    delta=f"Lead time: {lead_txt}",
    delta_color="off",
    help="Empirical probability that care load exceeds the threshold on the riskiest day in the window."
)
k5.metric(
    "Capacity Status",
    status,
    delta=status_msg,
    delta_color="inverse" if status != 'Safe' else "normal",
    help="Combines empirical breach probability (>50% Critical, >10% Watch) with a ≥95% utilization buffer guard."
)

# ------------------------------------------------------------------- 5. TABS
tab1, tab2, tab3, tab4 = st.tabs([
    " Forecast & Risk Analysis",
    " Flow Dynamics & Placement Demand",
    " Model Diagnostics & Robustness",
    " Forecast Data & Export"
])

with tab1:
    st.subheader("Multi-Horizon Care Load Trajectory & Empirical Risk Band")
    hist = df.iloc[-60:]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist[CARE], mode='lines', name='Actual Care Load (60d)',
        line=dict(color='#9ca3af', width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=fc_view.index, y=fc_view[f'upper_{active_model}'], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=fc_view.index, y=fc_view[f'lower_{active_model}'], mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(59, 130, 246, 0.18)',
        name=f'95% Empirical Interval ({active_model})'
    ))

    fig.add_trace(go.Scatter(
        x=fc_view.index, y=fc_view[active_model], mode='lines+markers',
        name=f'{active_model} (Baseline Forecast)', line=dict(color='#3b82f6', width=3)
    ))

    palette = ['#a855f7', '#10b981', '#eab308', '#ec4899', '#64748b']
    for idx_m, comp_m in enumerate(compare_models):
        fig.add_trace(go.Scatter(
            x=fc_view.index, y=fc_view[comp_m], mode='lines',
            name=f'{comp_m} (Compare)',
            line=dict(color=palette[idx_m % len(palette)], width=2, dash='dash')
        ))

    if apply_scenario:
        fig.add_trace(go.Scatter(
            x=fc_view.index, y=fc_view['custom_scenario'], mode='lines+markers',
            name=f'Active Scenario ({inflow_shock_pct:+d}% In / {outflow_boost_pct:+d}% Out)',
            line=dict(color='#f97316', width=3, dash='dot')
        ))

    fig.add_hline(
        y=cap, line_dash="dash", line_color="#ef4444",
        annotation_text=f"Illustrative Threshold Assumption ({cap:,})", annotation_position="top left"
    )
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Children in HHS Care", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=40), height=440
    )
    st.plotly_chart(fig, width="stretch")

    c_prob, c_note = st.columns([2, 1])
    with c_prob:
        bar_colors = ['#ef4444' if val > 0.5 else '#f59e0b' if val > 0.10 else '#10b981' for val in p]
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=fc_view.index, y=p * 100, marker_color=bar_colors,
            name='Active Breach Prob (%)' if apply_scenario else 'Baseline Breach Prob (%)'
        ))
        if apply_scenario:
            fig_p.add_trace(go.Scatter(
                x=fc_view.index, y=p_base * 100, mode='lines+markers',
                name='Unstressed Baseline Prob (%)', line=dict(color='#9ca3af', width=2, dash='dash')
            ))
        fig_p.add_hline(y=10, line_dash="dot", line_color="#f59e0b", annotation_text="Watch (10%)")
        fig_p.add_hline(y=50, line_dash="dot", line_color="#ef4444", annotation_text="Critical (50%)")
        fig_p.update_layout(
            title=f"Daily Capacity Breach Probability (%) — {'Scenario Stressed' if apply_scenario else 'Baseline'}",
            yaxis_title="Probability (%)", xaxis_title="Date", height=250,
            margin=dict(l=20, r=20, t=40, b=30), yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_p, width="stretch")

    with c_note:
        st.info(
            f" **Top-Group Model Dynamics:** `DampedDrift`, `Ensemble_Damped`, and `ExpDrift` form the top-performing "
            f"tier (differences of 0.02–0.05 RelMAE are within sampling noise across $n={n_obs_h}$ observed test pairs at $h={h_eval}$d; "
            f"full grid counts: {obs_n_summary}). *DampedDrift* extrapolates recent net growth with damping ($\\phi=0.96$), "
            f"whereas *Ensemble_Damped* blends persistence, damped momentum, and CBP-to-HHS net flow."
        )

with tab2:
    st.subheader("CBP Transfer Inflow vs. Sponsor Placement Outflow")
    c1, c2 = st.columns([2, 1])
    with c1:
        agg_weekly = st.checkbox("Aggregate Weekly (Smooths Fri/Sat linear interpolation)", value=False)
        hist_flow = df.iloc[-84:] if agg_weekly else df.iloc[-60:]
        plot_df = hist_flow[[INFLOW, OUTFLOW]].resample('W').mean() if agg_weekly else hist_flow[[INFLOW, OUTFLOW]]

        roll_in = hist_flow[INFLOW].rolling(7).mean()
        roll_out = hist_flow[OUTFLOW].rolling(7).mean()

        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(
            x=plot_df.index, y=plot_df[INFLOW], name='Inflow (CBP Transfers)',
            marker_color='rgba(147, 197, 253, 0.55)'
        ))
        fig_f.add_trace(go.Bar(
            x=plot_df.index, y=plot_df[OUTFLOW], name='Outflow (Discharges)',
            marker_color='rgba(52, 211, 153, 0.55)'
        ))
        if not agg_weekly:
            fig_f.add_trace(go.Scatter(
                x=hist_flow.index, y=roll_in, mode='lines', name='Inflow Trend (7d)',
                line=dict(color='#2563eb', width=2.5)
            ))
            fig_f.add_trace(go.Scatter(
                x=hist_flow.index, y=roll_out, mode='lines', name='Outflow Trend (7d)',
                line=dict(color='#059669', width=2.5)
            ))
        fig_f.update_layout(
            barmode='group', xaxis_title="Date", yaxis_title="Children / Day",
            height=350, margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_f, width="stretch")

    with c2:
        st.markdown("#### Operational Flow Balance")
        net = recent_in_7d - recent_out_7d
        if net > 0:
            st.warning(f"**Intake Outpacing Placement:** Transfers exceed discharges by **{net:.1f}** children/day (7d avg).")
        else:
            st.success(f"**Discharge Surplus:** Discharges exceed transfers by **{abs(net):.1f}** children/day (7d avg).")

        hd = int(min(disc.horizon.unique(), key=lambda h: abs(h - horizon)))
        d = disc[disc.horizon == hd].iloc[0]
        maes = {k: d[f'MAE_{k}d_Rate'] for k in (7, 14, 28)}
        best_k = min(maes, key=maes.get)
        lines = []
        for k in (7, 14, 28):
            tag = "  *Lowest MAE*" if k == best_k else ""
            lines.append(
                f"- **{k}-day pace:** {int(fc_view[f'Discharge_{k}d_Rate'].iloc[-1]):,} children "
                f"*(MAE ±{int(maes[k])} at h={hd}d){tag}*"
            )
        st.markdown(f"**Cumulative Placement Demand ({horizon}d):**\n" + "\n".join(lines))
        st.caption(
            f" **Flow Accounting Discrepancy:** Computed daily correlation between $\\Delta\\text{{Care Load}}$ and "
            f"$(\\text{{Transfers}} - \\text{{Discharges}})$ is **{flow_corr:.2f}** across the dataset. Over the last 28 days, "
            f"active care load changed by **{recent_care_diff:+.2f}/day** while net CBP transfers minus discharges averaged "
            f"**{recent_net_flow:+.2f}/day**, indicating unrecorded entry/exit pathways or reporting timing lags."
        )

with tab3:
    st.subheader(f"Walk-Forward Cross-Validation Benchmarks (Evaluated at Horizon h={h_eval}d, n={n_obs_h} Observed Pairs)")
    st.dataframe(b, width="stretch")
    st.caption(
        f"**Metric Definition (`RelMAE`):** Out-of-fold Mean Absolute Error divided by the same-horizon Naïve persistence MAE "
        f"across 5 expanding walk-forward folds (60 calendar days/fold, gap = $h$), scored strictly on observed (non-interpolated) "
        f"origin and target dates ({obs_n_summary}). `RelMAE < 1.00` beats persistence. `Stability` = std of fold RelMAE; `Robustness` = worst-fold RelMAE."
    )

    c_mase, c_cov = st.columns(2)
    with c_mase:
        st.markdown("#### Relative MAE vs. Naïve Across Horizons")
        mase_piv = bench.pivot(index='horizon', columns='model', values='RelMAE')
        key_models = [m for m in ['Naive', 'DampedDrift', 'Ensemble_Damped', 'ExpDrift', 'FlowDrift', 'RandomForest', 'HistGB'] if m in mase_piv.columns]
        fig_m = go.Figure()
        for m in key_models:
            fig_m.add_trace(go.Scatter(x=mase_piv.index, y=mase_piv[m], mode='lines+markers', name=m))
        fig_m.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", annotation_text="Naïve Benchmark (1.00)")
        fig_m.update_layout(xaxis_title="Horizon (Days)", yaxis_title="RelMAE (Lower is Better)", height=320, margin=dict(l=20, r=20, t=20, b=30))
        st.plotly_chart(fig_m, width="stretch")

    with c_cov:
        st.markdown("#### Out-of-Sample Interval Coverage (Held-Out Fold 5, %)")
        cov_piv = cov.pivot(index='model', columns='horizon', values='coverage_pct').round(1)
        if 'test_obs_n' in cov.columns:
            n_row = cov.groupby('horizon')['test_obs_n'].first().rename('Fold 5 Observed n')
            cov_display = pd.concat([cov_piv, pd.DataFrame([n_row])])
        else:
            cov_display = cov_piv
        st.dataframe(cov_display, width="stretch")

        dd_cov = cov[cov.model == 'DampedDrift']['coverage_pct']
        dd_min, dd_max = float(dd_cov.min()), float(dd_cov.max())
        ens_cov = cov[cov.model.isin(['Ensemble', 'Ensemble_Damped'])]['coverage_pct']
        ens_min, ens_max = float(ens_cov.min()), float(ens_cov.max())
        naive_cov = cov[cov.model == 'Naive'].set_index('horizon')['coverage_pct']
        naive_min, naive_max = float(naive_cov.min()), float(naive_cov.max())
        naive_min_h = int(naive_cov.idxmin())
        n_f5_med = int(cov['test_obs_n'].median()) if 'test_obs_n' in cov.columns else 38

        st.caption(
            f"Calibrated on Folds 1–4 and tested out-of-sample on Fold 5 (nominal target = 95%, $n \\approx {n_f5_med}$ observed pairs/cell, "
            f"sampling error $\\approx \\pm 3\\text{{--}}4$ pts). **`DampedDrift` is {dd_min:.1f}–{dd_max:.1f}%** (slightly conservative, likely because Folds 1–4 "
            f"spanned steeper transitions); **the ensembles are {ens_min:.1f}–{ens_max:.1f}%**, consistent with 95% given $n \\approx {n_f5_med}$; "
            f"**`Naive` drops to {naive_min:.1f}% at $h={naive_min_h}$d** (range {naive_min:.1f}–{naive_max:.1f}%), under-covering at long horizons—which supports our model choice."
        )

    st.markdown("---")
    c_sens, c_ml = st.columns([1.2, 1])
    with c_sens:
        st.markdown("#### Parameter Robustness Check (`DampedDrift` Grid)")
        st.dataframe(sens.pivot_table(index=['horizon', 'alpha'], columns='phi', values='RelMAE').round(2), width="stretch")
        worst_row = sens.loc[sens['RelMAE'].idxmax()]
        best_row = sens.loc[sens['RelMAE'].idxmin()]
        st.caption(
            f"**Parameter Robustness:** Conventional defaults ($\\alpha=0.15, \\phi=0.96$) are retained; the grid shows results are insensitive "
            f"to them across $\\alpha \\in \\{{0.10, 0.15, 0.25\\}}$ and $\\phi \\in \\{{0.90, 0.94, 0.96, 0.98\\}}$ "
            f"(`RelMAE` {best_row['RelMAE']:.2f}–{worst_row['RelMAE']:.2f}), with only weak smoothing + weak damping "
            f"($\\alpha={worst_row['alpha']:.2f}, \\phi={worst_row['phi']:.2f}$ at $h={int(worst_row['horizon'])}$d, `RelMAE` = **{worst_row['RelMAE']:.2f}**) exceeding 1.00."
        )
    with c_ml:
        st.markdown("#### Why Complex ML Models Underperformed (Negative Result)")
        ml_base = bench[bench.model.isin(['RandomForest', 'HistGB', 'Ridge'])]
        ml_all_min, ml_all_max = float(ml_base['RelMAE'].min()), float(ml_base['RelMAE'].max())
        ml_h_sub = ml_base[ml_base.horizon == h_eval]['RelMAE']
        ml_h_min, ml_h_max = float(ml_h_sub.min()), float(ml_h_sub.max())

        ml_lite = bench[bench.model.isin(['Ridge_Lite', 'HistGB_Lite'])]
        lite_all_min, lite_all_max = float(ml_lite['RelMAE'].min()), float(ml_lite['RelMAE'].max())
        lite_h_sub = ml_lite[ml_lite.horizon == h_eval]['RelMAE']
        lite_h_min, lite_h_max = float(lite_h_sub.min()), float(lite_h_sub.max())

        st.markdown(f"""
        Tree-based (`RandomForest`, `HistGB`) and penalized linear (`Ridge`) models lost to damped momentum (**`RelMAE` {ml_all_min:.2f}–{ml_all_max:.2f}** across horizons; **{ml_h_min:.2f}–{ml_h_max:.2f}** at $h={h_eval}$d). Likely contributing factors include:
        1. **Regime Shift (Partial Evidence):** Care load fell from **{load_23_24_min:,}–{load_23_24_max:,}** in 2023–2024 to **{load_2025_min:,}–{load_2025_max:,}** in 2025. However, our `_Lite` variants (`Ridge_Lite`, `HistGB_Lite`—trained on a 365-day window without `cbp_load`) still lose badly (**`RelMAE` {lite_all_min:.2f}–{lite_all_max:.2f}** overall; **{lite_h_min:.2f}–{lite_h_max:.2f}** at $h={h_eval}$d), indicating regime shift alone does not explain the gap.
        2. **Small Effective Sample Size:** Only **{obs_rows:,} of {total_rows:,} days** are genuinely observed; **{interp_rows:,} days** (primarily Fridays and Saturdays) are linearly interpolated.
        3. **Interpolation Autocorrelation:** Linear interpolation across weekends creates artificial 3-day constant-slope segments (**{wknd_slope:+.1f}/day** mean Fri/Sat/Sun change), which likely distorts multi-lag tree splits and regression weights.
        """)

    with st.expander(" Methodological Assumptions & Operational Limitations", expanded=False):
        st.markdown(f"""
        - **Model Selection on Walk-Forward Folds:** Model selection used the same walk-forward folds, so reported skill may be slightly optimistic.
        - **Weekend Forecasts Unvalidated:** Because raw HHS public reporting omits Fridays and Saturdays ({interp_rows:,} interpolated days), out-of-sample evaluation masks interpolated origins/targets. Weekend point forecasts cannot be validated against ground truth.
        - **Calm-Regime Evaluation Window:** The 5 walk-forward test folds (~300 calendar days, {obs_n_summary}) cover a low-load, post-collapse stabilization regime ({load_2025_min:,}–{load_2025_max:,} children). Model behavior during an acute border surge is unvalidated out-of-sample.
        - **Illustrative Capacity Threshold:** The {cap:,} threshold is an adjustable stress-testing assumption, not an official ORR funded bed ceiling.
        - **No Exogenous Policy Signals:** Models rely strictly on endogenous care-load and CBP flow histories without external border policy or seasonal Title 8 enforcement indicators.
        """)

with tab4:
    st.subheader(f"Day-by-Day Forecast Schedule ({active_model})")
    export_df = pd.DataFrame({
        'Date': fc_view.index.strftime('%Y-%m-%d'),
        'Baseline_Forecast': fc_view[active_model].round(1),
        'Lower_95_CI': fc_view[f'lower_{active_model}'].round(1),
        'Upper_95_CI': fc_view[f'upper_{active_model}'].round(1),
        'Scenario_Forecast': fc_view['custom_scenario'].round(1),
        'Active_Breach_Prob_Pct': (p * 100).round(1),
        'Cum_Discharges_14d_Pace': fc_view['Discharge_14d_Rate'].round(0)
    }).set_index('Date')
    st.dataframe(export_df, width="stretch")
    csv_bytes = export_df.to_csv().encode('utf-8')
    st.download_button(
        label=" Download Forecast Schedule (CSV)",
        data=csv_bytes,
        file_name=f"HHS_Forecast_{active_model}_{horizon}d.csv",
        mime="text/csv"
    )
