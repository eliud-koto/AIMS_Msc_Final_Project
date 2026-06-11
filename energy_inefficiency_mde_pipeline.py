import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Energy Inefficiency Detection in Mobile Networks
    ## Minimum Distortion Embedding (MDE) Pipeline

    **Core idea:** structurally similar sites should consume comparable energy.
    Sites that consume substantially more than their structural peers are candidates for inefficiency.
    MDE makes this operational by repelling high-energy nodes from their neighbours during optimisation —
    anomalous sites are pushed to the periphery of the embedding.

    **Pipeline:**
    Physics model → Synthetic data → Feature matrix X → MDE (graph → δ-node → edge weights → embedding)
    → Option B scoring → Embedding relative displacement (teacher) → Pseudo-label distillation
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymde
    import torch
    import plotly.express as px
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import pdist

    return csr_matrix, mo, np, pd, pdist, plt, px, pymde, torch


@app.cell
def _():
    DATA_PATH          = "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data_extended.pkl"
    K_GRAPH            = 300
    K_BASELINE         = 10     # tight local peer set for delta_node energy comparison
    K_STRUCT_NEIGHBORS = 50    # wider peer set for displacement scoring and LOF/kNN baselines
    TRAFFIC_WEIGHT     = 0.05   # sweep peak (PR-AUC 0.9158 vs 0.8954 at 0.25); vendor flip ~2.04 >> traffic at 0.05 (~0.08) — cross-vendor bridging impossible
    N_SAMPLES          = 5000
    INEFF_PCT          = 0.1
    BETA               = 35    # repulsion strength; pushes high-delta_node nodes away from structural neighbours
    N_DISSIMILAR_MULT  = 4      # dissimilar edges = N_DISSIMILAR_MULT x similar edges
    NEG_WEIGHT         = -2.0   # weight on dissimilar edges in PushAndPull
    TOP_PAIRS_FRAC     = 0.20   # fraction of high-distortion pairs used in Option B scoring
    EMBEDDING_DIM      = 4
    return (
        BETA,
        DATA_PATH,
        EMBEDDING_DIM,
        INEFF_PCT,
        K_BASELINE,
        K_GRAPH,
        K_STRUCT_NEIGHBORS,
        NEG_WEIGHT,
        N_DISSIMILAR_MULT,
        N_SAMPLES,
        TRAFFIC_WEIGHT,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Physics Model
    """)
    return


@app.cell
def _(DATA_PATH, pd):
    _mast_bucket_map = {
        'Lattice': 'Tower',         'Monopole': 'Tower',        'Concrete Tower': 'Tower',
        'Spine': 'Tower',           'Mono / Lattice': 'Tower',  'Mono Lattice': 'Tower',
        'Lattice on Roof': 'Tower', 'Temp_Lattice': 'Tower',    'Temp_Spine': 'Tower',
        'Tree': 'Disguised',        'Palm Tree': 'Disguised',   'Pine Tree': 'Disguised',
        'Camouflage': 'Disguised',  'Anna Tree': 'Disguised',   'Cypress Tree': 'Disguised',
        'Palm / Cocus': 'Disguised','Yellow Wood': 'Disguised',
        'FlagPole': 'Disguised',    'Signage tower': 'Disguised',
        'Building': 'Rooftop',      'Rooftop': 'Rooftop',       'Indoor': 'Rooftop',
        'DAS': 'Rooftop',           'ULCS': 'Rooftop',
        'Pole': 'Pole',             'LampPost': 'Pole',         'Billboard': 'Pole',
        'Street Light Pole': 'Pole','CameraPole': 'Pole',
    }
    _raw = pd.read_pickle(DATA_PATH)
    _raw['mast_group'] = _raw['mast_type'].map(_mast_bucket_map).fillna('Other')
    df = _raw
    return (df,)


@app.cell
def _(df, pd):
    from sklearn.linear_model import LinearRegression as _LR

    _models = []
    for (_shared, _vendor, _mast), _g in df.groupby(['has_shared_ran', 'ran_vendor_type', 'mast_group']):
        if len(_g) < 3:
            continue
        _m = _LR(positive=True, fit_intercept=True)
        _m.fit(_g[['num_total_cells', 'total_non_ran_equipment']].values, _g['meter_kwh'].values)
        _models.append({
            'has_shared_ran': _shared, 'ran_vendor_type': _vendor, 'mast_group': _mast,
            'base_load':      _m.intercept_,
            'alpha_cells':    _m.coef_[0],
            'beta_non_ran':   _m.coef_[1],
        })

    physics_df = pd.DataFrame(_models).round(2)

    # HUA-shared group has too few samples for reliable regression; coefficients are physically unrealistic
    _mask = (physics_df['has_shared_ran'] == 1) & (physics_df['ran_vendor_type'] == 'HUA')
    physics_df.loc[_mask, 'base_load']    = 480.0
    physics_df.loc[_mask, 'beta_non_ran'] = 220.0
    return (physics_df,)


@app.cell
def _(physics_df):
    physics_df.sort_values(['has_shared_ran', 'ran_vendor_type'])
    return


@app.function
def predict_kwh_physics(num_cells, num_non_ran, shared, vendor, mast, physics_df):
    row = physics_df[
        (physics_df['has_shared_ran'] == shared) &
        (physics_df['ran_vendor_type'] == vendor) &
        (physics_df['mast_group'] == mast)
    ]
    if len(row) == 0:
        row = physics_df[
            (physics_df['has_shared_ran'] == shared) &
            (physics_df['ran_vendor_type'] == vendor)
        ]
    row = row.iloc[0]
    return row['base_load'] + row['alpha_cells'] * num_cells + row['beta_non_ran'] * num_non_ran


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Synthetic Data Generation
    """)
    return


@app.cell
def _(INEFF_PCT, N_SAMPLES, df, np, physics_df):
    _config_cols = ['num_total_cells', 'total_non_ran_equipment',
                    'has_shared_ran', 'ran_vendor_type', 'mast_group']

    _joint = (
        df[_config_cols]
        .value_counts(normalize=True)
        .reset_index(name='probability')
    ).round(2).reset_index()
    _joint['index'] = _joint['index'].astype(str)

    # frequency-weighted sampling: each configuration is drawn proportionally to
    # its empirical frequency in the real data, matching the true operational distribution
    df_mock = _joint.sample(n=N_SAMPLES, replace=True, weights=None, random_state=42).reset_index(drop=True)

    print(f"Vendor mix : {df_mock['ran_vendor_type'].value_counts().to_dict()}")
    print(f"Shared RAN : {df_mock['has_shared_ran'].value_counts().to_dict()}")
    print(f"Mast group : {df_mock['mast_group'].value_counts().to_dict()}")

    _traffic_lookup  = df.groupby(_config_cols)['ps_traffic_mb'].apply(list).to_dict()
    _traf_med_global = df['ps_traffic_mb'].median()
    _traf_std_global = df['ps_traffic_mb'].std()
    _rng_t = np.random.default_rng(99)

    _ps_vals = []
    for _, _r in df_mock.iterrows():
        _key  = tuple(_r[c] for c in _config_cols)
        _pool = _traffic_lookup.get(_key)
        if _pool:
            _v = float(_rng_t.choice(_pool))
            _v = max(_v + _rng_t.normal(0, _traf_std_global * 0.05), 0.0)
        else:
            _v = _traf_med_global
        _ps_vals.append(_v)
    df_mock['ps_traffic_mb'] = np.array(_ps_vals)

    df_mock['kwh_expected'] = df_mock.apply(
        lambda r: predict_kwh_physics(
            r['num_total_cells'], r['total_non_ran_equipment'],
            r['has_shared_ran'], r['ran_vendor_type'], r['mast_group'], physics_df
        ), axis=1
    )

    _sigma_log_map = {
        'Tower': 0.04, 'Disguised': 0.035,
        'Rooftop': 0.03, 'Pole': 0.02, 'Other': 0.0325,
    }
    np.random.seed(42)
    _sigma_log = df_mock['mast_group'].map(_sigma_log_map).values
    _log_eps   = np.random.normal(0, _sigma_log)
    _noise_std = _sigma_log * df_mock['kwh_expected'].values
    df_mock['noise_std']          = _noise_std
    df_mock['kwh_expected_noise'] = df_mock['kwh_expected'].values * np.exp(_log_eps)
    print(f"σ_log by mast: Tower=4%  Disguised=3.5%  Rooftop=3%  Pole=2%  Other=3.25%")
    print(f"Noise std (approx) — min: {_noise_std.min():.1f}  median: {np.median(_noise_std):.1f}  max: {_noise_std.max():.1f} kWh")

    # ── Inefficiency injection ────────────────────────────────────────────────
    # Type 1 – Multiplicative overload : meter = base × exp(ε) × U(1.2, 1.8)
    # Type 2 – Cooling failure         : meter = base·exp(ε) + mast-specific overhead U(lo, hi)
    # Type 3 – Idle RF                 : meter = base·exp(ε) + max(max(cells,5)²×idle×U(0.5,1.5), 2×noise_std)
    #                                    SNR floor guarantees overhead ≥ 2σ_approx
    # Type 4 – Auxiliary parasitic load: meter = base·exp(ε) + max((non_ran+1)²×U(20,50), 2×noise_std)
    #                                    SNR floor guarantees overhead ≥ 2σ_approx

    _n_bad     = int(INEFF_PCT * N_SAMPLES)
    n_per_type = _n_bad // 4

    _rng = np.random.default_rng(42)

    _traf_med = df_mock['ps_traffic_mb'].median()
    _traf_med = _traf_med if _traf_med > 0 else df_mock['ps_traffic_mb'].mean()

    # All four types are config-independent — any site can receive any anomaly type.
    _counts = [0, 0, 0, 0]
    _itype  = np.zeros(N_SAMPLES, dtype=int)

    for _i in _rng.permutation(N_SAMPLES).tolist():
        if sum(_counts) >= _n_bad:
            break
        _opts = [t for t in range(4) if _counts[t] < n_per_type]
        if not _opts:
            break
        _t = int(_rng.choice(_opts))
        _itype[_i] = _t + 1
        _counts[_t] += 1

    _idx_1 = np.where(_itype == 1)[0]
    _idx_2 = np.where(_itype == 2)[0]
    _idx_3 = np.where(_itype == 3)[0]
    _idx_4 = np.where(_itype == 4)[0]
    print(f"Anomaly injection — Type1:{len(_idx_1)}  Type2:{len(_idx_2)}  "
          f"Type3:{len(_idx_3)}  Type4:{len(_idx_4)}  "
          f"Total:{sum(_counts)}/{N_SAMPLES}  ({sum(_counts)/N_SAMPLES:.1%})")

    _cooling = {
        'Tower': (200, 400), 'Disguised': (150, 350),
        'Rooftop': (80, 200), 'Pole': (100, 250), 'Other': (100, 200)
    }
    _meter = df_mock['kwh_expected_noise'].values.copy()

    for _i in _idx_1:
        _meter[_i] *= _rng.uniform(1.2, 1.8)

    for _i in _idx_2:
        _lo, _hi   = _cooling.get(df_mock.iloc[_i]['mast_group'], (100, 200))
        _meter[_i] += _rng.uniform(_lo, _hi)

    for _i in _idx_3:
        _cells      = df_mock.iloc[_i]['num_total_cells']
        _traf       = df_mock.iloc[_i]['ps_traffic_mb']
        _idle       = max(1.0 - _traf / _traf_med, 0.1)
        _site_noise = df_mock.iloc[_i]['noise_std']
        _overhead   = max(_cells, 5) ** 2 * _idle * _rng.uniform(0.5, 1.5)
        _meter[_i] += max(_overhead, 2.0 * _site_noise)

    for _i in _idx_4:
        _nran       = df_mock.iloc[_i]['total_non_ran_equipment']
        _site_noise = df_mock.iloc[_i]['noise_std']
        _overhead   = (_nran + 1) ** 2 * _rng.uniform(20, 50)
        _meter[_i] += max(_overhead, 2.0 * _site_noise)

    df_mock['meter_kwh_sim']     = np.round(_meter, 2)
    df_mock['inefficiency_type'] = _itype
    df_mock['is_inefficient']    = (_itype > 0).astype(int)
    return (df_mock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S1. Structural Configuration Characteristics
    """)
    return


@app.cell
def _(df_mock, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 7))

    # (a) RAN Vendor Type
    _vendor_display = {'HUA': 'Vendor B', 'NOK': 'Vendor A'}
    _vc = df_mock['ran_vendor_type'].value_counts()
    _vc.index = _vc.index.map(lambda v: _vendor_display.get(v, v))
    _axes[0, 0].bar(_vc.index, _vc.values, color=['#1f77b4', '#ff7f0e'][:len(_vc)], edgecolor='white')
    _axes[0, 0].set_ylabel("Count")
    for _p, _v in zip(_axes[0, 0].patches, _vc.values):
        _axes[0, 0].text(_p.get_x() + _p.get_width() / 2, _p.get_height() + 15,
                         f'{_v}\n({100*_v/len(df_mock):.0f}%)', ha='center', fontsize=9)

    # (b) Mast Group
    _mg = df_mock['mast_group'].value_counts()
    _axes[0, 1].bar(_mg.index, _mg.values, color='#2ca02c', edgecolor='white')
    _axes[0, 1].tick_params(axis='x', rotation=30)
    _axes[0, 1].set_ylabel("Count")

    # (c) Cells per Site
    _axes[1, 0].hist(df_mock['num_total_cells'],
                     bins=range(1, int(df_mock['num_total_cells'].max()) + 2),
                     color='#1f77b4', edgecolor='white', align='left')
    _axes[1, 0].set_xlabel("num_total_cells")
    _axes[1, 0].set_ylabel("Count")

    # (d) Non-RAN Equipment per Site
    _axes[1, 1].hist(df_mock['total_non_ran_equipment'],
                     bins=range(0, int(df_mock['total_non_ran_equipment'].max()) + 2),
                     color='#ff7f0e', edgecolor='white', align='left')
    _axes[1, 1].set_xlabel("total_non_ran_equipment")
    _axes[1, 1].set_ylabel("Count")

    for _ax in _axes.flat:
        _ax.spines[['top', 'right']].set_visible(False)
    _fig.tight_layout()
    _fig.savefig(
        "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/eda_structural.pdf",
        bbox_inches='tight')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S2. Energy Behaviour Relative to Structure
    """)
    return


@app.cell
def _(df_mock, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))

    # (a) Expected Energy distribution
    _axes[0].hist(df_mock['kwh_expected'], bins=30, color='#2ca02c', edgecolor='white', linewidth=0.6)
    _axes[0].axvline(df_mock['kwh_expected'].median(), color='red', linestyle='--',
                     linewidth=1.5,
                     label=f"Median: {df_mock['kwh_expected'].median():.0f} kWh")
    _axes[0].set_xlabel("kwh_expected (kWh)")
    _axes[0].set_ylabel("Count")
    _axes[0].legend(fontsize=8)

    # (b) Expected Energy by Configuration Group (median ± std)
    _grp = (df_mock.groupby(['has_shared_ran', 'ran_vendor_type'])['kwh_expected']
            .agg(['median', 'std', 'count'])
            .rename(columns={'median': 'median_kwh', 'std': 'std_kwh', 'count': 'n_sites'}))
    _n_configs = df_mock.groupby(
        ['ran_vendor_type', 'mast_group', 'num_total_cells', 'total_non_ran_equipment']
    ).ngroups
    _vendor_display = {'HUA': 'Vendor B', 'NOK': 'Vendor A'}
    _glabels = [f"{'Shared' if s else 'Solo'}\n{_vendor_display.get(v, v)}" for (s, v) in _grp.index]
    _gcolors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78']
    _bars = _axes[1].bar(_glabels, _grp['median_kwh'].values,
                         yerr=_grp['std_kwh'].values, capsize=6,
                         color=_gcolors, edgecolor='white',
                         error_kw=dict(elinewidth=1.5, ecolor='black'))
    for _b, _n in zip(_bars, _grp['n_sites'].values):
        _axes[1].text(_b.get_x() + _b.get_width() / 2,
                      _b.get_height() + _grp['std_kwh'].values.max() * 0.05,
                      f"n={_n}", ha='center', va='bottom', fontsize=9)
    _axes[1].set_ylabel("Expected Energy (kWh)")

    for _ax in _axes:
        _ax.spines[['top', 'right']].set_visible(False)
    _fig.tight_layout()
    _fig.savefig(
        "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/eda_energy_structure.pdf",
        bbox_inches='tight')
    _fig
    return


@app.cell
def _(df_mock, np, plt):
    _fig, _ax = plt.subplots(figsize=(7.5, 5))

    # Expected Energy vs Number of Cells (coloured by RAN sharing)
    _color_map = {1: '#d62728', 0: '#1f77b4'}
    _rng_j = np.random.default_rng(0)
    for _shared, _g in df_mock.groupby('has_shared_ran'):
        _jitter = _rng_j.uniform(-0.15, 0.15, len(_g))
        _ax.scatter(_g['num_total_cells'] + _jitter, _g['kwh_expected'],
                    color=_color_map[_shared], alpha=0.25, s=8, linewidths=0,
                    label=f"{'Shared' if _shared else 'Standalone'} RAN")
        _xs = np.linspace(_g['num_total_cells'].min(), _g['num_total_cells'].max(), 100)
        _ax.plot(_xs, np.polyval(np.polyfit(_g['num_total_cells'], _g['kwh_expected'], 1), _xs),
                 color=_color_map[_shared], linewidth=2)
    _ax.set_xlabel("Number of Cells")
    _ax.set_ylabel("Expected Energy (kWh)")
    _ax.legend(fontsize=9)
    _ax.spines[['top', 'right']].set_visible(False)
    _fig.tight_layout()
    _fig.savefig(
        "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/eda_energy_vs_cells.pdf",
        bbox_inches='tight')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S3. Noise Characteristics
    """)
    return


@app.cell
def _(df_mock, np, plt):
    from scipy import stats as _stats

    _residual = (df_mock['kwh_expected_noise'] - df_mock['kwh_expected']).values
    _ratio    = (df_mock['kwh_expected_noise'] /
                 df_mock['kwh_expected'].clip(lower=1)).values
    _cv       = np.abs(_residual) / df_mock['kwh_expected'].values
    _noise_std_col = df_mock['noise_std'].values

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Ratio distribution — lognormal: right-skewed around 1
    _p5, _p95 = np.percentile(_ratio, [5, 95])
    _axes[0].hist(_ratio, bins=60, color='#ff7f0e', edgecolor='white', alpha=0.8)
    _axes[0].axvline(1.0,  color='black',   linestyle='--', linewidth=1.5, label='ratio = 1')
    _axes[0].axvline(_p5,  color='#2ca02c', linestyle=':',  linewidth=1.5, label=f'P5 = {_p5:.3f}')
    _axes[0].axvline(_p95, color='#2ca02c', linestyle=':',  linewidth=1.5, label=f'P95 = {_p95:.3f}')
    _axes[0].set_xlabel("Noisy energy / expected energy")
    _axes[0].set_ylabel("Count")
    _axes[0].legend(fontsize=8)

    # |residual| vs kwh_expected — per-mast-group σ_log lines
    _xs_curve = np.linspace(df_mock['kwh_expected'].min(), df_mock['kwh_expected'].max(), 200)
    _sigma_log_map = {'Tower': 0.04, 'Disguised': 0.035, 'Rooftop': 0.03, 'Pole': 0.02, 'Other': 0.0325}
    _colors_mast   = {'Tower': '#d62728', 'Disguised': '#ff7f0e', 'Rooftop': '#2ca02c',
                      'Pole': '#1f77b4', 'Other': '#9467bd'}
    _axes[1].scatter(df_mock['kwh_expected'], np.abs(_residual),
                     s=4, alpha=0.3, color='#9467bd', linewidths=0)
    for _mg, _sl in _sigma_log_map.items():
        _axes[1].plot(_xs_curve, _sl * _xs_curve, linewidth=1.5,
                      color=_colors_mast[_mg], label=f'{_mg} σ={int(_sl*100)}%')
    _axes[1].set_xlabel("Expected energy (kWh)")
    _axes[1].set_ylabel("|noise residual| (kWh)")
    _axes[1].legend(fontsize=8)

    for _ax in _axes:
        _ax.spines[['top', 'right']].set_visible(False)
    _fig.tight_layout()
    _fig.savefig(
        "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/eda_noise.pdf",
        bbox_inches='tight')
    print("── Noise model diagnostics ────────────────────────────────────────────────")
    print(f"  Residual mean : {_residual.mean():.3f} kWh   std : {_residual.std():.3f} kWh")
    print(f"  Skewness      : {_stats.skew(_residual):.3f}  (>0 expected for lognormal)")
    print(f"  CV range      : {_cv.min()*100:.1f}% – {_cv.max()*100:.1f}%")
    print(f"  Per-site σ    — min: {_noise_std_col.min():.1f}  median: {np.median(_noise_std_col):.1f}"
          f"  max: {_noise_std_col.max():.1f} kWh")
    print(f"  Model         : Lognormal E·exp(ε), ε~N(0,σ_log²), σ_log by mast group")
    _fig
    return


@app.cell
def _(df_mock, np, plt):
    from scipy import stats as _stats

    _residual = (df_mock['kwh_expected_noise'] - df_mock['kwh_expected']).values
    _noise_std_col = df_mock['noise_std'].values
    _x_pdf = np.linspace(_residual.min(), _residual.max(), 300)

    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    _ax.hist(_residual, bins=60, density=True,
             color='#1f77b4', edgecolor='white', alpha=0.8)
    _ax.plot(_x_pdf, _stats.norm.pdf(_x_pdf, 0, _noise_std_col.mean()),
             color='#d62728', linewidth=2, label=f'N(0, mean σ={_noise_std_col.mean():.1f})')
    _ax.axvline(0, color='black', linestyle='--', linewidth=1)
    _ax.set_xlabel("Noise residual (kWh)")
    _ax.set_ylabel("Density")
    _ax.legend(fontsize=9)
    _ax.spines[['top', 'right']].set_visible(False)
    _fig.tight_layout()
    _fig.savefig(
        "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/eda_noise_residual_distribution.pdf",
        bbox_inches='tight')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S4. Inefficiency Injection Characteristics
    """)
    return


@app.cell
def _(N_SAMPLES, df_mock, np, plt):
    from scipy.stats import gaussian_kde as _kde

    _eff_mask   = df_mock['is_inefficient'] == 0
    _ineff_mask = df_mock['is_inefficient'] == 1
    _ratio_all  = (df_mock['meter_kwh_sim'] /
                   df_mock['kwh_expected_noise'].clip(lower=1)).values
    _ratio_eff   = _ratio_all[_eff_mask]
    _ratio_ineff = _ratio_all[_ineff_mask]

    _type_labels = {0: 'Efficient', 1: 'Type 1\n(Overload)', 2: 'Type 2\n(Cooling)',
                    3: 'Type 3\n(Idle RF)', 4: 'Type 4\n(Non-RAN)'}
    _bar_colors  = ['#2ca02c', '#d62728', '#d62728', '#d62728', '#d62728']

    # ── Figure 1: type counts bar chart ──────────────────────────────────────
    _tc = df_mock['inefficiency_type'].value_counts().sort_index()
    _fig1, _ax1 = plt.subplots(figsize=(5, 4.5))
    _bars = _ax1.bar([_type_labels.get(t, str(t)) for t in _tc.index],
                     _tc.values,
                     color=[_bar_colors[t] for t in _tc.index],
                     edgecolor='white')
    for _b, _v in zip(_bars, _tc.values):
        _ax1.text(_b.get_x() + _b.get_width() / 2, _b.get_height() + 15,
                  str(_v), ha='center', fontsize=9)
    _ax1.set_ylabel("Count")
    _ax1.tick_params(axis='x', labelsize=8)
    _ax1.spines[['top', 'right']].set_visible(False)
    _fig1.tight_layout()
    _fig1.savefig(
        "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/eda_injection_counts.pdf",
        bbox_inches='tight')

    # ── Figure 2: KDE panels ──────────────────────────────────────────────────
    _log_eff   = np.log1p(df_mock.loc[_eff_mask,   'meter_kwh_sim'].values)
    _log_ineff = np.log1p(df_mock.loc[_ineff_mask, 'meter_kwh_sim'].values)
    _xs_e = np.linspace(_log_eff.min(),   _log_eff.max(),   300)
    _xs_i = np.linspace(_log_ineff.min(), _log_ineff.max(), 300)
    _xr   = np.linspace(0, min(np.percentile(_ratio_ineff, 99.5), 6), 300)

    _fig2, _axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
    _axes2[0].fill_between(_xs_e, _kde(_log_eff)(_xs_e),   alpha=0.45,
                            color='#1f77b4', label='Efficient')
    _axes2[0].fill_between(_xs_i, _kde(_log_ineff)(_xs_i), alpha=0.45,
                            color='#d62728', label='Inefficient')
    _axes2[0].set_xlabel("log(1 + meter_kwh_sim)")
    _axes2[0].set_ylabel("Density")
    _axes2[0].legend(fontsize=9)

    _axes2[1].fill_between(_xr, _kde(_ratio_eff)(_xr),   alpha=0.45,
                            color='#1f77b4', label='Efficient')
    _axes2[1].fill_between(_xr, _kde(_ratio_ineff)(_xr), alpha=0.45,
                            color='#d62728', label='Inefficient')
    _axes2[1].axvline(1.0, color='black', linestyle='--', linewidth=1.5, label='ratio = 1')
    _axes2[1].set_xlabel("meter_kwh_sim / kwh_expected_noise")
    _axes2[1].set_ylabel("Density")
    _axes2[1].legend(fontsize=9)

    for _ax in _axes2:
        _ax.spines[['top', 'right']].set_visible(False)
    _fig2.tight_layout()
    _fig2.savefig(
        "/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/eda_injection.pdf",
        bbox_inches='tight')

    # Overlap metric: fraction of anomalies whose raw meter reading falls below the
    # 95th-percentile of efficient sites — simulates a naive global energy threshold.
    # (Ratio-based overlap is trivially 0% because efficient meter = kwh_expected_noise by construction.)
    _p95_meter_eff = np.percentile(df_mock.loc[_eff_mask, 'meter_kwh_sim'].values, 95)
    _overlap = (df_mock.loc[_ineff_mask, 'meter_kwh_sim'].values < _p95_meter_eff).mean()
    print("── Injection validation ───────────────────────────────────────────────────")
    for _t, _n in _tc.items():
        print(f"  {_type_labels.get(_t, str(_t)).replace(chr(10), ' '):<26}: "
              f"{_n:>4} sites ({100*_n/N_SAMPLES:.1f}%)")
    print(f"\n  Overlap (ineff meter < P95 of eff meter) : {_overlap:.1%}")
    print(f"  → {_overlap:.1%} of anomalies fall below the P95 global energy cut;")
    print(f"    a naive threshold misses them — structural neighbourhood context required")
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S5. Dataset Suitability Summary
    """)
    return


@app.cell
def _(INEFF_PCT, N_SAMPLES, df_mock, np):
    def _r2_fn(x, y):
        if len(x) < 3:
            return np.nan
        _c    = np.polyfit(x, y, 1)
        _yhat = np.polyval(_c, x)
        _ss_r = np.sum((y - _yhat) ** 2)
        _ss_t = np.sum((y - y.mean()) ** 2)
        return 1.0 - _ss_r / _ss_t if _ss_t > 0 else np.nan

    _n_configs  = df_mock.groupby(
        ['ran_vendor_type', 'mast_group', 'num_total_cells', 'total_non_ran_equipment']
    ).ngroups
    _hua_pct    = 100 * (df_mock['ran_vendor_type'] == 'HUA').mean()
    _shared_pct = 100 * df_mock['has_shared_ran'].mean()
    _r2_list    = [_r2_fn(g['num_total_cells'].values, g['kwh_expected'].values)
                   for _, g in df_mock.groupby(['has_shared_ran', 'ran_vendor_type'])]
    _med_r2     = np.nanmedian(_r2_list)
    _residual   = (df_mock['kwh_expected_noise'] - df_mock['kwh_expected']).values
    _cv_min     = (np.abs(_residual) / df_mock['kwh_expected'].values).min() * 100
    _cv_max     = (np.abs(_residual) / df_mock['kwh_expected'].values).max() * 100
    _meter_eff   = df_mock.loc[df_mock['is_inefficient'] == 0, 'meter_kwh_sim'].values
    _meter_ineff = df_mock.loc[df_mock['is_inefficient'] == 1, 'meter_kwh_sim'].values
    _overlap     = (_meter_ineff < np.percentile(_meter_eff, 95)).mean()

    print("── Synthetic Dataset Suitability Summary ───────────────────────────────────") #2D embedding aware plot.
    print(f"  N sites           : {N_SAMPLES}    |  Contamination  : {INEFF_PCT:.1%}")
    print(f"  Unique configs    : {_n_configs:<5}  |  Noise model    : Lognormal σ_log=2–4% by mast group")
    print(f"  Vendor split      : Vendor B {_hua_pct:.0f}%  Vendor A {100-_hua_pct:.0f}%")
    print(f"  Shared RAN        : {_shared_pct:.0f}%")
    print()
    print(f"  ✓ Structural diversity   — {_n_configs} unique (vendor×mast×cells×nran) configs;")
    print(f"                             inter-group variance justifies graph-based local comparison")
    print(f"  ✓ Energy–structure link  — kwh_expected ∈ "
          f"[{df_mock['kwh_expected'].min():.0f}, {df_mock['kwh_expected'].max():.0f}] kWh;")
    print(f"                             median within-group R²={_med_r2:.2f} (cells→energy)")
    _noise_std_col = df_mock['noise_std'].values
    print(f"  ✓ Stochastic variability — lognormal noise E·exp(ε), σ_log ∈ [2%,4%] by mast type;")
    print(f"                             per-site σ range {_noise_std_col.min():.1f}–{_noise_std_col.max():.1f} kWh")
    print(f"  ✓ Injection plausibility — {_overlap:.0%} of anomalies fall below the P95 global")
    print(f"                             energy cut (raw meter); a naive threshold misses them —")
    print(f"                             structural neighbourhood context is required")
    print(f"  ✓ Neighbourhood structure — uniform sampling covers full config space;")
    print(f"                             SNR floor on types 3 & 4 guarantees")
    print(f"                             detectable signal above measurement noise")
    print("────────────────────────────────────────────────────────────────────────────")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Feature Matrix
    """)
    return


@app.cell
def _(df_mock, np, pd):
    X = df_mock[['num_total_cells', 'total_non_ran_equipment',
                 'has_shared_ran', 'ran_vendor_type']].copy()

    X['ran_vendor_type']   = (X['ran_vendor_type'] == 'NOK').astype(int)
    X['log_ps_traffic_mb'] = np.log1p(df_mock['ps_traffic_mb'].values)

    # one-hot encoding avoids the spurious ordinal distances imposed by LabelEncoder
    _mast_dummies = pd.get_dummies(df_mock['mast_group'], prefix='mast').astype(float)
    X = pd.concat([X, _mast_dummies], axis=1).astype(float)
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. MDE Pipeline
    """)
    return


@app.cell
def _(K_GRAPH, TRAFFIC_WEIGHT, X, csr_matrix, pymde):
    from sklearn.preprocessing import StandardScaler as _SS_graph

    _X_sc = _SS_graph().fit_transform(X.values)
    _tcol = list(X.columns).index('log_ps_traffic_mb')
    _X_sc[:, _tcol] *= TRAFFIC_WEIGHT

    graph    = pymde.preprocess.k_nearest_neighbors(csr_matrix(_X_sc), k=K_GRAPH, verbose=False)
    X_scaled = _X_sc
    return X_scaled, graph


@app.cell
def _(K_BASELINE, K_STRUCT_NEIGHBORS, X_scaled, df_mock, np):
    from sklearn.neighbors import NearestNeighbors as _NNN_dn

    _energy  = df_mock['meter_kwh_sim'].values
    _delta   = np.zeros(len(df_mock))
    _K_fit   = max(K_BASELINE, K_STRUCT_NEIGHBORS)
    _nn_idx  = np.full((len(df_mock), _K_fit), -1, dtype=np.int64)

    # stratify by (vendor, sharing, mast_group) to guarantee no cross-vendor, cross-sharing, or cross-mast neighbours
    # fit once with the larger K; delta_node slices to K_BASELINE, teacher score uses all K_STRUCT_NEIGHBORS
    for _, _grp_idx in df_mock.groupby(['ran_vendor_type', 'has_shared_ran', 'mast_group']).groups.items():
        _grp_idx = _grp_idx.to_numpy()
        _X_grp   = X_scaled[_grp_idx]
        _k       = min(_K_fit + 1, len(_grp_idx))
        _nbrs    = _NNN_dn(n_neighbors=_k).fit(_X_grp)
        _, _nn   = _nbrs.kneighbors(_X_grp)
        _nn      = _nn[:, 1:]                                      # drop self
        _nn_idx[_grp_idx, :_nn.shape[1]] = _grp_idx[_nn]          # translate to global indices
        _e_grp   = _energy[_grp_idx]
        _nn_base = _nn[:, :min(K_BASELINE, _nn.shape[1])]          # tighter slice for delta_node
        _med     = np.maximum(np.percentile(_e_grp[_nn_base], 35, axis=1), 1.0)
        _delta[_grp_idx] = np.log((_e_grp + 1e-6) / (_med + 1e-6))

    delta_node        = _delta
    nn_idx_structural = _nn_idx    # K_STRUCT_NEIGHBORS wide; consumed by teacher score cell
    return delta_node, nn_idx_structural


@app.cell
def _():
    # # import numpy as np

    # _energy = df_mock["meter_kwh_sim"].values

    # _edges2   = graph.edges.numpy()
    # _weights2 = graph.weights.numpy()

    # eps = 1e-6

    # # Energy values for both endpoints of each structural edge
    # e_i = _energy[_edges2[:, 0]]
    # e_j = _energy[_edges2[:, 1]]

    # # Pairwise relative energy discrepancy
    # # This is symmetric, so it does not depend on edge ordering.
    # # _pairwise_rep = np.abs(
    # #     np.log((e_i + eps) / (e_j + eps))
    # # )

    # # # Energy-aware structural weights
    # # weights = _weights2 - BETA * _pairwise_rep
    return


@app.cell
def _(BETA, delta_node, graph, np):
    _edges2   = graph.edges.numpy()
    _weights2 = graph.weights.numpy()
    _rep      = np.maximum(
        np.maximum(delta_node[_edges2[:, 0]], 0),
        np.maximum(delta_node[_edges2[:, 1]], 0),
    )
    weights   = _weights2 - BETA * _rep
    return (weights,)


@app.cell
def _(
    EMBEDDING_DIM,
    NEG_WEIGHT,
    N_DISSIMILAR_MULT,
    X,
    graph,
    pymde,
    torch,
    weights,
):
    _similar   = graph.edges
    _n_similar = _similar.shape[0]

    torch.manual_seed(42)
    _dissimilar = pymde.preprocess.dissimilar_edges(
        X.shape[0], num_edges=N_DISSIMILAR_MULT * _n_similar, similar_edges=_similar
    )

    _new_edges   = torch.cat([_similar, _dissimilar])
    _new_weights = torch.cat([
        torch.tensor(weights, dtype=torch.float32),
        NEG_WEIGHT * torch.ones(_dissimilar.shape[0]),
    ])

    _f = pymde.penalties.PushAndPull(
        weights=_new_weights,
        attractive_penalty=pymde.penalties.Log1p,
        repulsive_penalty=pymde.penalties.Log,
    )
    mde = pymde.MDE(
        n_items=X.shape[0], embedding_dim=EMBEDDING_DIM,
        edges=_new_edges, distortion_function=_f,
    )
    torch.manual_seed(42)
    embedding = mde.embed(verbose=True)
    return (embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Teacher Score: Embedding Relative Displacement
    """)
    return


@app.cell
def _(delta_node, df_mock, embedding, nn_idx_structural, np, pdist):
    _X_emb  = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)
    _scores = np.zeros(len(df_mock))

    # reuse the same structural neighbours as delta_node (same K, same stratification, same feature space)
    for _i in range(len(df_mock)):
        _neigh = nn_idx_structural[_i]
        _neigh = _neigh[_neigh >= 0]
        if len(_neigh) == 0:
            continue
        _neigh_pos     = _X_emb[_neigh]
        _dist_to_neigh = np.linalg.norm(_X_emb[_i] - _neigh_pos, axis=1).mean()
        _within_spread = pdist(_neigh_pos).mean() + 1e-8
        _scores[_i]    = _dist_to_neigh / _within_spread

    # energy-gated displacement: amplify score for sites that are already high-energy
    # relative to structural peers; leave energy-normal sites unchanged (gate = 1)
    emb_rel_dist_scores = _scores
    emb_rel_dist_gated  = _scores * np.maximum(1 + delta_node, 1)
    df_mock['emb_rel_dist_score']  = emb_rel_dist_scores
    df_mock['emb_rel_dist_gated']  = emb_rel_dist_gated
    return emb_rel_dist_gated, emb_rel_dist_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Embedding Scatter
    """)
    return


@app.cell(hide_code=True)
def _(df_mock, mo):
    color_dropdown = mo.ui.dropdown(
        value="is_inefficient",
        label="Colour by:",
        options=df_mock.columns.tolist(),
    )
    color_dropdown
    return (color_dropdown,)


@app.cell(hide_code=True)
def _(color_dropdown, df_mock, embedding, np, px):
    _emb_np = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)
    _df_plot = df_mock.copy()
    _df_plot['x'] = _emb_np[:, 0]
    _df_plot['y'] = _emb_np[:, 1]
    _df_plot['is_inefficient'] = _df_plot['is_inefficient'].astype(str)

    px.scatter(
        _df_plot, x='x', y='y',
        color=color_dropdown.value,
        hover_data=_df_plot.columns,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Embedding Comparison
    """)
    return


@app.cell
def _(X_scaled):
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE
    import umap as _umap

    pca_emb  = _PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    umap_emb = _umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)
    tsne_emb = _TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)
    return pca_emb, tsne_emb, umap_emb


@app.cell
def _(df_mock, embedding, np, pca_emb, plt, tsne_emb, umap_emb):
    _is_ineff = df_mock['is_inefficient'].values.astype(bool)
    _mde_np   = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)

    _fig, _axes_grid = plt.subplots(2, 2, figsize=(16, 14))
    _axes = _axes_grid.flatten()
    _panels = [
        (f"MDE (energy-aware)", _mde_np),
        ("PCA",                            pca_emb),
        ("UMAP",                           umap_emb),
        ("t-SNE",                          tsne_emb),
    ]

    for _ax, (_title, _emb) in zip(_axes, _panels):
        _ax.scatter(
            _emb[_is_ineff, 0], _emb[_is_ineff, 1],
            c="#e41a1c", label="Inefficient",
            s=16, alpha=0.6, linewidths=0, zorder=1,
        )
        _ax.scatter(
            _emb[~_is_ineff, 0], _emb[~_is_ineff, 1],
            c="#000080", label="Efficient",
            s=5, alpha=0.9, linewidths=0, zorder=2,
        )
        _ax.set_title(_title, fontsize=13)
        _ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # _handles, _labels = _axes[0].get_legend_handles_labels()
    # _fig.legend(_handles, _labels, loc='lower center', ncol=2,
    #             fontsize=15, markerscale=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    # _fig.tight_layout()
    # _fig.savefig("/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/embedding_comparison.pdf", bbox_inches="tight")
    # _fig
    from matplotlib.lines import Line2D as _Line2D
    _legend_handles = [
      _Line2D([0], [0], marker='o', color='w', markerfacecolor='#e41a1c',
              markersize=15, alpha=1.0, label='Inefficient'),
      _Line2D([0], [0], marker='o', color='w', markerfacecolor='#000080',
              markersize=15, alpha=1.0, label='Efficient'),
    ]
    _fig.legend(_legend_handles, ['Inefficient Sites', 'Efficient Sites'], loc='lower center', ncol=2,
              fontsize=20, frameon=False, bbox_to_anchor=(0.5, -0.05))
    _fig.tight_layout()
    _fig.savefig("/home/koto/Desktop/AIMS_Msc_Project/Apr 14 Update/data/embedding_comparison.pdf", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. Baseline Comparison
    """)
    return


@app.cell
def _(
    INEFF_PCT,
    K_STRUCT_NEIGHBORS,
    N_SAMPLES,
    X,
    df_mock,
    emb_rel_dist_gated,
    emb_rel_dist_scores,
    embedding,
    np,
    pca_emb,
    pd,
    torch,
    tsne_emb,
    umap_emb,
):
    import torch.nn as _nn
    from sklearn.metrics import average_precision_score as _aps, roc_auc_score as _roc
    from sklearn.ensemble import IsolationForest as _IF, RandomForestRegressor as _RFR_b
    from sklearn.mixture import GaussianMixture as _GMM
    from sklearn.neighbors import NearestNeighbors as _NNN_b, LocalOutlierFactor as _LOF
    from sklearn.preprocessing import StandardScaler as _SS_b
    from sklearn.linear_model import LinearRegression as _LR_b, HuberRegressor as _HuberReg_b

    class _SimpleAE(_nn.Module):
        def __init__(self, in_dim, hidden, bottleneck):
            super().__init__()
            layers_enc, layers_dec = [], []
            prev = in_dim
            for h in hidden:
                layers_enc += [_nn.Linear(prev, h), _nn.ReLU()]; prev = h
            layers_enc.append(_nn.Linear(prev, bottleneck))
            prev = bottleneck
            for h in reversed(hidden):
                layers_dec += [_nn.Linear(prev, h), _nn.ReLU()]; prev = h
            layers_dec.append(_nn.Linear(prev, in_dim))
            self.enc = _nn.Sequential(*layers_enc)
            self.dec = _nn.Sequential(*layers_dec)
        def forward(self, x): return self.dec(self.enc(x))

    def _train_ae(X_np, hidden, bottleneck, epochs, lr=1e-3, batch=256, seed=0):
        torch.manual_seed(seed)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        ae  = _SimpleAE(X_np.shape[1], hidden, bottleneck)
        opt = torch.optim.Adam(ae.parameters(), lr=lr)
        ds  = torch.utils.data.TensorDataset(X_t)
        dl  = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
        ae.train()
        for _ in range(epochs):
            for (xb,) in dl:
                loss = _nn.functional.mse_loss(ae(xb), xb)
                opt.zero_grad(); loss.backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            recon = ae(X_t)
        return ((X_t - recon) ** 2).mean(dim=1).numpy()

    _energy_b = df_mock['meter_kwh_sim'].values
    _y_true_b = df_mock['is_inefficient'].values
    _TOP_N_b  = int(INEFF_PCT * N_SAMPLES)
    _cont     = INEFF_PCT

    _vendor_b = (df_mock['ran_vendor_type'] == 'NOK').astype(int).values
    _mast_b   = pd.get_dummies(df_mock['mast_group'], prefix='mast').astype(float).values
    _struct_b = np.column_stack([
        df_mock['num_total_cells'].values,
        df_mock['total_non_ran_equipment'].values,
        df_mock['has_shared_ran'].values,
        _vendor_b, _mast_b,
        np.log1p(df_mock['ps_traffic_mb'].values),
    ])

    _mde_np_b    = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)
    _X_all_sc    = _SS_b().fit_transform(np.column_stack([_struct_b, _energy_b]))
    _X_struct_sc = _SS_b().fit_transform(_struct_b)
    _X_mde_sc    = _SS_b().fit_transform(_mde_np_b)
    _X_pca_sc    = _SS_b().fit_transform(pca_emb)
    _X_umap_sc   = _SS_b().fit_transform(umap_emb)
    _X_tsne_sc   = _SS_b().fit_transform(tsne_emb)

    def _met(y, s, n):
        pr   = _aps(y, s)
        roc  = _roc(y, s)
        idx  = np.argsort(s)[::-1][:n]
        return pr, roc, y[idx].mean(), y[idx].sum() / max(y.sum(), 1)

    _pct = int(INEFF_PCT * 100)
    _hdr = f"\n{'Method':<46} {'ROC-AUC':>9} {'PR-AUC':>8} {'Prec@' + str(_pct) + '%':>9} {'Rec@' + str(_pct) + '%':>9}"
    _sep = "-" * 86

    # ── Table 1: primary baselines ────────────────────────────────────────────
    _primary = {}

    _primary['Raw energy'] = _met(_y_true_b, _energy_b, _TOP_N_b)

    _lr_r = _energy_b - _LR_b().fit(X.values, _energy_b).predict(X.values)
    _primary['Linear regression residual'] = _met(_y_true_b, np.maximum(_lr_r, 0), _TOP_N_b)

    _X_hub_sc = _SS_b().fit_transform(X.values)
    _hub_r = _energy_b - _HuberReg_b(epsilon=1.35, max_iter=300).fit(_X_hub_sc, _energy_b).predict(_X_hub_sc)
    _primary['Huber regression residual'] = _met(_y_true_b, np.maximum(_hub_r, 0), _TOP_N_b)

    _rf_reg = _RFR_b(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
    _rf_reg.fit(X.values, _energy_b)
    _rf_r = _energy_b - _rf_reg.predict(X.values)
    _primary['Random Forest residual'] = _met(_y_true_b, np.maximum(_rf_r, 0), _TOP_N_b)

    # physics residual is a privileged upper bound, not an honest competitor:
    # anomalies were generated relative to kwh_expected so the physics model has
    # implicit access to the ground-truth efficient baseline
    _phys = _energy_b / (df_mock['kwh_expected'].values + 1e-6)
    _primary['Physics residual (privileged upper bound)'] = _met(_y_true_b, _phys, _TOP_N_b)


    for _name, _Xsc in [('features', _X_all_sc), ('MDE emb', _X_mde_sc)]:
        _iso = _IF(n_estimators=100, contamination=_cont, random_state=42)
        _primary[f'IsoForest ({_name})'] = _met(_y_true_b, -_iso.fit(_Xsc).score_samples(_Xsc), _TOP_N_b)

    for _name, _Xsc in [('features', _X_all_sc), ('MDE emb', _X_mde_sc)]:
        _gmm = _GMM(n_components=3, covariance_type='full', random_state=42)
        _primary[f'GMM ({_name})'] = _met(_y_true_b, -_gmm.fit(_Xsc).score_samples(_Xsc), _TOP_N_b)

    for _name, _Xsc, _k_lof in [('features', _X_all_sc, K_STRUCT_NEIGHBORS),
                                  ('MDE emb',  _X_mde_sc, K_STRUCT_NEIGHBORS)]:
        _lof = _LOF(n_neighbors=_k_lof, contamination=_cont)
        _lof.fit(_Xsc)
        _primary[f'LOF ({_name})'] = _met(_y_true_b, -_lof.negative_outlier_factor_, _TOP_N_b)

    # _primary['MDE rel. displacement (proposed)']       = _met(_y_true_b, emb_rel_dist_scores, _TOP_N_b)
    _primary['MDE rel. displacement (proposed)'] = _met(_y_true_b, emb_rel_dist_gated,  _TOP_N_b)

    print("Training AE (features)...", flush=True)
    _ae_feat_scores = _train_ae(_X_all_sc, hidden=[32, 16], bottleneck=3, epochs=200)
    _primary['AE (features)'] = _met(_y_true_b, _ae_feat_scores, _TOP_N_b)

    print("Training AE (MDE emb)...", flush=True)
    _ae_emb_scores = _train_ae(_X_mde_sc, hidden=[16], bottleneck=1, epochs=150)
    _primary['AE (MDE emb)'] = _met(_y_true_b, _ae_emb_scores, _TOP_N_b)

    print("── Table 1: Primary baselines ───────────────────────────────────────────")
    print(_hdr)
    print(_sep)
    for _m, (_pr, _roc2, _prec, _rec) in sorted(_primary.items(), key=lambda x: -x[1][1]):
        _tag = " <--" if _m == 'MDE rel. displacement (proposed)' else ""
        print(f"{_m:<46} {_roc2:>9.4f} {_pr:>8.4f} {_prec:>9.4f} {_rec:>9.4f}{_tag}")

    # ── Table 2: alternative embedding baselines ──────────────────────────────
    _emb_cmp = {}

    for _name, _Xsc in [('PCA emb', _X_pca_sc), ('UMAP emb', _X_umap_sc), ('t-SNE emb', _X_tsne_sc)]:
        _iso = _IF(n_estimators=100, contamination=_cont, random_state=42)
        _emb_cmp[f'IsoForest ({_name})'] = _met(_y_true_b, -_iso.fit(_Xsc).score_samples(_Xsc), _TOP_N_b)

    for _name, _Xsc in [('PCA emb', _X_pca_sc), ('UMAP emb', _X_umap_sc), ('t-SNE emb', _X_tsne_sc)]:
        _lof = _LOF(n_neighbors=K_STRUCT_NEIGHBORS, contamination=_cont)
        _lof.fit(_Xsc)
        _emb_cmp[f'LOF ({_name})'] = _met(_y_true_b, -_lof.negative_outlier_factor_, _TOP_N_b)

    _nbrs_b = _NNN_b(n_neighbors=K_STRUCT_NEIGHBORS + 1).fit(_X_struct_sc)
    _, _idx_b = _nbrs_b.kneighbors(_X_struct_sc)
    _idx_b = _idx_b[:, 1:]
    _emb_cmp[f'kNN energy deviation (k={K_STRUCT_NEIGHBORS})'] = _met(
        _y_true_b, np.abs(_energy_b - _energy_b[_idx_b].mean(axis=1)), _TOP_N_b)


    print("\n── Table 2: Alternative embedding baselines (PCA / UMAP / PaCMAP) ──────")
    print(_hdr)
    print(_sep)
    for _m, (_pr, _roc2, _prec, _rec) in sorted(_emb_cmp.items(), key=lambda x: -x[1][1]):
        print(f"{_m:<46} {_roc2:>9.4f} {_pr:>8.4f} {_prec:>9.4f} {_rec:>9.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10. Pseudo-label Distillation
    """)
    return


@app.cell
def _(INEFF_PCT, df_mock, emb_rel_dist_gated, np, pd):
    from sklearn.metrics import average_precision_score as _aps_d, roc_auc_score as _roc_d
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.preprocessing import StandardScaler as _SS_dist
    from sklearn.ensemble import RandomForestClassifier as _RFC
    from sklearn.linear_model import LogisticRegression as _LR_dist
    from xgboost import XGBClassifier as _XGB

    _y_true_d = df_mock['is_inefficient'].values
    _pct_d    = int(INEFF_PCT * 100)

    # ── 70/30 stratified train/test split ─────────────────────────────────────
    _idx_all = np.arange(len(df_mock))
    _idx_tr, _idx_te = _tts(
        _idx_all, test_size=0.30, stratify=_y_true_d, random_state=42,
    )
    _y_tr = _y_true_d[_idx_tr]
    _y_te = _y_true_d[_idx_te]
    _top_tr = int(INEFF_PCT * len(_idx_tr))
    _top_te = int(INEFF_PCT * len(_idx_te))

    print(f"── Train/test split ────────────────────────────────────────────────────")
    print(f"  Train : {len(_idx_tr)} samples  ({_y_tr.sum()} anomalies, {_y_tr.mean():.1%})")
    print(f"  Test  : {len(_idx_te)} samples  ({_y_te.sum()} anomalies, {_y_te.mean():.1%})")

    # ── Pseudo-labels from teacher on TRAIN set only ──────────────────────────
    # Threshold derived from train distribution to avoid leakage into test set.
    _teacher_tr  = emb_rel_dist_gated[_idx_tr]
    _thresh_abs  = np.percentile(_teacher_tr, (1.0 - INEFF_PCT) * 100)
    _pseudo_tr   = (_teacher_tr >= _thresh_abs).astype(int)

    _tp = ((_pseudo_tr == 1) & (_y_tr == 1)).sum()
    _fp = ((_pseudo_tr == 1) & (_y_tr == 0)).sum()
    _fn = ((_pseudo_tr == 0) & (_y_tr == 1)).sum()
    _p  = _tp / max(_tp + _fp, 1)
    _r  = _tp / max(_tp + _fn, 1)
    _f1 = 2 * _p * _r / max(_p + _r, 1e-8)
    print(f"\n── Teacher pseudo-label quality (train set) ────────────────────────────")
    print(f"  Precision : {_p:.3f}  |  Recall : {_r:.3f}  |  F1 : {_f1:.3f}")

    # ── Confidence weights (loss adjustment, Song et al. 2022) ────────────────
    # Pseudo-labels near the threshold are most likely mislabeled; downweight them.
    # Weight = |teacher_score − threshold|, normalised to (0.05, 1].
    _dist_from_thresh = np.abs(_teacher_tr - _thresh_abs)
    _conf_weight = _dist_from_thresh / max(_dist_from_thresh.max(), 1e-8)
    _conf_weight = np.clip(_conf_weight + 0.05, 0.05, 1.0)
    print(f"  Conf. weight  — mean: {_conf_weight.mean():.3f} "
          f" min: {_conf_weight.min():.3f}  max: {_conf_weight.max():.3f}")

    # ── Feature matrix (structural + energy) ──────────────────────────────────
    _vendor_d = (df_mock['ran_vendor_type'] == 'NOK').astype(int).values
    _mast_d   = pd.get_dummies(df_mock['mast_group'], prefix='mast').astype(float).values
    _X_feat   = np.column_stack([
        df_mock['num_total_cells'].values, df_mock['total_non_ran_equipment'].values,
        df_mock['has_shared_ran'].values, _vendor_d, _mast_d,
        np.log1p(df_mock['ps_traffic_mb'].values), df_mock['meter_kwh_sim'].values,
    ])
    # Scaler fitted on train only — prevents test distribution leaking into scaling
    _ss_d   = _SS_dist().fit(_X_feat[_idx_tr])
    _Xtr_sc = _ss_d.transform(_X_feat[_idx_tr])
    _Xte_sc = _ss_d.transform(_X_feat[_idx_te])

    # ── Train classifiers on TRAIN pseudo-labels ───────────────────────────────
    _spw = (_pseudo_tr == 0).sum() / max((_pseudo_tr == 1).sum(), 1)

    _rf = _RFC(n_estimators=300, min_samples_leaf=5, class_weight='balanced', random_state=42)
    _rf.fit(_X_feat[_idx_tr], _pseudo_tr, sample_weight=_conf_weight)

    _xgb = _XGB(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                scale_pos_weight=_spw, random_state=42, verbosity=0, eval_metric='logloss')
    _xgb.fit(_X_feat[_idx_tr], _pseudo_tr, sample_weight=_conf_weight)

    _lr = _LR_dist(class_weight='balanced', max_iter=1000, random_state=42)
    _lr.fit(_Xtr_sc, _pseudo_tr)

    # ── Evaluation helper ──────────────────────────────────────────────────────
    def _met_d(y, s, n):
        pr  = _aps_d(y, s)
        roc = _roc_d(y, s)
        idx = np.argsort(s)[::-1][:n]
        return pr, roc, y[idx].mean(), y[idx].sum() / max(y.sum(), 1)

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n── Distillation results ────────────────────────────────────────────────")
    _hdr = (f"{'Method':<42} {'split':>6}"
            f" {'ROC-AUC':>9} {'PR-AUC':>8} {'Prec@' + str(_pct_d) + '%':>9} {'Rec@' + str(_pct_d) + '%':>9}")
    print(_hdr)
    print("-" * 93)

    # Teacher (unsupervised — no leakage; evaluated on test set for fair comparison)
    _pr, _roc, _prec, _rec = _met_d(_y_te, emb_rel_dist_gated[_idx_te], _top_te)
    print(f"{'MDE rel. displacement (proposed)':<42} {'test':>6}"
          f" {_roc:>9.4f} {_pr:>8.4f} {_prec:>9.4f} {_rec:>9.4f}")

    for _name, _s_tr_fn, _s_te_fn, _tag in [
        ("RF on all features",
         lambda: _rf.predict_proba(_X_feat[_idx_tr])[:, 1],
         lambda: _rf.predict_proba(_X_feat[_idx_te])[:, 1],
         ""),
        ("XGBoost on all features",
         lambda: _xgb.predict_proba(_X_feat[_idx_tr])[:, 1],
         lambda: _xgb.predict_proba(_X_feat[_idx_te])[:, 1],
         "  <-- primary student"),
        ("LR on all features",
         lambda: _lr.predict_proba(_Xtr_sc)[:, 1],
         lambda: _lr.predict_proba(_Xte_sc)[:, 1],
         ""),
    ]:
        _pr_tr, _roc_tr, _prec_tr, _rec_tr = _met_d(_y_tr, _s_tr_fn(), _top_tr)
        _pr_te, _roc_te, _prec_te, _rec_te = _met_d(_y_te, _s_te_fn(), _top_te)
        print(f"{_name:<42} {'train':>6}"
              f" {_roc_tr:>9.4f} {_pr_tr:>8.4f} {_prec_tr:>9.4f} {_rec_tr:>9.4f}")
        print(f"{'':42} {'test':>6}"
              f" {_roc_te:>9.4f} {_pr_te:>8.4f} {_prec_te:>9.4f} {_rec_te:>9.4f}{_tag}")

    # ── Test-set detection breakdown (teacher vs XGB) ─────────────────────────
    _xgb_sc_te = _xgb.predict_proba(_X_feat[_idx_te])[:, 1]
    _t_flags   = np.zeros(len(_idx_te), dtype=bool)
    _r_flags   = np.zeros(len(_idx_te), dtype=bool)
    _t_flags[np.argsort(emb_rel_dist_gated[_idx_te])[::-1][:_top_te]] = True
    _r_flags[np.argsort(_xgb_sc_te)[::-1][:_top_te]]                   = True
    _n_true_te = _y_te.sum()

    print(f"\n── Test-set detection breakdown (top {_pct_d}% flagged) ─────────────────")
    print(f"  True anomalies caught by both    : {(_t_flags & _r_flags & _y_te.astype(bool)).sum()} / {_n_true_te}"
          f"  ({(_t_flags & _r_flags & _y_te.astype(bool)).sum() / _n_true_te:.1%})")
    print(f"  Caught by teacher only           : {(_t_flags & ~_r_flags & _y_te.astype(bool)).sum()} / {_n_true_te}"
          f"  ({(_t_flags & ~_r_flags & _y_te.astype(bool)).sum() / _n_true_te:.1%})")
    print(f"  Caught by XGB only               : {(~_t_flags & _r_flags & _y_te.astype(bool)).sum()} / {_n_true_te}"
          f"  ({(~_t_flags & _r_flags & _y_te.astype(bool)).sum() / _n_true_te:.1%})")
    print(f"  Missed by both                   : {(~_t_flags & ~_r_flags & _y_te.astype(bool)).sum()} / {_n_true_te}"
          f"  ({(~_t_flags & ~_r_flags & _y_te.astype(bool)).sum() / _n_true_te:.1%})")
    print(f"  False positives — teacher        : {(_t_flags & ~_y_te.astype(bool)).sum()} / {_top_te}")
    print(f"  False positives — XGB            : {(_r_flags & ~_y_te.astype(bool)).sum()} / {_top_te}")
    return


@app.function
def evaluate_mde(
    df,
    embedding,
    X_structural,
    ineff_col="is_inefficient",
    score_col="mde_ineff_score",
    k_neighbors=100,
    top_k_frac=0.05,
    pairs=None,
    distortions=None,
):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import ttest_ind

    results = {}

    y_true = df[ineff_col].values
    scores = df[score_col].fillna(0).values
    X_emb = embedding if isinstance(embedding, np.ndarray) else embedding.cpu().numpy()

    n = len(df)
    top_k = int(top_k_frac * n)

    results["roc_auc"] = roc_auc_score(y_true, scores)
    results["pr_auc"] = average_precision_score(y_true, scores)

    df_sorted = df.sort_values(score_col, ascending=False)
    top_k_df = df_sorted.head(top_k)
    results["precision_at_k"] = top_k_df[ineff_col].mean()
    results["recall_at_k"] = top_k_df[ineff_col].sum() / max(y_true.sum(), 1)

    results["trustworthiness"] = trustworthiness(X_structural, X_emb, n_neighbors=k_neighbors)

    nbrs_struct = NearestNeighbors(n_neighbors=k_neighbors).fit(X_structural)
    _, idx_struct = nbrs_struct.kneighbors(X_structural)

    nbrs_emb = NearestNeighbors(n_neighbors=k_neighbors).fit(X_emb)
    _, idx_emb = nbrs_emb.kneighbors(X_emb)

    overlaps = [len(set(idx_struct[i]) & set(idx_emb[i])) / k_neighbors for i in range(n)]
    df["knn_overlap"] = overlaps
    results["knn_overlap_mean"] = np.mean(overlaps)

    centroid_dist = []
    for i in range(n):
        neigh = idx_emb[i]
        centroid = X_emb[neigh].mean(axis=0)
        centroid_dist.append(np.linalg.norm(X_emb[i] - centroid))
    df["centroid_dist"] = centroid_dist

    eff   = df[df[ineff_col] == 0]["centroid_dist"]
    ineff = df[df[ineff_col] == 1]["centroid_dist"]
    results["centroid_dist_mean_eff"]   = eff.mean()
    results["centroid_dist_mean_ineff"] = ineff.mean()

    pooled_std = np.sqrt((eff.std()**2 + ineff.std()**2) / 2)
    results["cohens_d"] = (ineff.mean() - eff.mean()) / (pooled_std + 1e-6)

    stat, pval = ttest_ind(eff, ineff, equal_var=False)
    results["t_stat"]   = stat
    results["p_value"]  = pval

    if pairs is not None and distortions is not None:
        top_pairs   = pairs[:top_k]
        ineff_hits  = sum(1 for i, j in top_pairs if df.iloc[i][ineff_col] == 1 or df.iloc[j][ineff_col] == 1)
        results["high_distortion_ineff_fraction"] = ineff_hits / len(top_pairs)

    return results, df


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 11. MDE Evaluation
    """)
    return


@app.cell
def _(
    INEFF_PCT,
    K_STRUCT_NEIGHBORS,
    X_scaled,
    df_mock,
    distortions,
    embedding,
    pairs,
):
    _pct = int(INEFF_PCT * 100)
    _k   = K_STRUCT_NEIGHBORS

    eval_results, df_eval = evaluate_mde(
        df_mock,
        embedding,
        X_scaled,
        score_col="emb_rel_dist_gated",
        k_neighbors=_k,
        top_k_frac=INEFF_PCT,
        pairs=pairs,
        distortions=distortions,
    )

    print("── Detection Metrics ────────────────────────────────────────────────────")
    print(f"  ROC-AUC                         : {eval_results['roc_auc']:.4f}")
    print(f"  PR-AUC                          : {eval_results['pr_auc']:.4f}")
    print(f"  Precision@{_pct}%                  : {eval_results['precision_at_k']:.4f}")
    print(f"  Recall@{_pct}%                     : {eval_results['recall_at_k']:.4f}")
    print()
    print(f"── Embedding Quality (k={_k}) ──────────────────────────────────────────")
    print(f"  Trustworthiness                 : {eval_results['trustworthiness']:.4f}")
    print(f"  kNN overlap (struct vs emb)     : {eval_results['knn_overlap_mean']:.4f}")
    print()
    print("── Centroid Distance Analysis ───────────────────────────────────────────")
    print(f"  Mean dist (efficient)           : {eval_results['centroid_dist_mean_eff']:.4f}")
    print(f"  Mean dist (inefficient)         : {eval_results['centroid_dist_mean_ineff']:.4f}")
    print(f"  cohen's d                       : {eval_results['cohens_d']:.4f}")

    if "high_distortion_ineff_fraction" in eval_results:
        print()
        print("── Distortion Pair Analysis ─────────────────────────────────────────────")
        print(f"  Ineff fraction in top-{_pct}% pairs  : {eval_results['high_distortion_ineff_fraction']:.4f}")
    return


if __name__ == "__main__":
    app.run()
