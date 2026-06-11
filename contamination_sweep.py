"""Contamination robustness sweep — v27: feature baselines include traffic.

Extends v25 by adding log_ps_traffic_mb to the feature matrix used by
iso_feat, lof_feat, gmm_feat, ae_feat.  All MDE scores and residual baselines
are loaded directly from the v25 pkl — only the 4 feature-based methods are
recomputed here (no MDE re-embedding needed).

v27 settings (identical to v25):
  BETA=35, K_BASELINE=10, K_GRAPH=300, K_STRUCT_NEIGHBORS=50, DELTA_PERCENTILE=35
  TRAFFIC_WEIGHT=0.05, EMBEDDING_DIM=4, N_DISSIMILAR_MULT=4, NEG_WEIGHT=-2.0
  Lognormal multiplicative noise, UNIFORM sampling, 4 injection types
  delta_node: stratified kNN (vendor+sharing+mast), P35 of neighbour energies
  Edge repulsion: max of both endpoints

Feature change (v27 only):
  X_feat = [struct_block | log_ps_traffic_mb | energy]  (traffic added)

Output: data/contamination_sweep_v27_results.pkl
        data/contamination_sweep_v27_residual.{png,pdf}
        data/contamination_sweep_v27_unsupervised.{png,pdf}
        data/contamination_sweep_v27_feat.{png,pdf}
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as _nn_module
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────────
K_STRUCT_NEIGHBORS = 50
N_SEEDS            = 10

SIGMA_LOG_MAP = {
    'Tower': 0.04, 'Disguised': 0.035,
    'Rooftop': 0.03, 'Pole': 0.02, 'Other': 0.0325,
}

CONTAMINATION_RATES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
                       0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]

# ── Load real data ────────────────────────────────────────────────────────────
DATA_PATH = "/home/eliud_aims_ac_za/myproject/data_extended.pkl"
df = pd.read_pickle(DATA_PATH)

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
df['mast_group'] = df['mast_type'].map(_mast_bucket_map).fillna('Other')

# ── Physics model ─────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
_models = []
for (_shared, _vendor, _mast), _g in df.groupby(['has_shared_ran', 'ran_vendor_type', 'mast_group']):
    if len(_g) < 3:
        continue
    _m = LinearRegression(positive=True, fit_intercept=True)
    _m.fit(_g[['num_total_cells', 'total_non_ran_equipment']].values, _g['meter_kwh'].values)
    _models.append({
        'has_shared_ran': _shared, 'ran_vendor_type': _vendor, 'mast_group': _mast,
        'base_load': _m.intercept_, 'alpha_cells': _m.coef_[0], 'beta_non_ran': _m.coef_[1],
    })
physics_df = pd.DataFrame(_models).round(2)
_mask = (physics_df['has_shared_ran'] == 1) & (physics_df['ran_vendor_type'] == 'HUA')
physics_df.loc[_mask, 'base_load']    = 480.0
physics_df.loc[_mask, 'beta_non_ran'] = 220.0

def _predict_kwh(num_cells, num_non_ran, shared, vendor, mast):
    row = physics_df[(physics_df['has_shared_ran'] == shared) &
                     (physics_df['ran_vendor_type'] == vendor) &
                     (physics_df['mast_group'] == mast)]
    if len(row) == 0:
        row = physics_df[(physics_df['has_shared_ran'] == shared) &
                         (physics_df['ran_vendor_type'] == vendor)]
    r = row.iloc[0]
    return r['base_load'] + r['alpha_cells'] * num_cells + r['beta_non_ran'] * num_non_ran

# ── Synthetic data (fixed, uniform sampling) ──────────────────────────────────
n_samples    = 5000
_config_cols = ['num_total_cells', 'total_non_ran_equipment',
                'has_shared_ran', 'ran_vendor_type', 'mast_group']

_joint = (df[_config_cols]
          .value_counts(normalize=True).reset_index(name='probability')).round(2).reset_index()
_joint['index'] = _joint['index'].astype(str)

df_mock = _joint.sample(n=n_samples, replace=True, weights=None, random_state=42).reset_index(drop=True)

_traffic_lookup     = df.groupby(_config_cols)['ps_traffic_mb'].apply(list).to_dict()
_traffic_global_med = df['ps_traffic_mb'].median()
_traffic_global_std = df['ps_traffic_mb'].std()

_rng_t = np.random.default_rng(99)
_ps_vals = []
for _, _row in df_mock.iterrows():
    _key  = tuple(_row[c] for c in _config_cols)
    _pool = _traffic_lookup.get(_key)
    if _pool:
        _val = float(_rng_t.choice(_pool))
        _val = max(_val + _rng_t.normal(0, _traffic_global_std * 0.05), 0.0)
    else:
        _val = _traffic_global_med
    _ps_vals.append(_val)
df_mock['ps_traffic_mb'] = np.array(_ps_vals)

df_mock['kwh_expected'] = df_mock.apply(
    lambda r: _predict_kwh(r['num_total_cells'], r['total_non_ran_equipment'],
                           r['has_shared_ran'], r['ran_vendor_type'], r['mast_group']), axis=1)

np.random.seed(42)
_sigma_log = df_mock['mast_group'].map(SIGMA_LOG_MAP).values
_log_eps   = np.random.normal(0, _sigma_log)
_noise_std = _sigma_log * df_mock['kwh_expected'].values
df_mock['noise_std'] = _noise_std
_kwh_base  = df_mock['kwh_expected'].values * np.exp(_log_eps)

_traf_med = df_mock['ps_traffic_mb'].median()
_traf_med = _traf_med if _traf_med > 0 else df_mock['ps_traffic_mb'].mean()

_cooling_range = {
    'Tower': (200, 400), 'Disguised': (150, 350),
    'Rooftop': (80, 200), 'Pole': (100, 250), 'Other': (100, 200),
}

# ── Feature block: struct + log_traffic (energy appended per seed) ─────────────
_vendor_b = (df_mock['ran_vendor_type'] == 'NOK').astype(int).values
_mast_b   = pd.get_dummies(df_mock['mast_group'], prefix='mast').astype(float).values
_log_traffic = np.log1p(df_mock['ps_traffic_mb'].values)

_struct_block_with_traffic = np.column_stack([
    df_mock['num_total_cells'].values,
    df_mock['total_non_ran_equipment'].values,
    df_mock['has_shared_ran'].values,
    _vendor_b, _mast_b,
    _log_traffic,
])

# ── Inject anomalies ──────────────────────────────────────────────────────────
def inject_anomalies(contamination_rate, rng_seed):
    n_bad      = int(contamination_rate * n_samples)
    n_per_type = n_bad // 4
    rng        = np.random.default_rng(rng_seed)

    counts = [0, 0, 0, 0]
    itype  = np.zeros(n_samples, dtype=int)
    for _i in rng.permutation(n_samples).tolist():
        if sum(counts) >= n_bad:
            break
        opts = [t for t in range(4) if counts[t] < n_per_type]
        if not opts:
            break
        t = int(rng.choice(opts))
        itype[_i] = t + 1
        counts[t] += 1

    meter = _kwh_base.copy()
    for i in np.where(itype == 1)[0]:
        meter[i] *= rng.uniform(1.2, 1.8)
    for i in np.where(itype == 2)[0]:
        lo, hi = _cooling_range.get(df_mock.iloc[i]['mast_group'], (100, 200))
        meter[i] += rng.uniform(lo, hi)
    for i in np.where(itype == 3)[0]:
        cells      = df_mock.iloc[i]['num_total_cells']
        traffic    = df_mock.iloc[i]['ps_traffic_mb']
        idle       = max(1.0 - traffic / _traf_med, 0.1)
        site_noise = df_mock.iloc[i]['noise_std']
        overhead   = max(cells, 5) ** 2 * idle * rng.uniform(0.5, 1.5)
        meter[i]  += max(overhead, 2.0 * site_noise)
    for i in np.where(itype == 4)[0]:
        nran       = df_mock.iloc[i]['total_non_ran_equipment']
        site_noise = df_mock.iloc[i]['noise_std']
        overhead   = (nran + 1) ** 2 * rng.uniform(20, 50)
        meter[i]  += max(overhead, 2.0 * site_noise)

    return np.round(meter, 2), (itype > 0).astype(int)

# ── Autoencoder helper ────────────────────────────────────────────────────────
class _SimpleAE(_nn_module.Module):
    def __init__(self, in_dim, hidden, bottleneck):
        super().__init__()
        enc, dec = [], []
        prev = in_dim
        for h in hidden:
            enc += [_nn_module.Linear(prev, h), _nn_module.ReLU()]; prev = h
        enc.append(_nn_module.Linear(prev, bottleneck))
        prev = bottleneck
        for h in reversed(hidden):
            dec += [_nn_module.Linear(prev, h), _nn_module.ReLU()]; prev = h
        dec.append(_nn_module.Linear(prev, in_dim))
        self.enc = _nn_module.Sequential(*enc)
        self.dec = _nn_module.Sequential(*dec)
    def forward(self, x): return self.dec(self.enc(x))

def _train_ae(X_np, hidden, bottleneck, epochs, seed):
    torch.manual_seed(seed)
    X_t = torch.tensor(X_np, dtype=torch.float32)
    ae  = _SimpleAE(X_np.shape[1], hidden, bottleneck)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    dl  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t), batch_size=256, shuffle=True)
    ae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            loss = _nn_module.functional.mse_loss(ae(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        recon = ae(X_t)
    return ((X_t - recon) ** 2).mean(dim=1).numpy()

# ── Load v25 base results ─────────────────────────────────────────────────────
_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_v25_pkl  = os.path.join(_data_dir, "contamination_sweep_v25_results.pkl")
with open(_v25_pkl, "rb") as _f:
    _v25 = pickle.load(_f)

METHODS = ['phys', 'mde_erd', 'mde_erd_gated', 'rf', 'lr', 'huber',
           'raw_energy', 'iso_feat', 'iso_mde', 'gmm_feat', 'gmm_mde',
           'lof_feat', 'lof_mde', 'ae_feat', 'ae_mde']

# Start from v25; overwrite only the 4 feat methods below
all_roc = {m: _v25['all_roc'][m].copy() for m in METHODS}
for m in ['iso_feat', 'gmm_feat', 'lof_feat', 'ae_feat']:
    all_roc[m] = np.full((len(CONTAMINATION_RATES), N_SEEDS), np.nan)

print(f"Loaded v25 base results from {_v25_pkl}")
print(f"Recomputing iso_feat, gmm_feat, lof_feat, ae_feat with traffic included.\n")
print(f"Contamination sweep v27  ({N_SEEDS} seeds, {len(CONTAMINATION_RATES)} rates)\n")

# ── Sweep ─────────────────────────────────────────────────────────────────────
for ri, rate in enumerate(CONTAMINATION_RATES):
    print(f"Rate {rate:.0%}  ", end="", flush=True)
    for seed in range(N_SEEDS):
        energy, y_true = inject_anomalies(rate, rng_seed=seed)
        _two_classes   = len(np.unique(y_true)) > 1

        X_feat_sc = StandardScaler().fit_transform(
            np.column_stack([_struct_block_with_traffic, energy]))

        def _roc(s):
            return roc_auc_score(y_true, s) if _two_classes else float("nan")

        _iso = IsolationForest(n_estimators=100, contamination='auto', random_state=seed)
        all_roc['iso_feat'][ri, seed] = _roc(-_iso.fit(X_feat_sc).score_samples(X_feat_sc))

        _gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=seed)
        all_roc['gmm_feat'][ri, seed] = _roc(-_gmm.fit(X_feat_sc).score_samples(X_feat_sc))

        _lof = LocalOutlierFactor(n_neighbors=K_STRUCT_NEIGHBORS, contamination='auto')
        _lof.fit(X_feat_sc)
        all_roc['lof_feat'][ri, seed] = _roc(-_lof.negative_outlier_factor_)

        all_roc['ae_feat'][ri, seed] = _roc(
            _train_ae(X_feat_sc, hidden=[32, 16], bottleneck=3, epochs=100, seed=seed))

        print(".", end="", flush=True)

    def _fmt(m):
        return f"{all_roc[m][ri].mean():.4f}±{all_roc[m][ri].std():.4f}"
    print(f"\n  gated={_fmt('mde_erd_gated')}  iso_feat={_fmt('iso_feat')}"
          f"  gmm_feat={_fmt('gmm_feat')}  lof_feat={_fmt('lof_feat')}"
          f"  ae_feat={_fmt('ae_feat')}\n", flush=True)

print("Done.")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\nSummary — mean ROC-AUC  ({N_SEEDS} seeds, feat methods now include traffic)")
_col = 12
_hdr = (f"{'Rate':>6} | {'Gated(v27)':>{_col}} | {'IsoFeat':>{_col}}"
        f" | {'GmmFeat':>{_col}} | {'LofFeat':>{_col}} | {'AEFeat':>{_col}}"
        f" | {'IsoFeat(v25)':>13} | {'RF':>{_col}}")
print(_hdr)
print("-" * len(_hdr))
for ri, rate in enumerate(CONTAMINATION_RATES):
    def _f(m): return f"{all_roc[m][ri].mean():.4f}"
    def _fv25(m): return f"{_v25['all_roc'][m][ri].mean():.4f}"
    print(f"{rate:>6.0%} | {_f('mde_erd_gated'):>{_col}} | {_f('iso_feat'):>{_col}}"
          f" | {_f('gmm_feat'):>{_col}} | {_f('lof_feat'):>{_col}} | {_f('ae_feat'):>{_col}}"
          f" | {_fv25('iso_feat'):>13} | {_f('rf'):>{_col}}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(_data_dir, exist_ok=True)
_pkl_path = os.path.join(_data_dir, "contamination_sweep_v27_results.pkl")
with open(_pkl_path, "wb") as _f:
    pickle.dump({
        'all_roc': all_roc, 'rates': CONTAMINATION_RATES,
        'n_seeds': N_SEEDS, 'methods': METHODS,
        'params': {
            'K_STRUCT_NEIGHBORS': K_STRUCT_NEIGHBORS,
            'noise_model': 'lognormal_multiplicative',
            'sampling': 'uniform', 'injection_types': [1, 2, 3, 4],
            'base': 'v25',
            'change': 'feat baselines now include log_ps_traffic_mb',
        },
    }, _f)
print(f"\nRaw results saved → {_pkl_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_rates = [r * 100 for r in CONTAMINATION_RATES]

def _draw_plot(ax, methods_spec, legend_loc="upper right"):
    for label, key, color, ls, marker in methods_spec:
        means = all_roc[key].mean(axis=1)
        lo    = np.percentile(all_roc[key], 2.5,  axis=1)
        hi    = np.percentile(all_roc[key], 97.5, axis=1)
        ax.plot(_rates, means, color=color, ls=ls, marker=marker,
                ms=5, label=label, lw=1.8)
        ax.fill_between(_rates, lo, hi, alpha=0.15, color=color)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random baseline")
    ax.set_xlabel("Contamination rate (%)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8.5, loc=legend_loc, framealpha=0.85,
              borderpad=0.5, labelspacing=0.3, handlelength=1.8)
    ax.grid(alpha=0.3)

_residual_methods = [
    ("Physics residual (upper bound)", "phys",          "#aec7e8", ":",  "x"),
    ("MDE emb_rel_dist (proposed)",    "mde_erd_gated", "#e67e22", "-",  "^"),
    ("Random Forest residual",         "rf",            "#2ca02c", "--", "s"),
    ("LR residual",                    "lr",            "#9467bd", "--", "P"),
    ("Huber residual",                 "huber",         "#e377c2", "--", "v"),
]
fig1, ax1 = plt.subplots(figsize=(9, 5))
_draw_plot(ax1, _residual_methods, legend_loc="lower left")
fig1.tight_layout()
fig1.savefig(os.path.join(_data_dir, "contamination_sweep_v27_residual.png"), dpi=150, bbox_inches="tight")
fig1.savefig(os.path.join(_data_dir, "contamination_sweep_v27_residual.pdf"), bbox_inches="tight")
plt.close(fig1)

_unsupervised_methods = [
    ("MDE emb_rel_dist (proposed)", "mde_erd_gated", "#e67e22", "-",  "^"),
    ("IsoForest (MDE emb)",         "iso_mde",       "#17becf", "--", "D"),
    ("GMM (MDE emb)",               "gmm_mde",       "#d62728", "--", "s"),
    ("AE (MDE emb)",                "ae_mde",        "#2ca02c", "--", "v"),
]
fig2, ax2 = plt.subplots(figsize=(9, 5))
_draw_plot(ax2, _unsupervised_methods, legend_loc="lower left")
fig2.tight_layout()
fig2.savefig(os.path.join(_data_dir, "contamination_sweep_v27_unsupervised.png"), dpi=150, bbox_inches="tight")
fig2.savefig(os.path.join(_data_dir, "contamination_sweep_v27_unsupervised.pdf"), bbox_inches="tight")
plt.close(fig2)

_feat_methods = [
    ("MDE emb_rel_dist (proposed)", "mde_erd_gated", "#e67e22", "-",  "^"),
    ("IsoForest (features)",        "iso_feat",      "#17becf", "--", "D"),
    ("LOF (features)",              "lof_feat",      "#8c564b", "--", "o"),
    ("GMM (features)",              "gmm_feat",      "#d62728", "--", "s"),
    ("AE (features)",               "ae_feat",       "#2ca02c", "--", "v"),
]
fig3, ax3 = plt.subplots(figsize=(9, 5))
_draw_plot(ax3, _feat_methods, legend_loc="upper right")
fig3.tight_layout()
fig3.savefig(os.path.join(_data_dir, "contamination_sweep_v27_feat.png"), dpi=150, bbox_inches="tight")
fig3.savefig(os.path.join(_data_dir, "contamination_sweep_v27_feat.pdf"), bbox_inches="tight")
plt.close(fig3)

print(f"Plots saved → {_data_dir}/contamination_sweep_v27_{{residual,unsupervised,feat}}.{{png,pdf}}")
