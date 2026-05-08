#!/usr/bin/env python3
"""
generate_paper_figures.py
Publication-ready figures (5 main + 2 supplementary) for CEUS submission.

Outputs
-------
outputs/figures/paper/          PDF + PNG @ 300 DPI
outputs/figures/paper/preview/  PNG @ 150 DPI
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

# ── paths ────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / 'data' / 'processed'
RES_DIR  = BASE / 'outputs' / 'results'
FIG_DIR  = BASE / 'outputs' / 'figures' / 'paper'
PRE_DIR  = FIG_DIR / 'preview'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRE_DIR.mkdir(parents=True, exist_ok=True)

# ── archetype palette ────────────────────────────────────────────────────
ARCH_ORDER  = ['Hotspot', 'Lowland baseline', 'Upland baseline', 'Upland shield']
ARCH_COLORS = {
    'Hotspot'         : '#C8102E',
    'Lowland baseline': '#E9C46A',
    'Upland baseline' : '#A8DADC',
    'Upland shield'   : '#2E86AB',
}

# ── global style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.titleweight' : 'normal',
    'legend.fontsize'  : 9,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
})

FEATURES = [
    'elevation', 'slope', 'TWI', 'log_flow_accum', 'dist_to_stream',
    'dist_to_street', 'ISA_frac', 'log_lot_area', 'is_enclave',
    'conn_topo', 'Cw_topo', 'HAND_min',
]
FEAT_LABELS = {
    'elevation'     : 'Elevation',
    'slope'         : 'Slope',
    'TWI'           : 'TWI',
    'log_flow_accum': 'log(Flow accum.)',
    'dist_to_stream': 'Dist. to stream',
    'dist_to_street': 'Dist. to street',
    'ISA_frac'      : 'ISA fraction',
    'log_lot_area'  : 'log(Lot area)',
    'is_enclave'    : 'Is enclave',
    'conn_topo'     : 'Conn. (topo.)',
    'Cw_topo'       : 'Cw (topo.)',
    'HAND_min'      : 'HANDₘᴵⁿ',
}

# ── helper ───────────────────────────────────────────────────────────────
def save_fig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(PRE_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    print(f'  [OK] {name}')

# ─────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('Loading data...')

fm   = pd.read_csv(DATA_DIR / 'feature_matrix.csv')
arc  = pd.read_csv(RES_DIR  / 'parcel_archetypes_v2.csv')
cv   = pd.read_csv(RES_DIR  / 'cv_summary_v2.csv')
cf   = pd.read_csv(RES_DIR  / 'counterfactual_20pct_isa.csv')
iv   = pd.read_csv(RES_DIR  / 'intervention_scenarios_residential.csv')
ac   = pd.read_csv(RES_DIR  / 'archetype_characteristics_v2.csv')
stab = pd.read_csv(RES_DIR  / 'cluster_stability_v2.csv', index_col=0)

print(f'  feature_matrix : {len(fm):,} rows')
print(f'  archetypes     : {len(arc):,} rows')
print(f'  counterfactual : {len(cf):,} rows')

# ─────────────────────────────────────────────────────────────────────────
# Re-run Ridge sub-basin CV to get per-fold Spearman
# ─────────────────────────────────────────────────────────────────────────
print('Re-running Ridge sub-basin CV (per-fold data)...')

X_all = fm[FEATURES].values
y_all = fm['target_max_log'].values
fold_ids    = fm['fold_subbasin'].values.astype(str)   # e.g. 'r27', 'nan'
valid_folds = sorted([f for f in np.unique(fold_ids) if f != 'nan'])

per_fold_rows = []
for fid in valid_folds:
    test_mask  = fold_ids == fid
    train_mask = (fold_ids != 'nan') & ~test_mask
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all[train_mask])
    X_te = sc.transform(X_all[test_mask])
    y_tr = y_all[train_mask]
    y_te = y_all[test_mask]
    mdl  = Ridge(alpha=1.0).fit(X_tr, y_tr)
    rho  = spearmanr(y_te, mdl.predict(X_te)).statistic
    per_fold_rows.append({
        'fold_id': fid,
        'n_test' : int(test_mask.sum()),
        'spearman': float(rho),
    })

pf = pd.DataFrame(per_fold_rows).sort_values('fold_id').reset_index(drop=True)
pf['pct_data'] = 100.0 * pf['n_test'] / len(fm)
print(f'  {len(pf)} sub-basin folds.  '
      f'Mean Spearman = {pf["spearman"].mean():.3f}')

# ─────────────────────────────────────────────────────────────────────────
# FIG 1 — Cross-validation performance
# ─────────────────────────────────────────────────────────────────────────
print('\nFig 1: CV performance...')

ranking_models = ['cw_only', 'ridge', 'lightgbm']
model_display  = {'cw_only': 'Cw-only', 'ridge': 'Ridge', 'lightgbm': 'LightGBM'}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 1a — grouped bar: Spearman for target_max_log, sub-basin vs block CV
ax = axes[0]
cv_max = cv[
    (cv['target'] == 'target_max_log') &
    cv['model'].isin(ranking_models)
].copy()

x = np.arange(len(ranking_models))
w = 0.35
scheme_style = [('fold_subbasin', '#2E86AB', 'Sub-basin CV'),
                ('fold_block',    '#C8102E', 'Block CV')]

for i, (scheme, clr, lbl) in enumerate(scheme_style):
    sub = cv_max[cv_max['fold_scheme'] == scheme].set_index('model')
    vals = [sub.loc[m, 'spearman_mean'] if m in sub.index else np.nan
            for m in ranking_models]
    errs = [sub.loc[m, 'spearman_std']  if m in sub.index else np.nan
            for m in ranking_models]
    ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, label=lbl,
           color=clr, alpha=0.85, capsize=4, error_kw={'linewidth': 1.2})

ax.axhline(0.401, ls='--', lw=1.3, color='#555555',
           label='11-feat baseline (ρ = 0.401)')
ax.set_xticks(x)
ax.set_xticklabels([model_display[m] for m in ranking_models])
ax.set_ylabel('Spearman ρ')
ax.set_title('(a) Ranking performance — max flood depth')
ax.legend(fontsize=9)
ax.set_ylim(0, 0.65)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

# 1b — per-fold bar for Ridge sub-basin
ax = axes[1]
xpos = np.arange(len(pf))
bars = ax.bar(xpos, pf['spearman'], color='#2E86AB', alpha=0.85, zorder=3)
mean_rho = pf['spearman'].mean()
ax.axhline(mean_rho, ls='--', lw=1.3, color='#C8102E',
           label=f'Mean ρ = {mean_rho:.3f}', zorder=4)

# annotate largest fold
lrg = pf.loc[pf['n_test'].idxmax()]
lrg_x = xpos[pf['n_test'].idxmax()]
ax.annotate(
    f'{lrg.fold_id}\n({lrg.pct_data:.1f}% of data)',
    xy=(lrg_x, lrg.spearman),
    xytext=(lrg_x + 0.6, lrg.spearman + 0.07),
    fontsize=8,
    arrowprops=dict(arrowstyle='->', lw=0.9, color='#333333'),
)
ax.set_xticks(xpos)
ax.set_xticklabels([r.fold_id for _, r in pf.iterrows()], fontsize=8)
ax.set_xlabel('Sub-basin fold')
ax.set_ylabel('Spearman ρ')
ax.set_title('(b) Ridge — per-fold Spearman (sub-basin CV, max depth)')
ax.legend(fontsize=9)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

fig.tight_layout(w_pad=3)
save_fig(fig, 'fig1_cv_performance')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# FIG 2 — SHAP feature importance
# ─────────────────────────────────────────────────────────────────────────
print('Fig 2: SHAP importance...')

shap_cols = [f'shap_{f}' for f in FEATURES]
shap_mat  = arc[shap_cols].values           # (118119, 12)
feat_mat  = arc[FEATURES].values

mean_abs  = np.abs(shap_mat).mean(axis=0)
order     = np.argsort(mean_abs)            # ascending → bottom-to-top in plot

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 2a — mean |SHAP| horizontal bar
ax = axes[0]
y_pos = np.arange(len(order))
ax.barh(y_pos, mean_abs[order], color='#2E86AB', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([FEAT_LABELS[FEATURES[i]] for i in order])
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('(a) Mean absolute SHAP values (LGB, max depth)')
ax.xaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

# 2b — beeswarm-style scatter (5 k subsample)
ax = axes[1]
rng   = np.random.default_rng(42)
idx_s = rng.choice(len(arc), size=5000, replace=False)
sh_s  = shap_mat[idx_s]
ft_s  = feat_mat[idx_s]

for rank, feat_idx in enumerate(order):
    shap_v = sh_s[:, feat_idx]
    feat_v = ft_s[:, feat_idx]
    fv_n   = (feat_v - feat_v.min()) / (np.ptp(feat_v) + 1e-9)
    y_jit  = rank + rng.uniform(-0.38, 0.38, size=len(shap_v))
    ax.scatter(shap_v, y_jit, c=fv_n, cmap='RdBu_r',
               alpha=0.18, s=4, linewidths=0, rasterized=True)

ax.axvline(0, lw=0.9, color='#333333')
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([FEAT_LABELS[FEATURES[i]] for i in order])
ax.set_xlabel('SHAP value (impact on log max flood depth)')
ax.set_title('(b) SHAP value distribution (n = 5,000 sample)')

sm = ScalarMappable(norm=Normalize(0, 1), cmap='RdBu_r')
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
cb.ax.set_ylabel('Feature value\n(low → high)', fontsize=8)

fig.tight_layout(w_pad=3)
save_fig(fig, 'fig2_shap_importance')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# FIG 3 — Flood risk archetypes
# ─────────────────────────────────────────────────────────────────────────
print('Fig 3: Archetypes...')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 3a — SHAP quadrant: slope SHAP vs HAND_min SHAP
ax = axes[0]
rng   = np.random.default_rng(0)
idx_q = rng.choice(len(arc), size=8000, replace=False)
sub_q = arc.iloc[idx_q]

for arch in ARCH_ORDER:
    m = sub_q['archetype'] == arch
    n = m.sum()
    ax.scatter(sub_q.loc[m, 'shap_slope'],
               sub_q.loc[m, 'shap_HAND_min'],
               c=ARCH_COLORS[arch], label=f'{arch} (n={n:,})',
               alpha=0.25, s=7, linewidths=0, rasterized=True)

ax.axhline(0, lw=0.8, color='#888888', ls='--')
ax.axvline(0, lw=0.8, color='#888888', ls='--')
ax.set_xlabel('SHAP value — slope')
ax.set_ylabel('SHAP value — HAND$_{min}$')
ax.set_title('(a) Archetype separation in SHAP space (n = 8,000 sample)')
leg = ax.legend(fontsize=8, markerscale=2.5, handlelength=0.8,
                framealpha=0.9, edgecolor='#CCCCCC')

# 3b — spatial map
ax = axes[1]
try:
    gdf = gpd.read_file(RES_DIR / 'parcel_archetypes_v2.gpkg')
    for arch in reversed(ARCH_ORDER):  # Hotspot on top
        gdf[gdf['archetype'] == arch].plot(
            ax=ax, color=ARCH_COLORS[arch], linewidth=0, label=arch)
    ax.set_axis_off()
    ax.set_title('(b) Spatial distribution of archetypes — Brays Bayou')
    legend_patches = [mpatches.Patch(color=ARCH_COLORS[a], label=a)
                      for a in ARCH_ORDER]
    ax.legend(handles=legend_patches, fontsize=9, loc='lower right',
              framealpha=0.9, edgecolor='#CCCCCC')
except Exception as e:
    ax.text(0.5, 0.5, f'Map error:\n{e}', transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='gray')
    ax.set_title('(b) Spatial distribution of archetypes')

fig.tight_layout(w_pad=2)
save_fig(fig, 'fig3_archetypes')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# FIG 4 — ISA × Archetype heatmap
# ─────────────────────────────────────────────────────────────────────────
print('Fig 4: ISA x archetype heatmap...')

arc['isa_q'] = pd.qcut(
    arc['ISA_frac'], q=5,
    labels=['Q1\n(low ISA)', 'Q2', 'Q3', 'Q4', 'Q5\n(high ISA)'],
)
pivot = (arc.groupby(['archetype', 'isa_q'])['target_mean']
           .mean()
           .unstack('isa_q')
           .reindex(ARCH_ORDER))

fig, ax = plt.subplots(figsize=(7.5, 3.8))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(5))
ax.set_xticklabels(pivot.columns.tolist())
ax.set_yticks(range(4))
ax.set_yticklabels(ARCH_ORDER)
ax.set_xlabel('ISA quintile')
ax.set_title('Mean simulated flood depth (ft) by archetype and impervious surface fraction')

vmax = np.nanmax(pivot.values)
for i in range(4):
    for j in range(5):
        v = pivot.values[i, j]
        if not np.isnan(v):
            clr = 'white' if v > 0.6 * vmax else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, color=clr, fontweight='bold')

cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.ax.set_ylabel('Mean flood depth (ft)', fontsize=9)

# delta annotation on right side
deltas = pivot.values[:, -1] - pivot.values[:, 0]
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(4))
ax2.set_yticklabels([f'Δ = {d:+.2f} ft' for d in deltas], fontsize=8)
ax2.tick_params(right=False)
ax2.spines['top'].set_visible(False)

fig.tight_layout()
save_fig(fig, 'fig4_isa_archetype_heatmap')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# FIG 5 — Development counterfactual (+20% ISA)
# ─────────────────────────────────────────────────────────────────────────
print('Fig 5: Counterfactual...')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 5a — spatial map of class jumpers
ax = axes[0]
try:
    gdf = gpd.read_file(RES_DIR / 'parcel_archetypes_v2.gpkg')
    cf_geo = cf[['parcel_id', 'jumped_up', 'jumped_to_class3', 'delta_ridge_max']]
    gdf = gdf.merge(cf_geo, on='parcel_id', how='left')
    gdf['jumped_up']       = gdf['jumped_up'].fillna(False)
    gdf['jumped_to_class3'] = gdf['jumped_to_class3'].fillna(False)

    # layers: stable → jumped → jumped-to-3
    gdf[~gdf['jumped_up']].plot(ax=ax, color='#DDDDDD', linewidth=0)
    gdf[gdf['jumped_up'] & ~gdf['jumped_to_class3']].plot(
        ax=ax, color='#E87722', linewidth=0)
    gdf[gdf['jumped_to_class3']].plot(ax=ax, color='#C8102E', linewidth=0)

    ax.set_axis_off()
    ax.set_title('(a) Risk class transitions under +20% ISA scenario')
    n_jumped   = int(gdf['jumped_up'].sum())
    n_to3      = int(gdf['jumped_to_class3'].sum())
    pct_jumped = 100 * n_jumped / len(gdf)
    legend_patches = [
        mpatches.Patch(color='#DDDDDD', label=f'No change ({100-pct_jumped:.1f}%)'),
        mpatches.Patch(color='#E87722', label=f'Jumped ≥1 class ({pct_jumped:.1f}%)'),
        mpatches.Patch(color='#C8102E', label=f'Jumped to class 3 (n={n_to3})'),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc='lower right',
              framealpha=0.9, edgecolor='#CCCCCC')
except Exception as e:
    ax.text(0.5, 0.5, f'Map error:\n{e}', transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='gray')
    ax.set_title('(a) Risk class transitions (+20% ISA)')

# 5b — archetype vulnerability
ax = axes[1]
arch_s = (cf.groupby('archetype')
           .agg(n=('parcel_id', 'count'),
                n_jumped=('jumped_up', 'sum'),
                mean_delta=('delta_ridge_max', 'mean'))
           .reindex(ARCH_ORDER))
arch_s['pct_jumped'] = 100 * arch_s['n_jumped'] / arch_s['n']

x = np.arange(4)
bars = ax.bar(x, arch_s['pct_jumped'],
              color=[ARCH_COLORS[a] for a in ARCH_ORDER],
              alpha=0.90, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(ARCH_ORDER, rotation=15, ha='right')
ax.set_ylabel('Parcels jumping ≥1 risk class (%)')
ax.set_title('(b) Archetype vulnerability to +20% ISA development')
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

for bar, (arch, row) in zip(bars, arch_s.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f'{row.pct_jumped:.1f}%\nn={int(row.n_jumped):,}',
            ha='center', va='bottom', fontsize=8)

# secondary axis: mean delta risk
ax2 = ax.twinx()
ax2.plot(x, arch_s['mean_delta'], 'D--', color='#555555',
         ms=7, lw=1.4, label='Mean ΔRisk (Ridge)')
ax2.set_ylabel('Mean ΔRisk score', fontsize=9)
ax2.legend(fontsize=8, loc='upper right')
ax2.spines['top'].set_visible(False)

fig.tight_layout(w_pad=2)
save_fig(fig, 'fig5_counterfactual')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# FIG S1 — Intervention scenarios
# ─────────────────────────────────────────────────────────────────────────
print('Fig S1: Intervention scenarios...')

intv_order  = ['depave', 'permeable', 'bioswales']
intv_labels = {'depave': 'De-pave\nimpervious', 'permeable': 'Permeable\npaving', 'bioswales': 'Bioswales'}
group_style = [('hotspot', '#C8102E', 'Hotspot parcels (n=1,233)'),
               ('top10',   '#2E86AB', 'Top-10% risk parcels (n=11,811)')]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# S1a — mean delta risk
ax = axes[0]
x = np.arange(3)
w = 0.35
for i, (grp, clr, lbl) in enumerate(group_style):
    sub = iv[iv['group'] == grp].set_index('intervention')
    vals = [sub.loc[intv, 'mean_delta_risk'] for intv in intv_order]
    ax.bar(x + (i - 0.5) * w, vals, w, label=lbl, color=clr, alpha=0.85)

ax.axhline(0, lw=0.9, color='#333333')
ax.set_xticks(x)
ax.set_xticklabels([intv_labels[i] for i in intv_order])
ax.set_ylabel('Mean ΔRisk score (Ridge, max depth)')
ax.set_title('(a) Mean risk change under intervention')
ax.legend(fontsize=9)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax.text(0.02, 0.97,
        'Positive Δ reflects ISA-infrastructure\ncorrelation (see §4)',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=8, color='#666666', style='italic')

# S1b — efficiency (total_delta_sum per treated parcel)
ax = axes[1]
for i, (grp, clr, lbl) in enumerate(group_style):
    sub = iv[iv['group'] == grp].set_index('intervention')
    vals = [abs(sub.loc[intv, 'efficiency']) for intv in intv_order]
    ax.bar(x + (i - 0.5) * w, vals, w, label=lbl, color=clr, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([intv_labels[i] for i in intv_order])
ax.set_ylabel('|Mean ΔRisk| per treated parcel')
ax.set_title('(b) Intervention efficiency')
ax.legend(fontsize=9)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

fig.tight_layout(w_pad=3)
save_fig(fig, 'figS1_interventions')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# FIG S2 — Cluster stability
# ─────────────────────────────────────────────────────────────────────────
print('Fig S2: Cluster stability...')

k_values = [3, 4, 5, 6]
k_means  = []
for k in k_values:
    cols = [c for c in stab.columns if c.startswith(f'k={k} ')]
    rows = [r for r in stab.index   if r.startswith(f'k={k} ')]
    sub  = stab.loc[rows, cols].values.astype(float)
    # upper triangle off-diagonal → pairwise ARI within k
    tri = sub[np.triu_indices(len(rows), k=1)]
    k_means.append(float(np.mean(tri)) if len(tri) else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# S2a — mean within-k ARI
ax = axes[0]
ax.plot(k_values, k_means, 'o-', color='#2E86AB', lw=2.2, ms=9, zorder=4)
ax.axhline(1.0, ls='--', lw=0.9, color='#AAAAAA', zorder=3)
for k, v in zip(k_values, k_means):
    ax.text(k, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(k_values)
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Mean pairwise ARI (within k)')
ax.set_title('(a) Cluster stability across k values')
ax.set_ylim(0, 1.06)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

# vertical band for selected k=4
ax.axvspan(3.7, 4.3, alpha=0.12, color='#C8102E', zorder=2, label='Selected k = 4')
ax.legend(fontsize=9)

# S2b — full ARI heatmap
ax = axes[1]
run_labels = list(stab.index)
ari_mat    = stab.values.astype(float)
im = ax.imshow(ari_mat, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(12))
ax.set_xticklabels(run_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(12))
ax.set_yticklabels(run_labels, fontsize=7)
ax.set_title('(b) Full pairwise ARI matrix (k ∈ {3,4,5,6}, seeds {0,42,99})')

# draw k-block borders
borders = [0, 3, 6, 9, 12]
for b in borders:
    ax.axhline(b - 0.5, lw=1.2, color='#333333')
    ax.axvline(b - 0.5, lw=1.2, color='#333333')

cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cb.ax.set_ylabel('Adjusted Rand Index', fontsize=9)

fig.tight_layout(w_pad=3)
save_fig(fig, 'figS2_cluster_stability')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('DONE')
print(f'  PDF + PNG (300 DPI) : {FIG_DIR}')
print(f'  Preview  (150 DPI)  : {PRE_DIR}')
print('=' * 60)
