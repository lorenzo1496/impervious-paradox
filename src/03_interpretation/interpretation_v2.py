#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flood-risk Interpretation v2 -- 12 features (+ HAND_min)
=========================================================
Trains final Ridge + LGB (x2 targets), computes full-dataset LGB SHAP,
clusters parcels into 4 archetypes, stability test, 9 figures.
"""

import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import geopandas as gpd

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE / 'data' / 'processed'
RESULTS_DIR = BASE / 'outputs' / 'results'
FIG_DIR     = BASE / 'outputs' / 'figures' / 'interpretation_v2'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS (match modeling notebook exactly) ────────────────────────────────
RANDOM_STATE = 42
LGB_ROUNDS   = 500
SHAP_SAMPLE  = 5000      # for beeswarm figure
K_ARCHETYPES = 4

FEATURES = [
    'elevation', 'slope', 'TWI', 'log_flow_accum',
    'dist_to_stream', 'dist_to_street',
    'ISA_frac', 'log_lot_area', 'is_enclave',
    'conn_topo', 'Cw_topo', 'HAND_min',
]

LGB_PARAMS = {
    'objective': 'lambdarank', 'metric': 'ndcg',
    'ndcg_at': [10, 50, 100], 'learning_rate': 0.05,
    'num_leaves': 63, 'min_data_in_leaf': 50,
    'feature_fraction': 0.9, 'bagging_fraction': 0.9,
    'bagging_freq': 5, 'verbose': -1,
    'random_state': RANDOM_STATE, 'max_position': 1000,
}

ARCHETYPE_ORDER  = ['Hotspot', 'Lowland baseline', 'Upland baseline', 'Upland shield']
ARCHETYPE_COLORS = {
    'Hotspot':          '#CC3311',
    'Lowland baseline': '#EE9900',
    'Upland baseline':  '#4477AA',
    'Upland shield':    '#228833',
}

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                     'axes.titlesize': 12, 'axes.labelsize': 11})

print('=' * 65)
print('Flood-risk Interpretation v2  (12 features, +HAND_min)')
print('=' * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_DIR / 'feature_matrix.csv')
FEATURES = [f for f in FEATURES if f in df.columns]

X_all   = df[FEATURES].values
y_mean  = df['target_mean_log'].values
y_max   = df['target_max_log'].values

print(f'Parcels : {len(df):,}  |  Features: {len(FEATURES)}')
print(f'Features: {", ".join(FEATURES)}')


# ══════════════════════════════════════════════════════════════════════════════
# 2. Train 4 final models (matching baseline notebook)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '-' * 65)
print('Training 4 final models...')

scaler      = StandardScaler().fit(X_all)
X_scaled    = scaler.transform(X_all)

ridge_mean  = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_scaled, y_mean)
ridge_max   = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_scaled, y_max)
print('  Ridge (mean + max): done')


def rank_labels(y, n_bins=32):
    return pd.qcut(pd.Series(y), q=n_bins, labels=False,
                   duplicates='drop').fillna(0).astype(int).values


def chunk_groups(n, size=9000):
    g = [size] * (n // size)
    if n % size:
        g.append(n % size)
    return g


def train_lgb(X, y, label):
    print(f'  LGB ({label})...', end=' ', flush=True)
    t = time.time()
    d = lgb.Dataset(X, label=rank_labels(y),
                    group=chunk_groups(len(y)), feature_name=FEATURES)
    m = lgb.train(LGB_PARAMS, d, num_boost_round=LGB_ROUNDS)
    print(f'{time.time()-t:.0f}s')
    return m


lgb_mean = train_lgb(X_all, y_mean, 'target_mean_log')
lgb_max  = train_lgb(X_all, y_max,  'target_max_log')
print('All 4 models trained.')


# ══════════════════════════════════════════════════════════════════════════════
# 3. Full-dataset LGB SHAP (target_max_log) -- for KMeans archetypes
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '-' * 65)
print('Computing full-dataset SHAP (LGB target_max_log)...')
t0            = time.time()
expl_max      = shap.TreeExplainer(lgb_max)
shap_all_max  = expl_max.shap_values(X_all)   # (N, 12)

# Handle list/3-D output from lambdarank
if isinstance(shap_all_max, list):
    shap_all_max = shap_all_max[0]
elif shap_all_max.ndim == 3:
    shap_all_max = shap_all_max[:, :, 0]

print(f'  Full SHAP done: {shap_all_max.shape}  ({time.time()-t0:.0f}s)')

# 5000-sample for beeswarm figures
rng       = np.random.RandomState(RANDOM_STATE)
s_idx     = rng.choice(len(df), SHAP_SAMPLE, replace=False)
X_samp    = X_all[s_idx]
shap_samp = shap_all_max[s_idx]

# Also compute SHAP for LGB mean (sample only, for step1 comparison)
print('  Computing LGB-mean SHAP (sample)...', end=' ', flush=True)
shap_samp_mean = shap.TreeExplainer(lgb_mean).shap_values(X_samp)
if isinstance(shap_samp_mean, list):
    shap_samp_mean = shap_samp_mean[0]
elif shap_samp_mean.ndim == 3:
    shap_samp_mean = shap_samp_mean[:, :, 0]
print('done')


# ══════════════════════════════════════════════════════════════════════════════
# 4. KMeans k=4 on full SHAP matrix -- archetypes
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '-' * 65)
print(f'KMeans (k={K_ARCHETYPES}) on full SHAP matrix...')
t0 = time.time()
km = KMeans(n_clusters=K_ARCHETYPES, random_state=RANDOM_STATE, n_init=10)
km.fit(shap_all_max)
raw_labels = km.labels_
print(f'  KMeans done ({time.time()-t0:.0f}s)  inertia={km.inertia_:.1f}')


def assign_archetype_labels(centroids, feat_names):
    """Map cluster IDs to archetype names based on SHAP centroid profiles."""
    c       = pd.DataFrame(centroids, columns=feat_names)
    mapping = {}
    pool    = list(c.index)

    # Hotspot: highest slope + log_flow_accum SHAP (convergent, flood-prone)
    idx = (c['slope'] + c['log_flow_accum']).loc[pool].idxmax()
    mapping[idx] = 'Hotspot'
    pool = [i for i in pool if i != idx]

    # Upland shield: most negative elevation SHAP (strongly protected)
    idx = c.loc[pool, 'elevation'].idxmin()
    mapping[idx] = 'Upland shield'
    pool = [i for i in pool if i != idx]

    # Upland baseline: more negative elevation SHAP of remaining two
    idx = c.loc[pool, 'elevation'].idxmin()
    mapping[idx] = 'Upland baseline'
    pool = [i for i in pool if i != idx]

    # Lowland baseline: whatever remains
    mapping[pool[0]] = 'Lowland baseline'
    return mapping


label_map   = assign_archetype_labels(km.cluster_centers_, FEATURES)
df['archetype'] = pd.Series(raw_labels).map(label_map).values

# SHAP DataFrame for all parcels (for outputs + figures)
shap_df_all = pd.DataFrame(shap_all_max, columns=[f'shap_{f}' for f in FEATURES])
shap_df_all.index = df.index


# ══════════════════════════════════════════════════════════════════════════════
# 5. Archetype characteristics table
# ══════════════════════════════════════════════════════════════════════════════
feature_cols = FEATURES + ['target_mean', 'target_max']
overall_mean = df[feature_cols].mean()

char_rows = []
for arch in ARCHETYPE_ORDER:
    sub  = df[df['archetype'] == arch]
    row  = sub[feature_cols].mean().rename(arch)
    char_rows.append(row)

char_df = pd.concat(char_rows, axis=1).T
char_df.index.name = 'archetype'

# Ratio vs overall mean
ratio_df = char_df[FEATURES].div(overall_mean[FEATURES])
ratio_df.index.name = 'archetype'


# ══════════════════════════════════════════════════════════════════════════════
# 6. ISA quintile x archetype table
# ══════════════════════════════════════════════════════════════════════════════
df['isa_quintile'] = pd.qcut(df['ISA_frac'], 5,
                              labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
isa_pivot = df.pivot_table(
    values='target_mean',
    index='archetype',
    columns='isa_quintile',
    aggfunc='mean',
).reindex(ARCHETYPE_ORDER)
isa_delta = isa_pivot['Q5'] - isa_pivot['Q1']


# ══════════════════════════════════════════════════════════════════════════════
# 7. Cluster stability (k in {3,4,5,6} x seeds {0,42,99})
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '-' * 65)
print('Cluster stability test...')
K_VALS    = [3, 4, 5, 6]
SEEDS     = [0, 42, 99]
stab_runs = {}  # (k, seed) -> labels

for k in K_VALS:
    for seed in SEEDS:
        key = (k, seed)
        km_s = KMeans(n_clusters=k, random_state=seed, n_init=10)
        stab_runs[key] = km_s.fit_predict(shap_all_max)
        print(f'  k={k} seed={seed}: done', flush=True)

# Pairwise ARI matrix across all 12 runs
run_keys  = [(k, s) for k in K_VALS for s in SEEDS]
run_labels = [f'k={k} s={s}' for k, s in run_keys]
n_runs    = len(run_keys)
ari_matrix = np.ones((n_runs, n_runs))

for i, ki in enumerate(run_keys):
    for j, kj in enumerate(run_keys):
        if i != j:
            ari_matrix[i, j] = adjusted_rand_score(
                stab_runs[ki], stab_runs[kj]
            )

stab_df = pd.DataFrame(ari_matrix, index=run_labels, columns=run_labels)

# Mean ARI per k (across all pairs of seeds within same k)
mean_ari_per_k = {}
for k in K_VALS:
    k_runs = [(k, s) for s in SEEDS]
    pairs  = list(combinations(range(len(k_runs)), 2))
    vals   = [adjusted_rand_score(stab_runs[k_runs[a]], stab_runs[k_runs[b]])
              for a, b in pairs]
    mean_ari_per_k[k] = np.mean(vals)

# Reference: ARI of all other runs vs (k=4, seed=42)
ref_labels = stab_runs[(4, 42)]


# ══════════════════════════════════════════════════════════════════════════════
# 8. Ridge interpretation (for step2)
# ══════════════════════════════════════════════════════════════════════════════
ridge_coef_df = pd.DataFrame({
    'feature':   FEATURES,
    'coef_mean': ridge_mean.coef_,
    'coef_max':  ridge_max.coef_,
})
ridge_coef_df['abs_mean'] = ridge_coef_df['coef_mean'].abs()


# ══════════════════════════════════════════════════════════════════════════════
# 9. Save outputs
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '-' * 65)
print('Saving outputs...')

# parcel_archetypes_v2.csv
out_cols = ['parcel_id', 'archetype'] + FEATURES + ['target_mean', 'target_max',
             'target_mean_log', 'target_max_log']
parcels_out = df[out_cols].copy()
for f in FEATURES:
    parcels_out[f'shap_{f}'] = shap_all_max[:, FEATURES.index(f)]
parcels_out.to_csv(RESULTS_DIR / 'parcel_archetypes_v2.csv', index=False)
print(f'  Saved parcel_archetypes_v2.csv  ({len(parcels_out):,} rows)')

# parcel_archetypes_v2.gpkg
gpkg_path = DATA_DIR / 'feature_matrix.gpkg'
if gpkg_path.exists():
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf.merge(
        parcels_out[['parcel_id', 'archetype'] + [f'shap_{f}' for f in FEATURES]],
        on='parcel_id', how='left',
    )
    if gdf.crs is None or gdf.crs.to_epsg() != 26915:
        gdf = gdf.to_crs(epsg=26915)
    gdf.to_file(RESULTS_DIR / 'parcel_archetypes_v2.gpkg', driver='GPKG')
    print(f'  Saved parcel_archetypes_v2.gpkg')
else:
    print(f'  [SKIP] gpkg not found at {gpkg_path}')
    gdf = None

# archetype_characteristics_v2.csv
char_out = char_df.copy()
for f in FEATURES:
    char_out[f'ratio_{f}'] = ratio_df[f]
char_out.to_csv(RESULTS_DIR / 'archetype_characteristics_v2.csv')
print(f'  Saved archetype_characteristics_v2.csv')

# cluster_stability_v2.csv
stab_df.to_csv(RESULTS_DIR / 'cluster_stability_v2.csv')
print(f'  Saved cluster_stability_v2.csv')


# ══════════════════════════════════════════════════════════════════════════════
# 10. Terminal prints
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('ARCHETYPE SIZES')
print('=' * 65)
total = len(df)
for arch in ARCHETYPE_ORDER:
    n = (df['archetype'] == arch).sum()
    print(f'  {arch:20s}: {n:6,}  ({n/total*100:.1f}%)')

print('\n' + '=' * 65)
print('TABLE 1 -- Hotspot vs Overall Mean (feature ratios, all 12 features)')
print('=' * 65)
hs_means   = char_df.loc['Hotspot', FEATURES]
ov_means   = overall_mean[FEATURES]
lb_means   = char_df.loc['Lowland baseline', FEATURES]
print(f'{"Feature":22s}  {"Hotspot":>10s}  {"Overall":>10s}  {"Ratio HS/Ov":>12s}  {"Ratio HS/LB":>12s}')
print('-' * 72)
for f in FEATURES:
    ov  = ov_means[f]
    hs  = hs_means[f]
    lb  = lb_means[f]
    r1  = hs / ov  if abs(ov) > 1e-9 else float('nan')
    r2  = hs / lb  if abs(lb) > 1e-9 else float('nan')
    print(f'  {f:20s}  {hs:10.3f}  {ov:10.3f}  {r1:12.3f}  {r2:12.3f}')

print('\n' + '=' * 65)
print('ISA QUINTILE x ARCHETYPE  (mean target_mean flood depth, ft)')
print('=' * 65)
print(isa_pivot.round(3).to_string())
print('\nQ5 - Q1 delta per archetype:')
for arch in ARCHETYPE_ORDER:
    if arch in isa_delta.index:
        print(f'  {arch:20s}: {isa_delta[arch]:+.3f} ft')

print('\n' + '=' * 65)
print('CLUSTER STABILITY  (mean ARI within k, across seed pairs)')
print('=' * 65)
for k, ari in mean_ari_per_k.items():
    print(f'  k={k}: mean ARI = {ari:.3f}')

print('\n' + '=' * 65)
print('HAND_min SHAP PATTERN CHECK')
print('=' * 65)
hand_idx = FEATURES.index('HAND_min')
for arch in ARCHETYPE_ORDER:
    mask      = df['archetype'] == arch
    mean_hand = df.loc[mask, 'HAND_min'].mean()
    mean_shap = shap_all_max[mask.values, hand_idx].mean()
    print(f'  {arch:20s}:  HAND_min = {mean_hand:.2f} m  |  SHAP = {mean_shap:+.4f}')

hs_mask    = df['archetype'] == 'Hotspot'
hs_shap    = shap_all_max[hs_mask.values, hand_idx].mean()
expected   = 'POSITIVE (low HAND_min drives high risk)'
observed   = 'POSITIVE' if hs_shap > 0 else 'NEGATIVE'
match_note = 'NOTE: Hotspot SHAP positive = low HAND_min inflating risk (protective when HIGH)'
print(f'\n  Expected in Hotspot: {expected}')
print(f'  Observed: {observed} ({hs_shap:+.4f})')
print(f'  {match_note}')


# ══════════════════════════════════════════════════════════════════════════════
# 11. Figures
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '-' * 65)
print('Generating figures...')


# ── step1: SHAP beeswarm LGB target_max_log ───────────────────────────────────
def step1_shap_summary_lgb():
    shap.summary_plot(
        shap_samp, X_samp, feature_names=FEATURES,
        plot_type='dot', show=False,
        plot_size=(10, 7), max_display=len(FEATURES),
    )
    plt.gcf().axes[0].set_title(
        'Step 1 -- SHAP Beeswarm: LightGBM-LambdaRank\n'
        f'target_max_log  |  n={SHAP_SAMPLE:,} sample', fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step1_shap_summary_lgb.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step1_shap_summary_lgb.png')


# ── step2: Ridge coefficient bar (HAND_min highlighted) ──────────────────────
def step2_shap_summary_ridge():
    rdf = ridge_coef_df.sort_values('abs_mean')
    colors_mean = ['#EE4444' if f == 'HAND_min' else '#4477AA' for f in rdf['feature']]
    colors_max  = ['#EE4444' if f == 'HAND_min' else '#CC6677' for f in rdf['feature']]

    fig, ax = plt.subplots(figsize=(10, 7))
    y   = np.arange(len(rdf))
    w   = 0.38
    ax.barh(y - w/2, rdf['coef_mean'], height=w,
            color=colors_mean, alpha=0.85, edgecolor='white',
            label='target_mean_log')
    ax.barh(y + w/2, rdf['coef_max'], height=w,
            color=colors_max, alpha=0.85, edgecolor='white',
            label='target_max_log')
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(rdf['feature'], fontsize=10)
    ax.set_xlabel('Standardised coefficient (Ridge, alpha=1.0)')
    ax.set_title('Step 2 -- Ridge Coefficients: Both Targets\n'
                 'Sorted by |coef|  |  HAND_min highlighted in red', fontsize=12)
    ax.legend(framealpha=0.9)

    patches = [
        mpatches.Patch(color='#EE4444', label='HAND_min (new)'),
        mpatches.Patch(color='#4477AA', label='Other (mean target)'),
        mpatches.Patch(color='#CC6677', label='Other (max target)'),
    ]
    ax.legend(handles=patches, framealpha=0.9, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step2_shap_summary_ridge.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step2_shap_summary_ridge.png')


# ── step3: KMeans elbow (k=2..10), mark k=4 ──────────────────────────────────
def step3_kmeans_elbow():
    k_range  = range(2, 11)
    inertias = []
    print('  Elbow plot fits...', end=' ', flush=True)
    for k in k_range:
        km_e = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        km_e.fit(shap_all_max)
        inertias.append(km_e.inertia_)
        print(k, end=' ', flush=True)
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, 'o-', color='#3B4992', linewidth=2, markersize=7)
    ax.axvline(K_ARCHETYPES, color='#CC3311', linestyle='--', linewidth=1.5,
               label=f'k={K_ARCHETYPES} (selected)')
    ax.set_xlabel('Number of clusters k')
    ax.set_ylabel('Within-cluster sum of squares (inertia)')
    ax.set_title('Step 3 -- KMeans Elbow Plot on SHAP Matrix\n'
                 'Full dataset LGB SHAP (target_max_log)', fontsize=12)
    ax.legend(framealpha=0.9)
    ax.set_xticks(list(k_range))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step3_kmeans_elbow.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step3_kmeans_elbow.png')


# ── step4: archetype SHAP quadrant scatter ────────────────────────────────────
def step4_archetype_shap_quadrant():
    slope_idx = FEATURES.index('slope')
    elev_idx  = FEATURES.index('elevation')

    # Use sample for scatter (all 118k is too dense to render)
    x_vals  = shap_samp[:, slope_idx]
    y_vals  = -shap_samp[:, elev_idx]   # negate so upland=positive y
    archs   = df.iloc[s_idx]['archetype'].values

    # Cluster centroids in the same space
    cx = km.cluster_centers_[:, slope_idx]
    cy = -km.cluster_centers_[:, elev_idx]
    c_labels = [label_map[i] for i in range(K_ARCHETYPES)]

    fig, ax = plt.subplots(figsize=(9, 7))
    for arch in ARCHETYPE_ORDER:
        mask = archs == arch
        ax.scatter(x_vals[mask], y_vals[mask], alpha=0.25, s=8,
                   c=ARCHETYPE_COLORS[arch], label=arch, rasterized=True)

    for i, lbl in enumerate(c_labels):
        ax.scatter(cx[i], cy[i], marker='*', s=350,
                   c=ARCHETYPE_COLORS[lbl], edgecolors='black',
                   linewidths=0.8, zorder=5)
        ax.annotate(lbl, (cx[i], cy[i]),
                    textcoords='offset points', xytext=(7, 5),
                    fontsize=9, fontweight='bold',
                    color=ARCHETYPE_COLORS[lbl])

    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.6)
    ax.axvline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.6)
    ax.set_xlabel('Slope SHAP  (positive = greater exposure  -->)')
    ax.set_ylabel('-Elevation SHAP  (positive = upland protection  ^)')
    ax.set_title('Step 4 -- Archetype SHAP Quadrant\n'
                 'slope SHAP (exposure) x -elevation SHAP (protection)\n'
                 'Stars = cluster centroids', fontsize=11)

    handles = [mpatches.Patch(color=ARCHETYPE_COLORS[a], label=a)
               for a in ARCHETYPE_ORDER]
    ax.legend(handles=handles, framealpha=0.9, markerscale=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step4_archetype_shap_quadrant.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step4_archetype_shap_quadrant.png')


# ── step5: archetype map ──────────────────────────────────────────────────────
def step5_archetype_map():
    if gdf is None:
        print('  [SKIP] step5: no gpkg')
        return

    fig, ax = plt.subplots(figsize=(13, 10))
    for arch in ARCHETYPE_ORDER:
        sub = gdf[gdf['archetype'] == arch]
        sub.plot(ax=ax, color=ARCHETYPE_COLORS[arch], linewidth=0,
                 label=arch, alpha=0.85)

    handles = [mpatches.Patch(color=ARCHETYPE_COLORS[a], label=a)
               for a in ARCHETYPE_ORDER]
    ax.legend(handles=handles, framealpha=0.9, loc='lower right', fontsize=10,
              title='Archetype')
    ax.set_title('Step 5 -- Parcel Archetypes (Geographic)\n'
                 'k=4 KMeans on full-dataset LGB SHAP (target_max_log)  |  EPSG:26915',
                 fontsize=12)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step5_archetype_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step5_archetype_map.png')


# ── step6: archetype characteristics (ratio vs overall mean) ─────────────────
def step6_archetype_characteristics():
    # Exclude binary/bounded features from ratio; use z-score instead
    plot_features = [f for f in FEATURES if f != 'is_enclave']
    ratio_vals    = ratio_df[plot_features]

    n_feat = len(plot_features)
    y      = np.arange(n_feat)
    w      = 0.18

    fig, ax = plt.subplots(figsize=(11, 9))
    offsets = np.linspace(-(len(ARCHETYPE_ORDER)-1)/2,
                           (len(ARCHETYPE_ORDER)-1)/2,
                           len(ARCHETYPE_ORDER)) * w

    for i, arch in enumerate(ARCHETYPE_ORDER):
        vals = ratio_vals.loc[arch, plot_features].values
        ax.barh(y + offsets[i], vals, height=w,
                color=ARCHETYPE_COLORS[arch], alpha=0.85,
                edgecolor='white', label=arch)

    ax.axvline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7,
               label='Overall mean (ratio = 1)')
    ax.set_xscale('symlog', linthresh=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_features, fontsize=9.5)
    ax.set_xlabel('Archetype mean / overall mean  (symlog scale)')
    ax.set_title('Step 6 -- Archetype Feature Profiles\n'
                 'Ratio vs overall mean  (symlog x-axis;  >1 = above average)',
                 fontsize=12)
    ax.legend(framealpha=0.9, fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step6_archetype_characteristics.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step6_archetype_characteristics.png')


# ── step7: ISA quintile x archetype heatmap ───────────────────────────────────
def step7_isa_quintile_heatmap():
    fig, ax = plt.subplots(figsize=(9, 5))
    vmax = isa_pivot.max().max()
    sns.heatmap(
        isa_pivot.round(3), annot=True, fmt='.3f',
        cmap='YlOrRd', vmin=0, vmax=vmax,
        cbar_kws={'label': 'Mean flood depth (ft)', 'shrink': 0.8},
        linewidths=0.5, ax=ax,
    )

    # Annotate Q5-Q1 delta to the right of each row
    for row_idx, arch in enumerate(ARCHETYPE_ORDER):
        if arch in isa_delta.index:
            ax.text(5.15, row_idx + 0.5,
                    f'd={isa_delta[arch]:+.3f}',
                    va='center', ha='left', fontsize=9,
                    color='#CC3311' if isa_delta[arch] > 0 else '#228833',
                    fontweight='bold')

    ax.set_title('Step 7 -- ISA Quintile x Archetype\n'
                 'Mean target_mean flood depth (ft)  |  d = Q5 - Q1 delta',
                 fontsize=12)
    ax.set_xlabel('ISA quintile (Q1=lowest imperviousness, Q5=highest)')
    ax.set_ylabel('Archetype')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step7_isa_quintile_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step7_isa_quintile_heatmap.png')


# ── step8: flood depth violin by archetype ────────────────────────────────────
def step8_flood_depth_by_archetype():
    plot_df = df[['archetype', 'target_mean']].copy()
    cap     = df['target_mean'].quantile(0.97)
    plot_df['target_mean'] = plot_df['target_mean'].clip(upper=cap)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {a: ARCHETYPE_COLORS[a] for a in ARCHETYPE_ORDER}
    sns.violinplot(
        data=plot_df, x='archetype', y='target_mean',
        order=ARCHETYPE_ORDER, palette=palette,
        inner='quartile', linewidth=1.2, ax=ax,
    )
    ax.set_xlabel('Archetype')
    ax.set_ylabel('Harvey flood depth (ft)  [capped at 97th pct]')
    ax.set_title('Step 8 -- Flood Depth Distribution by Archetype\n'
                 'target_mean (raw ft)  |  violins show full distribution + quartiles',
                 fontsize=12)

    # Annotate median per archetype
    for i, arch in enumerate(ARCHETYPE_ORDER):
        med = df.loc[df['archetype'] == arch, 'target_mean'].median()
        ax.text(i, med + 0.05, f'med={med:.2f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step8_flood_depth_by_archetype.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step8_flood_depth_by_archetype.png')


# ── step9: cluster stability heatmap ─────────────────────────────────────────
def step9_cluster_stability():
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(len(run_labels), dtype=bool)
    sns.heatmap(
        stab_df, annot=True, fmt='.2f',
        cmap='RdYlGn', vmin=0, vmax=1,
        mask=mask,
        cbar_kws={'label': 'Adjusted Rand Index', 'shrink': 0.7},
        linewidths=0.4, linecolor='white',
        annot_kws={'size': 8}, ax=ax,
    )
    # Diagonal = 1.0 (self)
    for i in range(len(run_labels)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                   color='#dddddd', lw=0))
        ax.text(i + 0.5, i + 0.5, '1.00', ha='center', va='center',
                fontsize=7.5, color='grey')

    # Highlight k=4 block
    k4_start = K_VALS.index(4) * len(SEEDS)
    ax.add_patch(plt.Rectangle((k4_start, k4_start), len(SEEDS), len(SEEDS),
                                fill=False, edgecolor='#3B4992', lw=2.5))

    ax.set_title('Step 9 -- Cluster Stability: Pairwise ARI\n'
                 'k in {3,4,5,6} x seeds {0,42,99}  |  blue box = k=4 (selected)',
                 fontsize=12)
    ax.set_xlabel('KMeans run  (k / seed)')
    ax.set_ylabel('KMeans run  (k / seed)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step9_cluster_stability.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  [OK] step9_cluster_stability.png')


# ── run all ───────────────────────────────────────────────────────────────────
step1_shap_summary_lgb()
step2_shap_summary_ridge()
step3_kmeans_elbow()
step4_archetype_shap_quadrant()
step5_archetype_map()
step6_archetype_characteristics()
step7_isa_quintile_heatmap()
step8_flood_depth_by_archetype()
step9_cluster_stability()

print('\n' + '=' * 65)
print('DONE')
print(f'  Results : {RESULTS_DIR}/')
print(f'  Figures : {FIG_DIR}/')
print('=' * 65)
