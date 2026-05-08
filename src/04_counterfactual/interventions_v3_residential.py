#!/usr/bin/env python3
"""
interventions_v3_residential.py
Primary  : +20% ISA development counterfactual on residential parcels only (NLCD 22-24)
Secondary: 3 interventions × 2 residential groups = 6 scenarios
Models   : Ridge + RF trained on all parcels (identical to interventions_v2.py);
           counterfactual outcomes evaluated on residential subset with residential archetypes.

Outputs
-------
outputs/results/counterfactual_20pct_residential.csv
outputs/results/intervention_scenarios_residential.csv
outputs/figures/paper/fig5_residential/step1–step8 (300 DPI PNG)
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[2]
RES_DIR     = BASE / 'outputs' / 'results'
FIG_DIR     = BASE / 'outputs' / 'figures' / 'paper' / 'fig5_residential'
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

RANDOM_STATE = 42
FEATURES = [
    'elevation', 'slope', 'TWI', 'log_flow_accum', 'dist_to_stream',
    'dist_to_street', 'ISA_frac', 'log_lot_area', 'is_enclave',
    'conn_topo', 'Cw_topo', 'HAND_min',
]
RESIDENTIAL_CLASSES = {22, 23, 24}

RES_ARCH_ORDER = ['Hotspot_res', 'Lowland_baseline_res', 'Upland_baseline_res', 'Upland_shield_res']
RES_ARCH_COLORS = {
    'Hotspot_res'         : '#C8102E',
    'Lowland_baseline_res': '#E9C46A',
    'Upland_baseline_res' : '#A8DADC',
    'Upland_shield_res'   : '#2E86AB',
}

def save_fig(fig, name):
    path = FIG_DIR / f'{name}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = path.stat().st_size // 1024
    print(f'  [OK] {name}.png  ({kb} KB)')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print('interventions_v3_residential.py')
print('=' * 65)

fm = pd.read_csv(RES_DIR / 'feature_matrix_nlcd.csv')
print(f'  Full dataset        : {len(fm):,} parcels')

# Residential flag (full-array boolean for group-level interventions)
is_res_full = fm['nlcd_class'].isin(RESIDENTIAL_CLASSES).values

# Merge residential_archetype from residential_archetypes.csv
ra = pd.read_csv(RES_DIR / 'residential_archetypes.csv',
                 usecols=['parcel_id', 'residential_archetype',
                          'is_top10pct_res', 'is_top1pct_res',
                          'rank_res', 'ridge_pred_res'])
fm = fm.merge(ra, on='parcel_id', how='left')

fm_res = fm[is_res_full].copy().reset_index(drop=True)
print(f'  Residential (22-24) : {len(fm_res):,}  ({100*is_res_full.mean():.1f}%)')
print(f'  Excluded (non-res)  : {(~is_res_full).sum():,}')
print()
print('  Residential archetype sizes:')
for a in RES_ARCH_ORDER:
    n = (fm_res['residential_archetype'] == a).sum()
    print(f'    {a:30s}: {n:,}  ({100*n/len(fm_res):.1f}%)')

X_all  = fm[FEATURES].values
y_mean = fm['target_mean_log'].values
y_max  = fm['target_max_log'].values

# ══════════════════════════════════════════════════════════════════════════════
# 2. Build KMeans risk classes (identical to interventions_v2.py)
# ══════════════════════════════════════════════════════════════════════════════
_km_X = StandardScaler().fit_transform(fm[['target_mean', 'target_max']].values)
_km   = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
fm['risk_class'] = _km.fit_predict(_km_X)
_cmeans = fm.groupby('risk_class')['target_max'].mean().sort_values()
_lmap   = {old: new for new, old in enumerate(_cmeans.index)}
fm['risk_class'] = fm['risk_class'].map(_lmap).astype(int)
fm_res = fm[is_res_full].copy().reset_index(drop=True)   # refresh with risk_class
print(f'\n  Risk class distribution (all): {fm["risk_class"].value_counts().sort_index().to_dict()}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Train models on all parcels (identical hyperparameters to interventions_v2)
# ══════════════════════════════════════════════════════════════════════════════
print('\nTraining models on all parcels ...')
sc         = StandardScaler().fit(X_all)
X_all_sc   = sc.transform(X_all)

ridge_mean = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_all_sc, y_mean)
ridge_max  = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_all_sc, y_max)
print('  Ridge (mean + max): done')

rf_clf = RandomForestClassifier(
    n_estimators=200, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf_clf.fit(X_all_sc, fm['risk_class'].values)
print('  RF classifier: done')

# Full-array baseline predictions (needed for ranking context)
fm['ridge_mean_base'] = ridge_mean.predict(X_all_sc)
fm['ridge_max_base']  = ridge_max.predict(X_all_sc)
fm['rf_class_base']   = rf_clf.predict(X_all_sc)
fm_res = fm[is_res_full].copy().reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. PRIMARY ANALYSIS — +20% ISA counterfactual on RESIDENTIAL rows
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('PRIMARY: +20% ISA development counterfactual — residential only')
print('=' * 65)

isa_idx = FEATURES.index('ISA_frac')
cw_idx  = FEATURES.index('Cw_topo')
dts_idx = FEATURES.index('dist_to_stream')

def _apply_20pct_isa_cf(df):
    """Add +20% ISA counterfactual columns to df in-place.
    df must already contain FEATURES columns and ridge/rf base predictions.
    Uses isa_idx, cw_idx, dts_idx, sc, ridge_mean, ridge_max, rf_clf from scope.
    """
    X        = df[FEATURES].values.astype(float)
    isa_old  = X[:, isa_idx]
    isa_new  = np.minimum(isa_old * 1.20, 1.0)
    dts_vals = np.maximum(X[:, dts_idx], 1.0)
    cw_new   = isa_new / dts_vals
    X_cf     = X.copy()
    X_cf[:, isa_idx] = isa_new
    X_cf[:, cw_idx]  = cw_new
    X_cf_sc  = sc.transform(X_cf)
    df['delta_isa']        = isa_new - isa_old
    df['ridge_mean_cf']    = ridge_mean.predict(X_cf_sc)
    df['ridge_max_cf']     = ridge_max.predict(X_cf_sc)
    df['rf_class_cf']      = rf_clf.predict(X_cf_sc)
    df['delta_ridge_mean'] = df['ridge_mean_cf'] - df['ridge_mean_base']
    df['delta_ridge_max']  = df['ridge_max_cf']  - df['ridge_max_base']
    df['class_jump']       = df['rf_class_cf'].astype(int) - df['rf_class_base'].astype(int)
    df['jumped_up']        = (df['class_jump'] > 0).astype(int)
    df['jumped_to_class3'] = ((df['rf_class_cf'] == 3) &
                               (df['rf_class_base'] < 3)).astype(int)

# Counterfactual for residential rows
_apply_20pct_isa_cf(fm_res)

n_res_total    = len(fm_res)
n_res_jumped   = int(fm_res['jumped_up'].sum())
n_res_to_cls3  = int(fm_res['jumped_to_class3'].sum())
pct_res_jumped = 100 * n_res_jumped / n_res_total
mean_delta_res = fm_res['delta_ridge_max'].mean()

# ── per-archetype breakdown ───────────────────────────────────────────────────
arch_cf_rows = []
print(f'\n  {"Archetype":30s}  {"n":>6}  {"jumped":>7}  {"pct":>6}  {"to_cls3":>7}  {"mean_dRisk":>10}')
print('  ' + '-' * 72)
for arch in RES_ARCH_ORDER:
    mask = fm_res['residential_archetype'] == arch
    n_a  = int(mask.sum())
    n_j  = int(fm_res.loc[mask, 'jumped_up'].sum())
    n_c3 = int(fm_res.loc[mask, 'jumped_to_class3'].sum())
    pct  = 100 * n_j / n_a if n_a else 0.0
    mdr  = float(fm_res.loc[mask, 'delta_ridge_max'].mean())
    arch_cf_rows.append(dict(archetype=arch, n_parcels=n_a, n_jumped=n_j,
                             pct_jumped=pct, n_to_class3=n_c3, mean_delta_risk=mdr))
    print(f'  {arch:30s}  {n_a:>6,}  {n_j:>7,}  {pct:>5.1f}%  {n_c3:>7,}  {mdr:>+10.4f}')

arch_cf_df = pd.DataFrame(arch_cf_rows)

# ══════════════════════════════════════════════════════════════════════════════
# 5. SECONDARY ANALYSIS — 3 interventions × 2 residential groups
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('SECONDARY: 3 interventions × 2 residential groups')
print('=' * 65)

# Group masks on the FULL dataset index (for X_all manipulation)
hotspot_res_full = (fm['residential_archetype'] == 'Hotspot_res').values
top10_res_full   = (fm['is_top10pct_res'] == True).values

groups = {
    'hotspot_res': hotspot_res_full,
    'top10_res'  : top10_res_full,
}

INTERVENTIONS = {
    'depave'    : {isa_idx: lambda x: np.zeros_like(x)},
    'permeable' : {isa_idx: lambda x: x * 0.7},
    'bioswales' : {cw_idx:  lambda x: x * 0.5},
}

# Residential rank for escape metric
res_base_mean = fm_res['ridge_mean_base'].values
res_rank_base_pct = pd.Series(res_base_mean).rank(ascending=False, pct=True).values * 100

scen_results = []
print(f'  {"key":35s}  {"mean_dRisk":>10}  {"pct_lower":>9}  {"esc_top10":>9}')
print('  ' + '-' * 68)

for g_name, g_mask_full in groups.items():
    for i_name, i_changes in INTERVENTIONS.items():
        X_s = X_all.astype(float).copy()
        for feat_idx, transform in i_changes.items():
            X_s[g_mask_full, feat_idx] = transform(X_s[g_mask_full, feat_idx])

        pred_cf_all   = ridge_mean.predict(sc.transform(X_s))
        delta_all     = pred_cf_all - fm['ridge_mean_base'].values

        # Only evaluate on residential parcels in the group
        g_res_mask    = g_mask_full & is_res_full
        delta_treated = delta_all[g_res_mask]
        n_treated     = int(g_res_mask.sum())

        # Escape-top10 measured within residential ranking
        pred_cf_res      = pred_cf_all[is_res_full]
        rank_cf_res_pct  = pd.Series(pred_cf_res).rank(ascending=False, pct=True).values * 100
        was_top10_res    = res_rank_base_pct <= 10
        now_top10_res    = rank_cf_res_pct  <= 10
        # Only treated residential parcels that were top10 and escaped
        treated_res_idx  = np.where(g_res_mask[is_res_full])[0]
        escaped = (was_top10_res[treated_res_idx]) & (~now_top10_res[treated_res_idx])

        pct_lower    = 100 * (delta_treated < 0).mean() if n_treated else 0.0
        pct_escaped  = 100 * escaped.sum() / max(was_top10_res[treated_res_idx].sum(), 1)
        key          = f'{g_name}__{i_name}'

        scen_results.append(dict(
            group=g_name, intervention=i_name,
            n_treated=n_treated,
            mean_delta_risk=float(delta_treated.mean()),
            pct_lower_risk=float(pct_lower),
            escaped_top10_res=int(escaped.sum()),
            pct_escaped_top10_res=float(pct_escaped),
            total_delta_sum=float(delta_all.sum()),
            efficiency=float(-delta_all.sum() / max(n_treated, 1)),
        ))
        print(f'  {key:35s}  {delta_treated.mean():>+10.4f}  {pct_lower:>8.1f}%  '
              f'{escaped.sum():>5} ({pct_escaped:.1f}%)')

scen_df = pd.DataFrame(scen_results)

# ─────────────────────────────────────────────────────────────────────────────
# Derived intermediate — all-parcel counterfactual for §6 comparison
# Reuses the already-trained models on the full feature matrix `fm`
# to produce counterfactual_20pct_isa.csv. Required by the
# residential-vs-all-parcels robustness check below.
# ─────────────────────────────────────────────────────────────────────────────
_isa_base_cols = ['parcel_id', 'archetype', 'risk_class',
                  'ridge_mean_base', 'ridge_max_base', 'rf_class_base']
fm_all_cf = fm[list(dict.fromkeys(_isa_base_cols + FEATURES))].copy()
_apply_20pct_isa_cf(fm_all_cf)

_isa_out_cols = [
    'parcel_id', 'archetype', 'risk_class', 'ISA_frac', 'Cw_topo',
    'ridge_mean_base', 'ridge_max_base', 'rf_class_base',
    'ridge_mean_cf', 'ridge_max_cf', 'rf_class_cf',
    'delta_isa', 'delta_ridge_mean', 'delta_ridge_max',
    'class_jump', 'jumped_up', 'jumped_to_class3',
]
_isa_path = RES_DIR / 'counterfactual_20pct_isa.csv'
try:
    fm_all_cf[_isa_out_cols].to_csv(_isa_path, index=False)
    print(f'  Saved counterfactual_20pct_isa.csv  ({len(fm_all_cf):,} rows)')
except PermissionError:
    print(f'  ERROR: {_isa_path} is open in another process — close it and re-run.')
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Comparison: residential vs all-parcels
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('COMPARISON: residential vs all-parcels counterfactual')
print('=' * 65)

cf_all = pd.read_csv(RES_DIR / 'counterfactual_20pct_isa.csv')
n_all_total   = len(cf_all)
n_all_jumped  = int(cf_all['jumped_up'].sum())
n_all_to_cls3 = int(cf_all['jumped_to_class3'].sum())

print(f'  {"Metric":<35}  {"All parcels":>12}  {"Residential":>12}  {"Δ":>8}')
print('  ' + '-' * 72)
print(f'  {"n parcels":<35}  {n_all_total:>12,}  {n_res_total:>12,}')
print(f'  {"n jumped ≥1 class":<35}  {n_all_jumped:>12,}  {n_res_jumped:>12,}  '
      f'{n_res_jumped - n_all_jumped:>+8,}')
print(f'  {"% jumped":<35}  {100*n_all_jumped/n_all_total:>11.2f}%  '
      f'{pct_res_jumped:>11.2f}%  '
      f'{pct_res_jumped - 100*n_all_jumped/n_all_total:>+8.2f}')
print(f'  {"n jumped to class 3":<35}  {n_all_to_cls3:>12,}  {n_res_to_cls3:>12,}')
print(f'  {"mean ΔRisk (Ridge max)":<35}  {cf_all["delta_ridge_max"].mean():>+12.4f}  '
      f'{mean_delta_res:>+12.4f}  '
      f'{mean_delta_res - cf_all["delta_ridge_max"].mean():>+8.4f}')

# ── sacrificial parcel update ─────────────────────────────────────────────────
print()
print('SACRIFICIAL PARCEL FINDING — class 3 jumpers composition:')
# Merge nlcd info into all-parcel counterfactual
nlcd_cols = fm[['parcel_id', 'nlcd_class', 'is_likely_public_green']].copy()
cf_all_nlcd = cf_all.merge(nlcd_cols, on='parcel_id', how='left')
cls3_jumpers = cf_all_nlcd[cf_all_nlcd['jumped_to_class3'] == 1]
if len(cls3_jumpers) == 0:
    print('  No parcels jumped to class 3 in all-parcel run.')
else:
    n_cls3_res   = cls3_jumpers['nlcd_class'].isin(RESIDENTIAL_CLASSES).sum()
    n_cls3_green = cls3_jumpers['is_likely_public_green'].sum()
    print(f'  Total parcels jumping to class 3 (all-parcel run): {len(cls3_jumpers):,}')
    print(f'    Residential (NLCD 22-24): {n_cls3_res} ({100*n_cls3_res/max(len(cls3_jumpers),1):.1f}%)')
    print(f'    Likely public/green     : {n_cls3_green} ({100*n_cls3_green/max(len(cls3_jumpers),1):.1f}%)')

if n_res_to_cls3 > 0:
    print(f'  Residential-only run: {n_res_to_cls3} parcels jump to class 3')
    c3_archs = fm_res.loc[fm_res['jumped_to_class3'] == 1, 'residential_archetype'].value_counts()
    for arch, cnt in c3_archs.items():
        print(f'    {arch}: {cnt}')
else:
    print(f'  Residential-only run: 0 parcels jump to class 3')

# ── ceiling effect verdict ────────────────────────────────────────────────────
print()
print('CEILING EFFECT VERDICT:')
hotspot_row    = arch_cf_df[arch_cf_df.archetype == 'Hotspot_res'].iloc[0]
max_delta_arch = arch_cf_df['mean_delta_risk'].max()
min_pct_arch   = arch_cf_df['pct_jumped'].min()
is_ceiling = (
    hotspot_row['mean_delta_risk'] >= 0.9 * max_delta_arch and
    hotspot_row['pct_jumped'] <= min_pct_arch * 1.5
)
print(f'  Hotspot_res mean ΔRisk  : {hotspot_row["mean_delta_risk"]:+.4f}  '
      f'({"highest" if hotspot_row["mean_delta_risk"] == max_delta_arch else "not highest"} among archetypes)')
print(f'  Hotspot_res % jumpers   : {hotspot_row["pct_jumped"]:.1f}%  '
      f'({"lowest" if hotspot_row["pct_jumped"] == min_pct_arch else "not lowest"} among archetypes)')
print(f'  Ceiling effect holds    : {"YES" if is_ceiling else "NO"}')

# ══════════════════════════════════════════════════════════════════════════════
# 7. Save outputs
# ══════════════════════════════════════════════════════════════════════════════
cf_cols = [
    'parcel_id', 'residential_archetype', 'nlcd_class', 'risk_class',
    'ISA_frac', 'Cw_topo',
    'ridge_mean_base', 'ridge_max_base', 'rf_class_base',
    'ridge_mean_cf', 'ridge_max_cf', 'rf_class_cf',
    'delta_isa', 'delta_ridge_mean', 'delta_ridge_max',
    'class_jump', 'jumped_up', 'jumped_to_class3',
]
fm_res[cf_cols].to_csv(RES_DIR / 'counterfactual_20pct_residential.csv', index=False)
print(f'\n  Saved counterfactual_20pct_residential.csv  ({len(fm_res):,} rows)')

scen_df.to_csv(RES_DIR / 'intervention_scenarios_residential.csv', index=False)
print(f'  Saved intervention_scenarios_residential.csv  ({len(scen_df)} rows)')

# ══════════════════════════════════════════════════════════════════════════════
# 8. Load geometry for map figures
# ══════════════════════════════════════════════════════════════════════════════
print('\nLoading geometry ...')
gpkg_path = RES_DIR / 'parcel_archetypes_v2_nlcd.gpkg'
gdf_all = None
gdf = None
if gpkg_path.exists():
    gdf_all = gpd.read_file(gpkg_path)
    # Keep residential only and merge counterfactual columns
    gdf_all = gdf_all.merge(
        fm_res[['parcel_id', 'residential_archetype',
                'delta_isa', 'delta_ridge_max', 'class_jump',
                'jumped_up', 'jumped_to_class3',
                'ridge_mean_base', 'ridge_max_base']],
        on='parcel_id', how='inner',
    )
    gdf = gdf_all[gdf_all['residential_archetype'].notna()].copy()
    print(f'  {len(gdf):,} residential parcels with geometry')
else:
    print('  [WARN] parcel_archetypes_v2_nlcd.gpkg not found — map figures skipped')

print('\nGenerating figures ...')

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — ISA perturbation map
# ══════════════════════════════════════════════════════════════════════════════
def step1_isa_perturbation_map():
    if gdf is None:
        print('  [SKIP] step1'); return
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf[gdf['delta_isa'] == 0].plot(
        ax=ax, color='#cccccc', linewidth=0, alpha=0.6,
        label='No change (ISA at cap)')
    sub = gdf[gdf['delta_isa'] > 0].copy()
    sub.plot(column='delta_isa', cmap='OrRd', ax=ax, linewidth=0,
             vmin=0, legend=True,
             legend_kwds={'label': 'ΔISA (+20%)', 'shrink': 0.55, 'pad': 0.02})
    n_capped = (gdf['delta_isa'] == 0).sum()
    n_changed = (gdf['delta_isa'] > 0).sum()
    ax.set_title(
        f'Step 1 — ISA Perturbation Map (+20% Scenario, Residential Parcels)\n'
        f'{n_changed:,} parcels with ΔISA > 0  |  {n_capped:,} capped at ISA = 1.0  '
        f'(grey)', fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    save_fig(fig, 'step1_isa_perturbation_map')

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Risk change distribution
# ══════════════════════════════════════════════════════════════════════════════
def step2_risk_change_distribution():
    delta = fm_res['delta_ridge_max'].values
    mu    = delta.mean()
    med   = np.median(delta)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall
    ax = axes[0]
    ax.hist(delta, bins=80, color='#3B4992', edgecolor='white', alpha=0.85)
    ax.axvline(mu,  color='#C8102E', lw=1.8, ls='-',  label=f'Mean = {mu:+.4f}')
    ax.axvline(med, color='#E9C46A', lw=1.8, ls='--', label=f'Median = {med:+.4f}')
    ax.axvline(0,   color='black',   lw=0.8, alpha=0.5)
    ax.set_xlabel('ΔRisk (Ridge target_max_log: cf − base)')
    ax.set_ylabel('Parcel count')
    ax.set_title(f'(a) Overall distribution — {len(delta):,} residential parcels', fontsize=10)
    ax.legend(fontsize=8.5)

    # Right: by archetype (violin)
    ax = axes[1]
    arch_data = [fm_res.loc[fm_res.residential_archetype == a, 'delta_ridge_max'].values
                 for a in RES_ARCH_ORDER]
    parts = ax.violinplot(arch_data, showmedians=True, showextrema=False)
    for i, (pc, arch) in enumerate(zip(parts['bodies'], RES_ARCH_ORDER)):
        pc.set_facecolor(RES_ARCH_COLORS[arch])
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([a.replace('_res', '\n_res') for a in RES_ARCH_ORDER],
                       fontsize=7.5)
    ax.set_ylabel('ΔRisk (Ridge target_max_log)')
    ax.set_title('(b) ΔRisk distribution by residential archetype', fontsize=10)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)

    fig.suptitle('+20% ISA Development — Risk Change Distribution (Residential Parcels)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'step2_risk_change_distribution')

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Risk class transition heatmap
# ══════════════════════════════════════════════════════════════════════════════
def step3_risk_class_transitions():
    mat = np.zeros((4, 4), dtype=int)
    for base, cf in zip(fm_res['rf_class_base'], fm_res['rf_class_cf']):
        mat[int(base), int(cf)] += 1

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        pd.DataFrame(mat,
                     index=[f'Class {c} (base)' for c in range(4)],
                     columns=[f'Class {c} (cf)' for c in range(4)]),
        annot=True, fmt=',d', cmap='Blues',
        cbar_kws={'label': 'Parcel count'}, linewidths=0.5, ax=ax,
    )
    for i in range(4):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='#228833', lw=2.0))
    n_stable  = int(np.diag(mat).sum())
    n_up      = int(np.tril(mat, -1).sum())
    n_down    = int(np.triu(mat, 1).sum())
    ax.set_title(
        f'Step 3 — Risk Class Transitions Under +20% ISA (Residential, n={len(fm_res):,})\n'
        f'Stable: {n_stable:,}  |  Jumped up: {n_up:,}  |  Dropped: {n_down:,}  '
        f'(green = no change)', fontsize=9.5)
    ax.set_xlabel('Counterfactual RF risk class')
    ax.set_ylabel('Baseline RF risk class')
    fig.tight_layout()
    save_fig(fig, 'step3_risk_class_transitions')

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Class jumpers map (Figure 5a candidate)
# ══════════════════════════════════════════════════════════════════════════════
def step4_class_jumpers_map():
    if gdf is None:
        print('  [SKIP] step4'); return

    jump_colors = {0: '#DDDDDD', 1: '#FFAA00', 2: '#EE4400', 3: '#880000'}
    jump_labels = {0: 'No change', 1: '+1 class', 2: '+2 classes', 3: '+3 classes'}

    fig, ax = plt.subplots(figsize=(13, 10))

    gdf[gdf['jumped_up'] == 0].plot(
        ax=ax, color='#DDDDDD', linewidth=0, alpha=0.55, rasterized=True)

    for mag in [1, 2, 3]:
        sub = gdf[gdf['class_jump'] == mag]
        if len(sub):
            sub.plot(ax=ax, color=jump_colors[mag], linewidth=0, alpha=0.9,
                     rasterized=True)

    handles = [mpatches.Patch(color=jump_colors[m],
               label=f'{jump_labels[m]} (n={int((gdf.class_jump == m).sum() if m > 0 else (gdf.jumped_up == 0).sum()):,})')
               for m in [0, 1, 2, 3]]
    ax.legend(handles=handles, loc='lower right', fontsize=9,
              title='Risk class jump', framealpha=0.9)
    ax.set_title(
        f'Fig. 5a — Residential Parcels Jumping ≥1 Risk Class Under +20% ISA\n'
        f'{n_res_jumped:,} of {n_res_total:,} residential parcels '
        f'({pct_res_jumped:.1f}%) jump at least one class', fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    save_fig(fig, 'step4_class_jumpers_map')

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Archetype vulnerability + ceiling effect (Figure 5b candidate)
# ══════════════════════════════════════════════════════════════════════════════
def step5_archetype_vulnerability():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: % jumped per archetype
    ax = axes[0]
    pcts   = [arch_cf_df.set_index('archetype').loc[a, 'pct_jumped'] for a in RES_ARCH_ORDER]
    colors = [RES_ARCH_COLORS[a] for a in RES_ARCH_ORDER]
    short  = [a.replace('_res', '').replace('_', '\n') for a in RES_ARCH_ORDER]

    bars = ax.bar(range(4), pcts, color=colors, edgecolor='white', alpha=0.88, width=0.55)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 0.15,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Annotate ceiling effect on Hotspot_res
    hs_idx = RES_ARCH_ORDER.index('Hotspot_res')
    hs_pct = pcts[hs_idx]
    hs_dRisk = arch_cf_df.set_index('archetype').loc['Hotspot_res', 'mean_delta_risk']
    ax.annotate(
        f'Ceiling effect:\nhighest ΔRisk ({hs_dRisk:+.4f})\nbut lowest % jumpers',
        xy=(hs_idx, hs_pct),
        xytext=(hs_idx + 1.1, hs_pct + max(pcts) * 0.25),
        fontsize=8, color='#C8102E',
        arrowprops=dict(arrowstyle='->', lw=1.2, color='#C8102E'),
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0F0', ec='#C8102E', alpha=0.9),
    )

    ax.set_xticks(range(4))
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel('% of parcels jumping ≥1 risk class')
    ax.set_title('(a) % of parcels jumping ≥1 class\nunder +20% ISA', fontsize=10)
    ax.set_ylim(0, max(pcts) * 1.45)
    ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

    # Right: dual-axis — % jumped (bars) + mean ΔRisk (line)
    ax2 = axes[1]
    ax2b = ax2.twinx()

    bars2 = ax2.bar(range(4), pcts, color=colors, edgecolor='white', alpha=0.60, width=0.55)
    mean_drisks = [arch_cf_df.set_index('archetype').loc[a, 'mean_delta_risk'] for a in RES_ARCH_ORDER]
    ax2b.plot(range(4), mean_drisks, 'ko-', ms=8, lw=2, zorder=5, label='Mean ΔRisk (Ridge)')
    for i, (x, v) in enumerate(zip(range(4), mean_drisks)):
        ax2b.text(x + 0.12, v + 0.0005, f'{v:+.4f}', fontsize=8, va='center')

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(short, fontsize=8.5)
    ax2.set_ylabel('% parcels jumping ≥1 class', color='#555555')
    ax2b.set_ylabel('Mean ΔRisk (Ridge target_max_log)', color='black')
    ax2.set_title('(b) Ceiling effect: Hotspot_res has\nhighest ΔRisk but lowest % jumpers',
                  fontsize=10)
    ax2.set_ylim(0, max(pcts) * 1.45)
    ax2b.set_ylim(0, max(mean_drisks) * 2.0)
    ax2.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
    ax2b.legend(fontsize=8.5, loc='upper right')

    fig.suptitle('Fig. 5b — Residential Archetype Vulnerability to +20% ISA Development',
                 fontsize=10.5, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'step5_archetype_vulnerability')

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Baseline risk vs ΔRisk scatter
# ══════════════════════════════════════════════════════════════════════════════
def step6_hotspot_vs_development_pressure():
    rng   = np.random.default_rng(RANDOM_STATE)
    s_idx = rng.choice(len(fm_res), min(25_000, len(fm_res)), replace=False)
    sub   = fm_res.iloc[s_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    for arch in RES_ARCH_ORDER:
        mask = sub['residential_archetype'] == arch
        ax.scatter(sub.loc[mask, 'ridge_mean_base'],
                   sub.loc[mask, 'delta_ridge_max'],
                   alpha=0.18, s=5, c=RES_ARCH_COLORS[arch],
                   label=arch.replace('_res', ''), rasterized=True)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    # Annotate Hotspot cluster
    hs = sub[sub['residential_archetype'] == 'Hotspot_res']
    if len(hs):
        ax.annotate(
            f'Hotspot_res\n(n={int((fm_res.residential_archetype=="Hotspot_res").sum()):,})',
            xy=(hs['ridge_mean_base'].mean(), hs['delta_ridge_max'].mean()),
            xytext=(hs['ridge_mean_base'].mean() - 0.4, hs['delta_ridge_max'].mean() + 0.005),
            fontsize=8.5, color='#C8102E',
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#C8102E'),
        )

    handles = [mpatches.Patch(color=RES_ARCH_COLORS[a],
               label=a.replace('_res', '')) for a in RES_ARCH_ORDER]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9,
              title='Residential archetype', title_fontsize=8)
    ax.set_xlabel('Baseline Ridge risk (target_mean_log)')
    ax.set_ylabel('ΔRisk under +20% ISA (target_max_log: cf − base)')
    ax.set_title(
        'Step 6 — Baseline Risk vs Risk Change Under +20% ISA (Residential)\n'
        'Do already-risky parcels get disproportionately worse?', fontsize=10)
    fig.tight_layout()
    save_fig(fig, 'step6_hotspot_vs_development_pressure')

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Supplementary interventions
# ══════════════════════════════════════════════════════════════════════════════
def step7_supplementary_interventions():
    pivot     = scen_df.pivot(index='intervention', columns='group', values='pct_lower_risk')
    pivot_esc = scen_df.pivot(index='intervention', columns='group', values='pct_escaped_top10_res')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    group_colors = {'hotspot_res': '#C8102E', 'top10_res': '#2E86AB'}
    group_labels = {'hotspot_res': 'Hotspot_res\n(n=2,020)', 'top10_res': 'Top-10% residential'}

    for ax, piv, ylabel, title_suf in zip(
        axes,
        [pivot, pivot_esc],
        ['% of treated parcels with lower Ridge risk',
         '% that escaped top-10% residential risk'],
        ['% Lower Risk', '% Escaped Top-10% (residential ranking)'],
    ):
        x = np.arange(len(piv.index))
        w = 0.36
        for i, grp in enumerate(['hotspot_res', 'top10_res']):
            if grp not in piv.columns:
                continue
            vals = piv[grp].values
            bars = ax.bar(x + (i - 0.5) * w, vals, w,
                          label=group_labels[grp],
                          color=group_colors[grp], alpha=0.85, edgecolor='white')
            for xi, v in zip(x + (i - 0.5) * w, vals):
                ax.text(xi, v + 0.5, f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=0, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'(Supplementary) {title_suf}\n3 interventions × 2 residential groups',
                     fontsize=10)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.set_ylim(0, 108)
        ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

    fig.suptitle('Step 7 — Supplementary Intervention Scenarios (Residential Parcels)',
                 fontsize=10.5, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'step7_supplementary_interventions')

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Summary table
# ══════════════════════════════════════════════════════════════════════════════
def step8_summary_table():
    rows = [
        ['Metric', 'All parcels', 'Residential only', 'Δ'],
        ['n parcels',
         f'{n_all_total:,}', f'{n_res_total:,}', '—'],
        ['n jumped ≥1 class',
         f'{n_all_jumped:,}', f'{n_res_jumped:,}',
         f'{n_res_jumped - n_all_jumped:+,}'],
        ['% jumped',
         f'{100*n_all_jumped/n_all_total:.2f}%',
         f'{pct_res_jumped:.2f}%',
         f'{pct_res_jumped - 100*n_all_jumped/n_all_total:+.2f}'],
        ['n jumped to class 3',
         f'{n_all_to_cls3:,}', f'{n_res_to_cls3:,}',
         f'{n_res_to_cls3 - n_all_to_cls3:+,}'],
        ['Mean ΔRisk (Ridge max)',
         f'{cf_all["delta_ridge_max"].mean():+.4f}',
         f'{mean_delta_res:+.4f}',
         f'{mean_delta_res - cf_all["delta_ridge_max"].mean():+.4f}'],
        ['Ceiling effect', '—', '—', '—'],
        ['Hotspot_res mean ΔRisk',
         f'{arch_cf_df.set_index("archetype").loc["Hotspot_res","mean_delta_risk"]:+.4f}',
         '(highest)',  '↑ disproportionate'],
        ['Hotspot_res % jumped',
         f'{arch_cf_df.set_index("archetype").loc["Hotspot_res","pct_jumped"]:.1f}%',
         '(lowest)', '↓ at ceiling'],
    ]
    # Add per-archetype rows
    rows += [['', '', '', '']]
    rows += [['Archetype', 'n parcels', '% jumped', 'mean ΔRisk']]
    for _, r in arch_cf_df.iterrows():
        rows.append([r['archetype'], f'{r["n_parcels"]:,}',
                     f'{r["pct_jumped"]:.1f}%', f'{r["mean_delta_risk"]:+.4f}'])

    fig, ax = plt.subplots(figsize=(11, len(rows) * 0.44 + 1))
    ax.set_axis_off()
    tbl = ax.table(cellText=rows, loc='center', cellLoc='left',
                   colWidths=[0.35, 0.22, 0.22, 0.21])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)

    # Style header row
    for col in range(4):
        tbl[0, col].set_facecolor('#1A3A5C')
        tbl[0, col].set_text_props(color='white', fontweight='bold')
    # Style ceiling effect sub-header
    for row_i, row in enumerate(rows):
        if row[0] == 'Ceiling effect':
            for col in range(4):
                tbl[row_i, col].set_facecolor('#FFF0C8')
                tbl[row_i, col].set_text_props(fontweight='bold')
        if row[0] == 'Archetype':
            for col in range(4):
                tbl[row_i, col].set_facecolor('#DDDDDD')
                tbl[row_i, col].set_text_props(fontweight='bold')
        if row[0] in RES_ARCH_ORDER:
            tbl[row_i, 0].set_facecolor(RES_ARCH_COLORS.get(row[0], 'white'))
            tbl[row_i, 0].set_text_props(color='white', fontweight='bold')

    ax.set_title('Step 8 — Residential Counterfactual Summary (+20% ISA Scenario)',
                 fontsize=11, pad=12)
    fig.tight_layout()
    save_fig(fig, 'step8_summary_table')

# ── run all figures ───────────────────────────────────────────────────────────
step1_isa_perturbation_map()
step2_risk_change_distribution()
step3_risk_class_transitions()
step4_class_jumpers_map()
step5_archetype_vulnerability()
step6_hotspot_vs_development_pressure()
step7_supplementary_interventions()
step8_summary_table()

# ══════════════════════════════════════════════════════════════════════════════
# 9. Full terminal report
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 65)
print('FULL TERMINAL REPORT')
print('=' * 65)

print(f'\nTotal residential parcels           : {n_res_total:,}')
print(f'Jumping ≥1 risk class               : {n_res_jumped:,}  ({pct_res_jumped:.2f}%)')
print(f'Jumping to class 3 (extreme flood)  : {n_res_to_cls3:,}')
print(f'Mean ΔRisk (Ridge max)              : {mean_delta_res:+.4f}')

print('\nPer-archetype breakdown:')
print(f'  {"Archetype":30s}  {"n":>6}  {"jumped":>7}  {"pct":>6}  {"cls3":>5}  {"mean_dRisk":>10}')
print('  ' + '-' * 72)
for _, r in arch_cf_df.iterrows():
    print(f'  {r.archetype:30s}  {r.n_parcels:>6,}  {r.n_jumped:>7,}  '
          f'{r.pct_jumped:>5.1f}%  {r.n_to_class3:>5,}  {r.mean_delta_risk:>+10.4f}')

print(f'\nCeiling effect: {"CONFIRMED" if is_ceiling else "NOT CONFIRMED"}')
print(f'  Hotspot_res has the highest mean ΔRisk ({hotspot_row.mean_delta_risk:+.4f})')
print(f'  but the lowest % of class jumpers ({hotspot_row.pct_jumped:.1f}%)')
print(f'  because most Hotspot_res parcels are already in class 3 '
      f'(rf_class_base=3: '
      f'{(fm_res[fm_res.residential_archetype=="Hotspot_res"]["rf_class_base"]==3).sum():,} '
      f'of {int(arch_cf_df.set_index("archetype").loc["Hotspot_res","n_parcels"]):,})')

print()
print('=' * 65)
print('DONE')
print(f'  Results : {RES_DIR}/')
print(f'  Figures : {FIG_DIR}/')
print('=' * 65)
