#!/usr/bin/env python3
"""
residential_hotspots.py
Policy-clean hotspot definition: re-rank and re-cluster within residential
parcels only (NLCD 22/23/24 — Developed Low/Medium/High Intensity).

Outputs
-------
outputs/results/residential_archetypes.csv   (parcel-level labels + risk ranks)
outputs/figures/residential_hotspots/        6 figures (PNG @ 150 DPI)
"""
import sys, warnings, time
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import lightgbm as lgb
import shap

# ── paths ─────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / 'data' / 'processed'
RES_DIR  = BASE / 'outputs' / 'results'
FIG_DIR  = BASE / 'outputs' / 'figures' / 'residential_hotspots'
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'elevation','slope','TWI','log_flow_accum','dist_to_stream',
    'dist_to_street','ISA_frac','log_lot_area','is_enclave',
    'conn_topo','Cw_topo','HAND_min',
]
RESIDENTIAL_NLCD = {22, 23, 24}

RES_ARCH_ORDER  = ['Hotspot_res','Lowland_baseline_res',
                   'Upland_baseline_res','Upland_shield_res']
RES_ARCH_COLORS = {
    'Hotspot_res'          : '#C8102E',
    'Lowland_baseline_res' : '#E9C46A',
    'Upland_baseline_res'  : '#A8DADC',
    'Upland_shield_res'    : '#2E86AB',
}

plt.rcParams.update({
    'font.family':'sans-serif','font.size':9,'axes.titlesize':10,
    'axes.spines.top':False,'axes.spines.right':False,
})

def save_fig(fig, name):
    p = FIG_DIR / f'{name}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f'  [OK] {name}.png  ({p.stat().st_size/1024:.0f} KB)')

# ─────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('residential_hotspots.py')
print('=' * 60)

fm  = pd.read_csv(RES_DIR / 'feature_matrix_nlcd.csv')
arc = pd.read_csv(RES_DIR / 'parcel_archetypes_v2.csv')
fm  = fm.merge(arc[['parcel_id','archetype'] +
                   [f'shap_{f}' for f in FEATURES]],
               on='parcel_id', how='left', suffixes=('','_arc'))

N   = len(fm)
res_mask = fm['nlcd_class'].isin(RESIDENTIAL_NLCD).values
n_res    = res_mask.sum()
n_pg     = (~res_mask).sum()

print(f'  Total parcels      : {N:,}')
print(f'  Residential (22-24): {n_res:,}  ({100*n_res/N:.1f}%)')
print(f'  Excluded (non-res) : {n_pg:,}  ({100*n_pg/N:.1f}%)')

X_all = fm[FEATURES].values
y_all = fm['target_max_log'].values

# residential sub-arrays
X_res = X_all[res_mask]
y_res = y_all[res_mask]
idx_res = np.where(res_mask)[0]   # global indices for residential parcels

# ─────────────────────────────────────────────────────────────────────────
# 2. Ridge on full data → predict risk → rank residential
# ─────────────────────────────────────────────────────────────────────────
print('\nFitting Ridge (full data) …')
sc        = StandardScaler()
X_scaled  = sc.fit_transform(X_all)
ridge_mdl = Ridge(alpha=1.0).fit(X_scaled, y_all)
preds_all = ridge_mdl.predict(X_scaled)

# rank within residential subset
preds_res = preds_all[res_mask]
ranks_res = pd.Series(preds_res).rank(ascending=False, method='min').values.astype(int)
top1pct   = int(np.ceil(n_res * 0.01))
top10pct  = int(np.ceil(n_res * 0.10))
is_top1   = ranks_res <= top1pct
is_top10  = ranks_res <= top10pct
print(f'  Top 1%  threshold rank ≤ {top1pct:,}  (n={is_top1.sum():,})')
print(f'  Top 10% threshold rank ≤ {top10pct:,}  (n={is_top10.sum():,})')

# ─────────────────────────────────────────────────────────────────────────
# 3. LGB on full data → SHAP for residential subset only
# ─────────────────────────────────────────────────────────────────────────
def scale_to_rank_labels(y, n_bins=32):
    ranks = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates='drop')
    return ranks.fillna(0).astype(int).values

print('Fitting LGB (full data) …')
y_lbl  = scale_to_rank_labels(y_all)
chunk  = 9000
n_all  = len(y_lbl)
groups = [chunk]*(n_all//chunk) + ([n_all%chunk] if n_all%chunk else [])
ds_all = lgb.Dataset(X_all, label=y_lbl, group=groups)
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[10],
              num_leaves=63, learning_rate=0.05, verbose=-1,
              min_data_in_group=1)
t0 = time.time()
lgb_mdl = lgb.train(params, ds_all, num_boost_round=500,
                    callbacks=[lgb.log_evaluation(-1)])
print(f'  LGB trained in {time.time()-t0:.0f}s')

print('Computing SHAP for residential parcels …')
t0  = time.time()
ex  = shap.TreeExplainer(lgb_mdl)
sv  = ex.shap_values(X_res)
if isinstance(sv, list): sv = sv[0]
if sv.ndim == 3:         sv = sv[:,:,0]
print(f'  SHAP done in {time.time()-t0:.0f}s  shape={sv.shape}')

# ─────────────────────────────────────────────────────────────────────────
# 4. KMeans k=4 on residential SHAP → archetype labels
# ─────────────────────────────────────────────────────────────────────────
print('KMeans k=4 on residential SHAP …')
km_res = KMeans(n_clusters=4, random_state=42, n_init=10).fit(sv)

C     = km_res.cluster_centers_
si    = FEATURES.index('slope')
fi    = FEATURES.index('log_flow_accum')
ei    = FEATURES.index('elevation')

hs_cl = int(np.argmax(C[:,si] + C[:,fi]))
rem   = [c for c in range(4) if c != hs_cl]
us_cl = rem[int(np.argmin([C[c,ei] for c in rem]))]
rem2  = [c for c in rem if c != us_cl]
ub_cl = rem2[int(np.argmin([C[c,ei] for c in rem2]))]
lb_cl = [c for c in rem2 if c != ub_cl][0]

lmap = {hs_cl:'Hotspot_res', lb_cl:'Lowland_baseline_res',
        ub_cl:'Upland_baseline_res', us_cl:'Upland_shield_res'}
res_arch = np.array([lmap[c] for c in km_res.labels_])

print('Residential archetype sizes:')
for a in RES_ARCH_ORDER:
    n = (res_arch == a).sum()
    print(f'  {a:<28}: {n:,}  ({100*n/n_res:.1f}%)')

# ─────────────────────────────────────────────────────────────────────────
# 5. Build residential DataFrame
# ─────────────────────────────────────────────────────────────────────────
fm_res = fm[res_mask].copy().reset_index(drop=True)
fm_res['residential_archetype'] = res_arch
fm_res['ridge_pred_res']        = preds_res
fm_res['rank_res']              = ranks_res
fm_res['is_top1pct_res']        = is_top1
fm_res['is_top10pct_res']       = is_top10
# SHAP columns
for i, f in enumerate(FEATURES):
    fm_res[f'res_shap_{f}'] = sv[:, i]

# ─────────────────────────────────────────────────────────────────────────
# 6. Overlap analysis — original Hotspot vs Hotspot_res
# ─────────────────────────────────────────────────────────────────────────
orig_hs    = set(fm[fm['archetype'] == 'Hotspot']['parcel_id'])
res_hs     = set(fm_res[fm_res['residential_archetype'] == 'Hotspot_res']['parcel_id'])
overlap    = orig_hs & res_hs
dropped    = orig_hs - res_hs           # orig Hotspot → NOT Hotspot_res
added      = res_hs  - orig_hs          # new residential hotspot entries

n_orig     = len(orig_hs)
n_res_hs   = len(res_hs)
n_overlap  = len(overlap)
n_dropped  = len(dropped)
n_added    = len(added)

# feature profile comparison
hs_orig_rows = fm[fm['parcel_id'].isin(orig_hs)]
hs_res_rows  = fm_res[fm_res['residential_archetype'] == 'Hotspot_res']

compare_feats = ['slope','HAND_min','ISA_frac','log_lot_area',
                 'dist_to_stream','target_mean']
prof = pd.DataFrame({
    'Original Hotspot (n={:,})'.format(n_orig):
        hs_orig_rows[compare_feats].mean(),
    'Hotspot_res (n={:,})'.format(n_res_hs):
        hs_res_rows[compare_feats].mean(),
})
prof['Ratio (res/orig)'] = (prof.iloc[:,1] / prof.iloc[:,0])

# where are dropped parcels (public/green or reclassified)?
dropped_rows = fm[fm['parcel_id'].isin(dropped)]
dropped_nlcd = dropped_rows['nlcd_class'].value_counts()
dropped_arch_res = fm_res[fm_res['parcel_id'].isin(dropped)]['residential_archetype'] \
    if len(fm_res[fm_res['parcel_id'].isin(dropped)]) else pd.Series()

# geographic concentration of new hotspots
if len(fm_res[fm_res['parcel_id'].isin(added)]) > 0:
    added_rows = fm_res[fm_res['parcel_id'].isin(added)]
    geo_conc   = added_rows['fold_subbasin'].value_counts().head(3)
else:
    geo_conc = pd.Series(dtype=int)

# ─────────────────────────────────────────────────────────────────────────
# 7. ISA paradox — residential archetypes
# ─────────────────────────────────────────────────────────────────────────
qlabels = ['Q1','Q2','Q3','Q4','Q5']
fm_res['isa_q'] = pd.qcut(fm_res['ISA_frac'], q=5, labels=qlabels,
                           duplicates='drop')

piv  = (fm_res.groupby(['residential_archetype','isa_q'])['target_mean']
              .mean().unstack().reindex(RES_ARCH_ORDER))
cnt  = (fm_res.groupby(['residential_archetype','isa_q'])['target_mean']
              .count().unstack().reindex(RES_ARCH_ORDER))
std_ = (fm_res.groupby(['residential_archetype','isa_q'])['target_mean']
              .std().unstack().reindex(RES_ARCH_ORDER))

def bootstrap_q5q1_ci(df, isa_col='isa_q', target='target_mean',
                      n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    q1  = df[df[isa_col]=='Q1'][target].values
    q5  = df[df[isa_col]=='Q5'][target].values
    if len(q1)==0 or len(q5)==0:
        return np.nan, np.nan, np.nan
    point = q5.mean() - q1.mean()
    boot  = np.array([
        rng.choice(q5,len(q5),replace=True).mean() -
        rng.choice(q1,len(q1),replace=True).mean()
        for _ in range(n_boot)
    ])
    ci = np.percentile(boot, [2.5, 97.5])
    return point, ci[0], ci[1]

ci_rows = []
for arch in RES_ARCH_ORDER:
    sub = fm_res[fm_res['residential_archetype'] == arch]
    p, lo, hi = bootstrap_q5q1_ci(sub)
    ci_rows.append(dict(arch=arch, delta=p, lo=lo, hi=hi,
                        n_q1=int((sub['isa_q']=='Q1').sum()),
                        n_q5=int((sub['isa_q']=='Q5').sum())))
ci_df = pd.DataFrame(ci_rows)

# ─────────────────────────────────────────────────────────────────────────
# 8. Save results CSV
# ─────────────────────────────────────────────────────────────────────────
out_cols = ['parcel_id','nlcd_class','nlcd_class_name','archetype',
            'residential_archetype','ridge_pred_res','rank_res',
            'is_top1pct_res','is_top10pct_res','ISA_frac','HAND_min',
            'slope','target_mean','target_max']
out_cols = [c for c in out_cols if c in fm_res.columns]
fm_res[out_cols].to_csv(RES_DIR / 'residential_archetypes.csv', index=False)
print(f'\n  Saved residential_archetypes.csv  ({len(fm_res):,} rows)')

# ─────────────────────────────────────────────────────────────────────────
# 9. Terminal report
# ─────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('OVERLAP ANALYSIS')
print('=' * 60)
print(f'  Original Hotspot (full data KMeans): {n_orig:,}')
print(f'  Residential Hotspot_res (res KMeans): {n_res_hs:,}')
print(f'  Overlap (both):  {n_overlap:,}  ({100*n_overlap/n_orig:.1f}% of orig, '
      f'{100*n_overlap/n_res_hs:.1f}% of res)')
print(f'  Dropped out:     {n_dropped:,}  '
      f'(orig Hotspot → NOT Hotspot_res)')
print(f'  Added (new):     {n_added:,}  '
      f'(new residential hotspot)')

if len(dropped_nlcd):
    print(f'\n  NLCD classes of dropped-out parcels:')
    for cls, n in dropped_nlcd.items():
        from add_nlcd_filter import NLCD_NAMES
        print(f'    {cls}: {NLCD_NAMES.get(cls,"?")}  → {n}')

if len(geo_conc):
    print(f'\n  New Hotspot_res concentration by sub-basin:')
    for fold, n in geo_conc.items():
        print(f'    {fold}: {n} new hotspot parcels')

print()
print('FEATURE PROFILE COMPARISON')
print('-' * 55)
print(prof.to_string(float_format='{:.3f}'.format))

print()
print('=' * 60)
print('ISA QUINTILE × RESIDENTIAL ARCHETYPE')
print('=' * 60)
print(piv.to_string(float_format='{:.3f}'.format))

print()
print('Q5-Q1 DELTAS WITH BOOTSTRAP 95% CI (1000 resamples):')
print(f'  {"Archetype":<28}  {"delta":>8}  {"lo":>8}  {"hi":>8}  {"sig?":>6}')
print('  ' + '-'*60)
for _, r in ci_df.iterrows():
    sig = 'YES' if not (r.lo <= 0 <= r.hi) else 'no'
    print(f'  {r.arch:<28}  {r.delta:>+8.3f}  {r.lo:>+8.3f}  {r.hi:>+8.3f}  {sig:>6}')

hs_row   = ci_df[ci_df.arch=='Hotspot_res'].iloc[0]
hs_sig   = not (hs_row.lo <= 0 <= hs_row.hi)
hs_delta = hs_row.delta

print()
print('VERDICT:')
if hs_sig and hs_delta < 0:
    verdict = ('ISA paradox HOLDS for Hotspot_res — CI excludes zero '
               f'({hs_row.lo:+.3f}, {hs_row.hi:+.3f})')
elif not hs_sig:
    verdict = ('ISA paradox INCONCLUSIVE for Hotspot_res — CI crosses zero '
               f'({hs_row.lo:+.3f}, {hs_row.hi:+.3f})')
else:
    verdict = 'ISA paradox SIGN REVERSES for Hotspot_res (unexpected)'
print(f'  {verdict}')

cnt_warn = [(a,q) for a in RES_ARCH_ORDER for q in qlabels
            if not np.isnan(cnt.loc[a,q]) and cnt.loc[a,q] < 100]
if cnt_warn:
    print(f'\n  Sample-size warnings (n<100):')
    for a,q in cnt_warn:
        print(f'    {a} / {q}: n={int(cnt.loc[a,q])}')

# ─────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────
print('\nGenerating figures …')

# load geometry once (slim read)
gdf = gpd.read_file(RES_DIR / 'parcel_archetypes_v2_nlcd.gpkg',
                    columns=['parcel_id','nlcd_class','geometry'])
# merge residential_archetype
gdf = gdf.merge(fm_res[['parcel_id','residential_archetype',
                         'is_top1pct_res','is_top10pct_res']],
                on='parcel_id', how='left')

# ── STEP 1: residential filter map ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

gdf[gdf['nlcd_class'].isin(RESIDENTIAL_NLCD)].plot(
    ax=ax, color='#CCCCCC', linewidth=0, rasterized=True, label='Residential (NLCD 22–24)')
gdf[~gdf['nlcd_class'].isin(RESIDENTIAL_NLCD)].plot(
    ax=ax, color='#267300', linewidth=0, rasterized=True,
    label=f'Excluded — public/green (n={n_pg:,}, {100*n_pg/N:.1f}%)')

ax.set_axis_off()
ax.set_title(f'Residential filter — Brays Bayou\n'
             f'Residential parcels: {n_res:,} ({100*n_res/N:.1f}%)  |  '
             f'Excluded: {n_pg:,} ({100*n_pg/N:.1f}%)')
ax.legend(fontsize=9, loc='lower right', framealpha=0.9, edgecolor='#CCCCCC')
save_fig(fig, 'step1_residential_filter_map')
plt.close(fig)

# ── STEP 2: residential archetype SHAP quadrant ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
rng  = np.random.default_rng(0)
idx_s = rng.choice(len(fm_res), min(6000, len(fm_res)), replace=False)
sub   = fm_res.iloc[idx_s]

for arch in RES_ARCH_ORDER:
    m = sub['residential_archetype'] == arch
    ax.scatter(sub.loc[m,'res_shap_slope'],
               sub.loc[m,'res_shap_HAND_min'],
               c=RES_ARCH_COLORS[arch], label=f'{arch} (n={m.sum():,})',
               alpha=0.25, s=6, linewidths=0, rasterized=True)

# centroids
for arch in RES_ARCH_ORDER:
    m = fm_res['residential_archetype'] == arch
    cx = fm_res.loc[m,'res_shap_slope'].mean()
    cy = fm_res.loc[m,'res_shap_HAND_min'].mean()
    ax.scatter(cx, cy, marker='*', s=220, c=RES_ARCH_COLORS[arch],
               edgecolors='black', linewidths=0.8, zorder=6)
    ax.annotate(arch.replace('_res','').replace('_',' '),
                (cx,cy), textcoords='offset points', xytext=(5,4),
                fontsize=7.5, fontweight='bold', color=RES_ARCH_COLORS[arch])

ax.axhline(0, lw=0.8, color='#888', ls='--')
ax.axvline(0, lw=0.8, color='#888', ls='--')
ax.set_xlabel('SHAP value — slope')
ax.set_ylabel('SHAP value — HAND$_{min}$')
ax.set_title('Residential archetype separation in SHAP space\n'
             '(6,000-parcel sample, ★ = cluster centroid)')
ax.legend(fontsize=8, markerscale=2.5, loc='upper right',
          framealpha=0.9, edgecolor='#CCCCCC')
save_fig(fig, 'step2_residential_archetype_quadrant')
plt.close(fig)

# ── STEP 3: residential archetype map ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
# non-residential background
gdf[gdf['residential_archetype'].isna()].plot(
    ax=ax, color='#EEEEEE', linewidth=0, rasterized=True)
for arch in reversed(RES_ARCH_ORDER):
    gdf[gdf['residential_archetype'] == arch].plot(
        ax=ax, color=RES_ARCH_COLORS[arch], linewidth=0,
        rasterized=True, label=arch)
ax.set_axis_off()
ax.set_title('Residential archetypes — Brays Bayou\n'
             '(public/green excluded, shown as light gray)')
legend_patches = [mpatches.Patch(color=RES_ARCH_COLORS[a],
                                  label=f'{a}  (n={int((fm_res.residential_archetype==a).sum()):,})')
                  for a in RES_ARCH_ORDER]
legend_patches.append(mpatches.Patch(color='#EEEEEE', label='Excluded (non-residential)'))
ax.legend(handles=legend_patches, fontsize=8, loc='lower right',
          framealpha=0.9, edgecolor='#CCCCCC')
save_fig(fig, 'step3_residential_archetype_map')
plt.close(fig)

# ── STEP 4: hotspot overlap composition ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left: composition bars
ax = axes[0]
cats = ['Overlap\n(in both)', 'Dropped out\n(orig only)', 'Added\n(res only)']
vals_orig = [n_overlap, n_dropped, 0]
vals_res  = [n_overlap, 0, n_added]

x = np.arange(3)
w = 0.35
b1 = ax.bar(x[:2] - w/2, [n_overlap, n_dropped], w, color=['#2CA02C','#C8102E'],
            alpha=0.85, label='From original Hotspot (n={:,})'.format(n_orig))
b2 = ax.bar(x[[0,2]] + w/2, [n_overlap, n_added], w, color=['#2CA02C','#2E86AB'],
            alpha=0.85, label='In Hotspot_res (n={:,})'.format(n_res_hs))

for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x()+bar.get_width()/2, h+10, f'{int(h):,}',
                ha='center', va='bottom', fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('Number of parcels')
ax.set_title('(a) Hotspot composition: original vs residential re-cluster')
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)

# right: Sankey-style flow
ax = axes[1]
ax.set_xlim(0, 3); ax.set_ylim(-0.2, 3.5)
ax.axis('off')

# Original Hotspot box
y_orig_top = n_orig / 200
rect_orig = mpatches.FancyBboxPatch((0.1, 1.5), 0.9, y_orig_top,
    boxstyle='round,pad=0.05', fc='#E8E8E8', ec='#888888', lw=1.5)
ax.add_patch(rect_orig)
ax.text(0.55, 1.5 + y_orig_top/2, f'Original\nHotspot\n(n={n_orig:,})',
        ha='center', va='center', fontsize=9, fontweight='bold')

# Hotspot_res box
y_res_top = n_res_hs / 200
rect_res = mpatches.FancyBboxPatch((2.0, 1.5), 0.9, y_res_top,
    boxstyle='round,pad=0.05', fc='#E8E8E8', ec='#888888', lw=1.5)
ax.add_patch(rect_res)
ax.text(2.45, 1.5 + y_res_top/2, f'Hotspot_res\n(n={n_res_hs:,})',
        ha='center', va='center', fontsize=9, fontweight='bold')

# Overlap arrow (green)
ax.annotate('', xy=(2.0, 1.5+y_res_top*0.6), xytext=(1.0, 1.5+y_orig_top*0.6),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#2CA02C'))
ax.text(1.5, 1.5+(y_orig_top+y_res_top)/2 + 0.15,
        f'Overlap\n{n_overlap:,}\n({100*n_overlap/n_orig:.0f}% of orig)',
        ha='center', va='bottom', fontsize=8.5, color='#2CA02C', fontweight='bold')

# Dropped (red arrow down)
ax.annotate('', xy=(0.55, 1.3), xytext=(0.55, 1.5),
            arrowprops=dict(arrowstyle='->', lw=2.0, color='#C8102E'))
ax.text(0.55, 1.15, f'Dropped out\n{n_dropped:,} parcels',
        ha='center', va='top', fontsize=8, color='#C8102E')

# Added (blue arrow up into Hotspot_res)
ax.annotate('', xy=(2.45, 1.5+y_res_top), xytext=(2.45, 1.7+y_res_top),
            arrowprops=dict(arrowstyle='->', lw=2.0, color='#2E86AB'))
ax.text(2.45, 1.85+y_res_top, f'New additions\n{n_added:,} parcels',
        ha='center', va='bottom', fontsize=8, color='#2E86AB')

ax.set_title('(b) Parcel flow: original → residential re-cluster', pad=6)

fig.suptitle('Original Hotspot vs Residential Hotspot_res',
             fontsize=11, y=1.01)
fig.tight_layout()
save_fig(fig, 'step4_hotspot_overlap')
plt.close(fig)

# ── STEP 5: ISA quintile × residential archetype heatmap ─────────────────
vmax = np.nanmax(piv.values)
fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(piv.values, aspect='auto', cmap='YlOrRd', vmin=1, vmax=vmax)
ax.set_xticks(range(5))
ax.set_xticklabels(['Q1\n(low ISA)','Q2','Q3','Q4','Q5\n(high ISA)'])
ax.set_yticks(range(4))
ax.set_yticklabels(RES_ARCH_ORDER, fontsize=8.5)
ax.set_xlabel('ISA quintile (residential parcels only)')
ax.set_title('Mean flood depth (ft) — residential parcels × residential archetype')
for i, arch in enumerate(RES_ARCH_ORDER):
    for j, q in enumerate(qlabels):
        try:
            v = piv.loc[arch, q];  n = int(cnt.loc[arch, q])
        except KeyError:
            continue
        c = 'white' if v > 0.6*vmax else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=c,
                fontweight='bold')
        if n < 100:
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5), 1, 1,
                         fill=False, ec='dodgerblue', lw=2, clip_on=True))
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.ax.set_ylabel('Mean flood depth (ft)', fontsize=8)

# Q5-Q1 delta on right axis
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(4))
ax2.set_yticklabels(
    [f'Δ={ci_df.loc[ci_df.arch==a,"delta"].values[0]:+.2f} ft'
     for a in RES_ARCH_ORDER], fontsize=8)
ax2.tick_params(right=False)
ax2.spines['top'].set_visible(False)

ax.text(0.5, -0.16, 'Blue border = n < 100  |  ISA quintiles recomputed within residential subset',
        transform=ax.transAxes, ha='center', fontsize=7.5, color='#666666', style='italic')
fig.tight_layout()
save_fig(fig, 'step5_residential_isa_paradox')
plt.close(fig)

# ── STEP 6: forest plot of Q5-Q1 CIs ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(4)

ax.errorbar(
    ci_df['delta'], y,
    xerr=[ci_df['delta']-ci_df['lo'], ci_df['hi']-ci_df['delta']],
    fmt='none', lw=2.2, capsize=7, capthick=2.2, zorder=4, ecolor='#555555',
)

# color each marker individually
for i, (_, r) in enumerate(ci_df.iterrows()):
    ax.scatter(r['delta'], i, marker='D', s=100,
               c=RES_ARCH_COLORS[r['arch']], zorder=5, edgecolors='white',
               linewidths=0.8)
    # shade if significant
    if not (r.lo <= 0 <= r.hi):
        ax.axhspan(i-0.28, i+0.28, alpha=0.07, color='#2CA02C', zorder=1)

ax.axvline(0, lw=1.3, color='black', zorder=3, label='No effect (0)')
ax.set_yticks(y)
ax.set_yticklabels(
    [f'{r.arch}\n(n_Q1={r.n_q1:,}  n_Q5={r.n_q5:,})'
     for _, r in ci_df.iterrows()], fontsize=8.5)
ax.set_xlabel('Q5 − Q1 mean flood depth (ft)  [negative = ISA paradox]', fontsize=9.5)
ax.set_title('ISA paradox robustness — residential archetypes\n'
             'Bootstrap 95% CI  |  green shading = CI excludes 0  |  '
             'ISA quintiles recomputed within residential subset')
ax.xaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax.invert_yaxis()

# annotate significance
for i, (_, r) in enumerate(ci_df.iterrows()):
    sig_lbl = '★ significant' if not (r.lo <= 0 <= r.hi) else '— inconclusive'
    clr     = '#2CA02C' if not (r.lo <= 0 <= r.hi) else '#888888'
    ax.text(ax.get_xlim()[0] + 0.01*np.ptp(ax.get_xlim()),
            i + 0.22, sig_lbl, va='top', fontsize=7.5, color=clr, style='italic')

fig.tight_layout()
save_fig(fig, 'step6_residential_paradox_forest')
plt.close(fig)

print()
print('=' * 60)
print('DONE')
print(f'  Results : {RES_DIR}/residential_archetypes.csv')
print(f'  Figures : {FIG_DIR}/')
print('=' * 60)
