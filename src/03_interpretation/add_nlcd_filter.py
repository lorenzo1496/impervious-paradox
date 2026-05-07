#!/usr/bin/env python3
"""
add_nlcd_filter.py
Download NLCD 2016 for Brays Bayou AOI, compute per-parcel dominant land-cover
class, add filter columns to feature_matrix / gpkg, and run ISA-paradox
robustness check on residential-only parcels.

Outputs
-------
data/raw/nlcd_2016_brays.tif                      downloaded raster (EPSG:26915)
outputs/results/feature_matrix_nlcd.csv           + nlcd_class / nlcd_class_name /
                                                    is_likely_public_green
outputs/results/parcel_archetypes_v2_nlcd.gpkg    same new columns
outputs/figures/nlcd/step1_public_green_parcels.png
outputs/figures/nlcd/step2_paradox_all_vs_residential.png
outputs/figures/nlcd/step3_paradox_ci_forest.png
"""
import sys, warnings, io, time
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap

# ── paths ─────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / 'data' / 'processed'
NLCD_TIF = BASE / 'data' / 'raw' / 'nlcd_2016_brays.tif'
NLCD_26915 = BASE / 'data' / 'raw' / 'nlcd_2016_brays_26915.tif'
RES_DIR  = BASE / 'outputs' / 'results'
FIG_DIR  = BASE / 'outputs' / 'figures' / 'nlcd'
FIG_DIR.mkdir(parents=True, exist_ok=True)

ARCH_ORDER  = ['Hotspot', 'Lowland baseline', 'Upland baseline', 'Upland shield']
ARCH_COLORS = {'Hotspot':'#C8102E', 'Lowland baseline':'#E9C46A',
               'Upland baseline':'#A8DADC', 'Upland shield':'#2E86AB'}

plt.rcParams.update({
    'font.family':'sans-serif','font.size':9,'axes.titlesize':10,
    'axes.spines.top':False,'axes.spines.right':False,
})

# ── NLCD class metadata ───────────────────────────────────────────────────
NLCD_NAMES = {
    11: 'Open Water',
    12: 'Perennial Ice/Snow',
    21: 'Developed, Open Space',
    22: 'Developed, Low Intensity',
    23: 'Developed, Med Intensity',
    24: 'Developed, High Intensity',
    31: 'Barren Land',
    41: 'Deciduous Forest',
    42: 'Evergreen Forest',
    43: 'Mixed Forest',
    51: 'Dwarf Scrub',
    52: 'Shrub/Scrub',
    71: 'Grassland/Herbaceous',
    81: 'Pasture/Hay',
    82: 'Cultivated Crops',
    90: 'Woody Wetlands',
    95: 'Emergent Herbaceous Wetlands',
}

PUBLIC_GREEN_CLASSES = {11, 21, 41, 42, 43, 71, 81, 90, 95}
RESIDENTIAL_CLASSES  = {22, 23, 24}

# colors for step1 map
NLCD_COLORS = {
    11: '#476BA1', 21: '#D1FF73', 22: '#FFD27F', 23: '#FF9999',
    24: '#FF0000', 31: '#B2B2B2', 41: '#38A800', 42: '#267300',
    43: '#70A800', 71: '#FEFF73', 81: '#D3FFBE', 82: '#FFFF00',
    90: '#7AF5CA', 95: '#00A884',
}

# ─────────────────────────────────────────────────────────────────────────
# 0.  Download NLCD 2016
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('NLCD 2016 — Brays Bayou')
print('=' * 60)

gdf_raw = gpd.read_file(RES_DIR / 'parcel_archetypes_v2.gpkg')
bounds_native = gdf_raw.total_bounds          # EPSG:26915
buf = 500                                      # 500 m buffer
bbox_26915 = [bounds_native[0]-buf, bounds_native[1]-buf,
              bounds_native[2]+buf, bounds_native[3]+buf]

if NLCD_TIF.exists():
    print(f'  Raster already exists: {NLCD_TIF}  (skip download)')
else:
    print('  Downloading NLCD 2016 from MRLC WCS 2.0.1 ...')
    # WCS 2.0.1 — namespace-level endpoint, correct coverage ID
    wcs_url  = 'https://www.mrlc.gov/geoserver/mrlc_download/wcs'
    b4326    = gdf_raw.to_crs('EPSG:4326').total_bounds
    buf_d    = 0.01
    # SUBSET params must be passed as a list so requests repeats the key
    params = {
        'SERVICE'      : 'WCS',
        'VERSION'      : '2.0.1',
        'REQUEST'      : 'GetCoverage',
        'COVERAGEID'   : 'mrlc_download__NLCD_2016_Land_Cover_L48',
        'SUBSETTINGCRS': 'http://www.opengis.net/def/crs/EPSG/0/4326',
        'OUTPUTCRS'    : 'http://www.opengis.net/def/crs/EPSG/0/4326',
        'SUBSET'       : [f'Long({b4326[0]-buf_d:.4f},{b4326[2]+buf_d:.4f})',
                          f'Lat({b4326[1]-buf_d:.4f},{b4326[3]+buf_d:.4f})'],
        'FORMAT'       : 'image/tiff',
    }
    try:
        resp = requests.get(wcs_url, params=params, timeout=180)
        ct   = resp.headers.get('Content-Type', '')
        if ('tiff' in ct or 'image' in ct) and len(resp.content) > 50_000:
            NLCD_TIF.parent.mkdir(parents=True, exist_ok=True)
            NLCD_TIF.write_bytes(resp.content)
            print(f'  Saved {NLCD_TIF}  ({NLCD_TIF.stat().st_size/1e6:.1f} MB)')
        else:
            print(f'  ERROR: WCS download failed (status={resp.status_code})')
            print('  Response:', resp.text[:400])
            print('  Please download NLCD 2016 manually from https://www.mrlc.gov/viewer/')
            print(f'  Save to: {NLCD_TIF}')
            sys.exit(1)
    except Exception as e:
        print(f'  Download error: {e}')
        print('  Please download NLCD 2016 manually from https://www.mrlc.gov/viewer/')
        print(f'  Save to: {NLCD_TIF}')
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# 1.  Reproject NLCD to EPSG:26915 if needed
# ─────────────────────────────────────────────────────────────────────────
with rasterio.open(NLCD_TIF) as src:
    src_crs = src.crs

if src_crs and src_crs.to_epsg() == 26915:
    NLCD_26915 = NLCD_TIF
    print(f'  Raster already in EPSG:26915')
else:
    print(f'  Reprojecting {src_crs} → EPSG:26915 ...')
    dst_crs = CRS.from_epsg(26915)
    with rasterio.open(NLCD_TIF) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=30)
        meta = src.meta.copy()
        meta.update(crs=dst_crs, transform=transform,
                    width=width, height=height, nodata=0)
        with rasterio.open(NLCD_26915, 'w', **meta) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=dst_crs,
                      resampling=Resampling.nearest)
    print(f'  Saved reprojected raster: {NLCD_26915}')

# ─────────────────────────────────────────────────────────────────────────
# 2.  Per-parcel dominant NLCD class via exactextract
# ─────────────────────────────────────────────────────────────────────────
print('\nComputing per-parcel dominant NLCD class (exactextract mode) ...')
from exactextract import exact_extract

t0 = time.time()
# pass a minimal GeoDataFrame (exactextract errors on mixed-type columns)
gdf_slim = gdf_raw[['parcel_id', 'geometry']].copy()
ee_result = exact_extract(
    str(NLCD_26915),
    gdf_slim,
    ops=['mode'],
    include_cols=['parcel_id'],
    output='pandas',
    strategy='raster-sequential',
)
print(f'  Done in {time.time()-t0:.0f}s  ({len(ee_result):,} parcels)')

# rename column: exactextract names it 'band_1_mode' or similar
mode_col = [c for c in ee_result.columns if 'mode' in c.lower()][0]
ee_result = ee_result.rename(columns={mode_col: 'nlcd_class'})
ee_result['nlcd_class'] = ee_result['nlcd_class'].fillna(0).astype(int)

# ─────────────────────────────────────────────────────────────────────────
# 3.  Join to feature_matrix, add derived columns, save
# ─────────────────────────────────────────────────────────────────────────
print('Adding NLCD columns to feature_matrix ...')
fm = pd.read_csv(DATA_DIR / 'feature_matrix.csv')
fm = fm.merge(ee_result[['parcel_id','nlcd_class']], on='parcel_id', how='left')
fm['nlcd_class']         = fm['nlcd_class'].fillna(0).astype(int)
fm['nlcd_class_name']    = fm['nlcd_class'].map(NLCD_NAMES).fillna('Unknown')
fm['is_likely_public_green'] = fm['nlcd_class'].isin(PUBLIC_GREEN_CLASSES)

# add archetype from parcel_archetypes_v2 (for convenience)
arc = pd.read_csv(RES_DIR / 'parcel_archetypes_v2.csv')
fm  = fm.merge(arc[['parcel_id','archetype']], on='parcel_id', how='left')

# save updated feature matrix
out_fm = RES_DIR / 'feature_matrix_nlcd.csv'
fm.to_csv(out_fm, index=False)
print(f'  Saved {out_fm}')

# update gpkg
gdf_out = gdf_raw.merge(
    fm[['parcel_id','nlcd_class','nlcd_class_name','is_likely_public_green']],
    on='parcel_id', how='left')
out_gpkg = RES_DIR / 'parcel_archetypes_v2_nlcd.gpkg'
gdf_out.to_file(out_gpkg, driver='GPKG')
print(f'  Saved {out_gpkg}')

# ─────────────────────────────────────────────────────────────────────────
# 4.  Terminal report — NLCD distribution
# ─────────────────────────────────────────────────────────────────────────
N = len(fm)
print()
print('=' * 60)
print(f'NLCD CLASS DISTRIBUTION  (n={N:,} parcels)')
print('=' * 60)
dist = fm.groupby(['nlcd_class','nlcd_class_name']).size().reset_index(name='n')
dist['pct'] = 100 * dist['n'] / N
dist = dist.sort_values('n', ascending=False)
for _, row in dist.iterrows():
    marker = ' *' if row.nlcd_class in PUBLIC_GREEN_CLASSES else ''
    print(f'  {int(row.nlcd_class):3d}  {row.nlcd_class_name:<30s}'
          f'  {int(row.n):7,}  ({row.pct:5.1f}%){marker}')
print('  (* = likely public/green)')

n_pg  = fm['is_likely_public_green'].sum()
pct_pg= 100 * n_pg / N
print()
print(f'  Flagged as likely public/green : {n_pg:,}  ({pct_pg:.1f}%)')
n_res = fm['nlcd_class'].isin(RESIDENTIAL_CLASSES).sum()
print(f'  Flagged as residential (22-24)  : {n_res:,}  ({100*n_res/N:.1f}%)')

print()
print('PUBLIC/GREEN GROUP CHARACTERISTICS')
print('-' * 55)
pg    = fm[fm['is_likely_public_green']]
nonpg = fm[~fm['is_likely_public_green']]
cols  = ['ISA_frac','log_lot_area','dist_to_stream','target_mean']
comp  = pd.DataFrame({
    'Public/green mean' : pg[cols].mean(),
    'Non-public mean'   : nonpg[cols].mean(),
    'Ratio'             : pg[cols].mean() / nonpg[cols].mean(),
})
print(comp.to_string(float_format='{:.3f}'.format))

print()
print('Archetype distribution within public/green:')
print(pg['archetype'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────
# 5.  Step 1 map — NLCD class map, highlighting public/green
# ─────────────────────────────────────────────────────────────────────────
print('\nStep 1 map ...')
present_classes = sorted(fm['nlcd_class'].unique())
present_pg  = [c for c in present_classes if c in PUBLIC_GREEN_CLASSES]
present_dev = [c for c in present_classes if c not in PUBLIC_GREEN_CLASSES]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# left: all parcels coloured by NLCD class
ax = axes[0]
for cls in present_dev:
    mask = gdf_out['nlcd_class'] == cls
    if mask.any():
        clr = NLCD_COLORS.get(cls, '#CCCCCC')
        gdf_out[mask].plot(ax=ax, color=clr, linewidth=0)
for cls in present_pg:
    mask = gdf_out['nlcd_class'] == cls
    if mask.any():
        clr = NLCD_COLORS.get(cls, '#AAFFAA')
        gdf_out[mask].plot(ax=ax, color=clr, linewidth=0)
ax.set_axis_off()
ax.set_title('(a) NLCD 2016 dominant class per parcel')
handles = [mpatches.Patch(color=NLCD_COLORS.get(c,'#CCCCCC'),
                           label=f'{c}: {NLCD_NAMES.get(c,"?")}')
           for c in sorted(present_classes) if c in NLCD_COLORS]
ax.legend(handles=handles, fontsize=6.5, loc='lower right',
          framealpha=0.9, ncol=1, edgecolor='#CCCCCC')

# right: public/green flag
ax = axes[1]
gdf_out[~gdf_out['is_likely_public_green']].plot(
    ax=ax, color='#DDDDDD', linewidth=0, label='Residential / other developed')
gdf_out[gdf_out['is_likely_public_green']].plot(
    ax=ax, color='#267300', linewidth=0, label=f'Likely public/green  (n={n_pg:,}, {pct_pg:.1f}%)')
ax.set_axis_off()
ax.set_title(f'(b) Likely public/green parcels  (n={n_pg:,}, {pct_pg:.1f}%)')
ax.legend(fontsize=9, loc='lower right', framealpha=0.9, edgecolor='#CCCCCC')

fig.suptitle('NLCD 2016 land cover — Brays Bayou parcels', fontsize=11)
fig.tight_layout()
out1 = FIG_DIR / 'step1_public_green_parcels.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f'  [OK] {out1}')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# 6.  ISA-quintile × archetype — all vs residential
# ─────────────────────────────────────────────────────────────────────────
print('Step 2 — ISA quintile heatmaps ...')

res_fm = fm[fm['nlcd_class'].isin(RESIDENTIAL_CLASSES)].copy()
print(f'  Residential subset: {len(res_fm):,} parcels  '
      f'({100*len(res_fm)/N:.1f}%)')

qlabels = ['Q1','Q2','Q3','Q4','Q5']

# global quintiles (all parcels)
fm['isa_q_all'] = pd.qcut(fm['ISA_frac'], q=5, labels=qlabels)
piv_all = (fm.groupby(['archetype','isa_q_all'])['target_mean']
             .mean().unstack().reindex(ARCH_ORDER))
cnt_all = (fm.groupby(['archetype','isa_q_all'])['target_mean']
             .count().unstack().reindex(ARCH_ORDER))

# residential-only quintiles (recomputed on subset)
res_fm['isa_q_res'] = pd.qcut(res_fm['ISA_frac'], q=5, labels=qlabels,
                               duplicates='drop')
piv_res = (res_fm.groupby(['archetype','isa_q_res'])['target_mean']
                  .mean().unstack().reindex(ARCH_ORDER))
cnt_res = (res_fm.groupby(['archetype','isa_q_res'])['target_mean']
                  .count().unstack().reindex(ARCH_ORDER))

def q5q1(piv, arch):
    try:
        return piv.loc[arch,'Q5'] - piv.loc[arch,'Q1']
    except KeyError:
        return np.nan

print()
print('Q5-Q1 deltas (all parcels vs residential only):')
print(f'  {"Archetype":<22}  {"All":>8}  {"Residential":>12}  {"Diff":>8}')
print('  ' + '-'*56)
for a in ARCH_ORDER:
    d_all = q5q1(piv_all, a)
    d_res = q5q1(piv_res, a)
    print(f'  {a:<22}  {d_all:>+8.3f}  {d_res:>+12.3f}  {d_res-d_all:>+8.3f}')

print()
print('Per-quintile N — Residential-only:')
print(cnt_res.to_string())
small_cells = [(a,q) for a in ARCH_ORDER for q in qlabels
               if (not np.isnan(cnt_res.loc[a,q])) and cnt_res.loc[a,q] < 100]
if small_cells:
    print(f'  WARNING: {len(small_cells)} cells with n<100: '
          + ', '.join(f'{a}/{q}' for a,q in small_cells))

# heatmap figure
vmax = max(piv_all.values.max(), piv_res.values.max())
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for ax, piv, cnt, title_sfx, which in [
    (axes[0], piv_all, cnt_all, f'All parcels (n={N:,})',       'all'),
    (axes[1], piv_res, cnt_res, f'Residential only (n={len(res_fm):,})', 'res'),
]:
    im = ax.imshow(piv.values, aspect='auto', cmap='YlOrRd', vmin=1, vmax=vmax)
    ax.set_xticks(range(5)); ax.set_xticklabels(qlabels)
    ax.set_yticks(range(4)); ax.set_yticklabels(ARCH_ORDER, fontsize=8.5)
    ax.set_xlabel('ISA quintile')
    ax.set_title(f'Mean flood depth (ft) — {title_sfx}')

    for i, arch in enumerate(ARCH_ORDER):
        for j, q in enumerate(qlabels):
            try:
                v  = piv.loc[arch, q]
                n  = int(cnt.loc[arch, q])
            except KeyError:
                continue
            c  = 'white' if v > 0.65 * vmax else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7.5, color=c, fontweight='bold')
            # blue border if n<100
            if n < 100:
                ax.add_patch(plt.Rectangle(
                    (j-0.5, i-0.5), 1, 1,
                    fill=False, ec='dodgerblue', lw=2.2, clip_on=True))

    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).ax.set_ylabel('ft', fontsize=8)

axes[1].text(0.5, -0.14, 'Blue border = n < 100', transform=axes[1].transAxes,
             ha='center', fontsize=8, color='dodgerblue', style='italic')

# Q5-Q1 delta annotation bar
for ax_i, (ax, piv) in enumerate([(axes[0], piv_all), (axes[1], piv_res)]):
    deltas = [q5q1(piv, a) for a in ARCH_ORDER]
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([f'Δ={d:+.2f} ft' for d in deltas], fontsize=8)
    ax2.tick_params(right=False)
    ax2.spines['top'].set_visible(False)

fig.suptitle('ISA-quintile × Archetype: All parcels vs Residential-only', fontsize=11)
fig.tight_layout(w_pad=3)
out2 = FIG_DIR / 'step2_paradox_all_vs_residential.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
print(f'\n  [OK] {out2}')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# 7.  Bootstrap 95% CI for Q5-Q1 deltas
# ─────────────────────────────────────────────────────────────────────────
print('Step 3 — Bootstrap CIs (1,000 resamples per archetype) ...')

def bootstrap_q5q1_ci(df, isa_col, target_col='target_mean', n_boot=1000, seed=42):
    """
    Bootstrap 95% CI for Q5 mean - Q1 mean within df.
    Returns (point_estimate, lower, upper).
    """
    rng  = np.random.default_rng(seed)
    q1   = df[df[isa_col] == 'Q1'][target_col].values
    q5   = df[df[isa_col] == 'Q5'][target_col].values
    if len(q1) == 0 or len(q5) == 0:
        return np.nan, np.nan, np.nan
    point = q5.mean() - q1.mean()
    boot  = np.array([
        rng.choice(q5, len(q5), replace=True).mean() -
        rng.choice(q1, len(q1), replace=True).mean()
        for _ in range(n_boot)
    ])
    ci = np.percentile(boot, [2.5, 97.5])
    return point, ci[0], ci[1]

ci_rows = []
for arch in ARCH_ORDER:
    sub_all = fm[fm['archetype'] == arch]
    sub_res = res_fm[res_fm['archetype'] == arch]

    p_all, lo_all, hi_all = bootstrap_q5q1_ci(sub_all, 'isa_q_all')
    p_res, lo_res, hi_res = bootstrap_q5q1_ci(sub_res, 'isa_q_res')

    ci_rows.append(dict(
        archetype=arch,
        delta_all=p_all, lo_all=lo_all, hi_all=hi_all,
        n_all_q1=int((sub_all['isa_q_all']=='Q1').sum()),
        n_all_q5=int((sub_all['isa_q_all']=='Q5').sum()),
        delta_res=p_res, lo_res=lo_res, hi_res=hi_res,
        n_res_q1=int((sub_res['isa_q_res']=='Q1').sum()) if len(sub_res)>0 else 0,
        n_res_q5=int((sub_res['isa_q_res']=='Q5').sum()) if len(sub_res)>0 else 0,
    ))

ci_df = pd.DataFrame(ci_rows)

print()
print('BOOTSTRAP 95% CI — Q5-Q1 delta (ft)')
print('-' * 70)
print(f'{"Archetype":<22}  {"All parcels":>18}  {"Residential only":>20}')
print(f'  {"":22}  {"delta [95% CI]":>18}  {"delta [95% CI]":>20}')
print('  ' + '-'*66)
for _, r in ci_df.iterrows():
    all_str = f'{r.delta_all:+.3f} [{r.lo_all:+.3f}, {r.hi_all:+.3f}]'
    res_str = f'{r.delta_res:+.3f} [{r.lo_res:+.3f}, {r.hi_res:+.3f}]'
    print(f'  {r.archetype:<22}  {all_str:>20}  {res_str:>22}')

# verdict
hs_all_sig = not (ci_df.loc[ci_df.archetype=='Hotspot','lo_all'].values[0] <= 0 <=
                   ci_df.loc[ci_df.archetype=='Hotspot','hi_all'].values[0])
hs_res_sig = not (ci_df.loc[ci_df.archetype=='Hotspot','lo_res'].values[0] <= 0 <=
                   ci_df.loc[ci_df.archetype=='Hotspot','hi_res'].values[0])
hs_res_d   = ci_df.loc[ci_df.archetype=='Hotspot','delta_res'].values[0]

print()
print('VERDICT — ISA paradox in Hotspot archetype:')
if hs_res_sig and hs_res_d < 0:
    verdict = 'PERSISTS in residential subset (95% CI excludes 0)'
elif not hs_res_sig:
    verdict = 'DISAPPEARS / inconclusive in residential subset (CI crosses 0)'
else:
    verdict = 'SIGN REVERSES in residential subset'
print(f'  {verdict}')

# forest plot
fig, ax = plt.subplots(figsize=(9, 5))

y_all = np.arange(4) + 0.18
y_res = np.arange(4) - 0.18
arch_labels = [f'{a}\n(n_all={ci_df.loc[i,"n_all_q1"]+ci_df.loc[i,"n_all_q5"]:,}  '
               f'n_res={ci_df.loc[i,"n_res_q1"]+ci_df.loc[i,"n_res_q5"]:,})'
               for i, a in enumerate(ARCH_ORDER)]

# all parcels
ax.errorbar(
    ci_df['delta_all'], y_all,
    xerr=[ci_df['delta_all']-ci_df['lo_all'],
          ci_df['hi_all']-ci_df['delta_all']],
    fmt='D', color='#555555', ms=8, lw=2, capsize=6, capthick=2,
    label='All parcels', zorder=4,
)
# residential only
ax.errorbar(
    ci_df['delta_res'], y_res,
    xerr=[ci_df['delta_res']-ci_df['lo_res'],
          ci_df['hi_res']-ci_df['delta_res']],
    fmt='o', color='#C8102E', ms=8, lw=2, capsize=6, capthick=2,
    label='Residential only (NLCD 22–24)', zorder=5,
)

ax.axvline(0, lw=1.2, color='black', zorder=3, label='No ISA effect (0)')
ax.set_yticks(np.arange(4))
ax.set_yticklabels(arch_labels, fontsize=8.5)
ax.set_xlabel('Q5 − Q1 mean flood depth (ft)', fontsize=10)
ax.set_title('ISA-paradox robustness check: Q5−Q1 delta with bootstrap 95% CI\n'
             '(negative = higher ISA → lower flood depth)', fontsize=10)
ax.legend(fontsize=9, loc='lower right')
ax.xaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax.invert_yaxis()

# shade significant (CI not crossing 0) in light background
for i, (_, r) in enumerate(ci_df.iterrows()):
    for yi, lo, hi in [(y_all[i], r.lo_all, r.hi_all),
                       (y_res[i], r.lo_res, r.hi_res)]:
        if not (lo <= 0 <= hi):   # significant
            ax.axhspan(yi-0.16, yi+0.16, alpha=0.06, color='green', zorder=1)

fig.tight_layout()
out3 = FIG_DIR / 'step3_paradox_ci_forest.png'
fig.savefig(out3, dpi=150, bbox_inches='tight')
print(f'\n  [OK] {out3}')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('DONE')
print(f'  Updated CSV : {RES_DIR}/feature_matrix_nlcd.csv')
print(f'  Updated GPKG: {RES_DIR}/parcel_archetypes_v2_nlcd.gpkg')
print(f'  Figures     : {FIG_DIR}/')
print('=' * 60)
