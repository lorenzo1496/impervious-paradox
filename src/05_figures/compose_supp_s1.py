#!/usr/bin/env python3
"""
compose_supp_s1.py
Supplementary Figure S1 — Per-fold CV diagnostics (3-panel composite).

Panel A: Per-fold Spearman bar charts (sub-basin left, block right)
Panel B: Geographic maps of CV fold assignments
Panel C: Spearman vs n_test scatter

Output: outputs/figures/paper/cv_diagnostics/CV1.{png,pdf}  @ 300 DPI
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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.cm as mplcm
import matplotlib.colors as mcolors
import geopandas as gpd

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parents[2]
OUT_DIR      = BASE / 'outputs' / 'figures' / 'paper' / 'cv_diagnostics'
OUT_PAPER    = BASE / 'outputs' / 'figures' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Primary output (canonical supplementary figure path)
OUT_PNG      = OUT_PAPER / 'supp_s1_cv_diagnostics.png'
OUT_PDF      = OUT_PAPER / 'supp_s1_cv_diagnostics.pdf'
# Backward-compatible alias for embed_figures.py
OUT_PNG_CV1  = OUT_DIR / 'CV1.png'
OUT_PDF_CV1  = OUT_DIR / 'CV1.pdf'

METRICS_CSV = BASE / 'outputs' / 'tables' / 'per_fold_metrics_ridge.csv'
GPKG        = BASE / 'data' / 'processed' / 'feature_matrix.gpkg'
NHD_PATH    = BASE / 'data' / 'raw' / 'nhd_streams_brays.gpkg'

# Pooled Spearman (requires raw predictions; hardcoded from verify_spearman_scope.py)
POOL_SUB = 0.3932
POOL_BLK = 0.4308

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# ── load metrics ──────────────────────────────────────────────────────────────
df  = pd.read_csv(METRICS_CSV)
sub = df[df.fold_scheme == 'fold_subbasin'].copy()
blk = df[df.fold_scheme == 'fold_block'].copy()

# Compute Spearman summary statistics
w_sub     = sub['n_test'] / sub['n_test'].sum()
WTD_SUB   = float((w_sub * sub['spearman']).sum())
UWTD_SUB  = float(sub['spearman'].mean())

w_blk     = blk['n_test'] / blk['n_test'].sum()
WTD_BLK   = float((w_blk * blk['spearman']).sum())
UWTD_BLK  = float(blk['spearman'].mean())

print(f'Sub-basin  weighted={WTD_SUB:.4f}  unweighted={UWTD_SUB:.4f}  pooled={POOL_SUB}')
print(f'Block      weighted={WTD_BLK:.4f}  unweighted={UWTD_BLK:.4f}  pooled={POOL_BLK}')

# r27 stats
r27       = sub[sub.fold_id == 'r27'].iloc[0]
R27_PCT   = 100 * r27.n_test / sub['n_test'].sum()
R27_RHO   = r27.spearman

# Block fold color function (prime-multiplier cycling, 20 colors)
CMAP20 = mplcm.get_cmap('tab20', 20)
def block_color_idx(fid):
    try:
        r, c = int(fid.split('_')[0]), int(fid.split('_')[1])
        return (r * 3 + c * 7) % 20
    except Exception:
        return 0

# Sub-basin fold color palette (7 distinct colors)
SUB_FOLDS     = sorted(sub['fold_id'].tolist())
SUB_CMAP      = mplcm.get_cmap('tab10', len(SUB_FOLDS))
SUB_COLOR_MAP = {f: SUB_CMAP(i) for i, f in enumerate(SUB_FOLDS)}
SUB_COLOR_MAP['r27'] = '#C8102E'   # override r27 with paper red

# ══════════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 14))
gs  = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[4, 5, 4],
    hspace=0.46,
    left=0.07, right=0.97, top=0.95, bottom=0.05,
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Per-fold Spearman bar charts
# ══════════════════════════════════════════════════════════════════════════════
gs_a    = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                           wspace=0.32, width_ratios=[1, 1.6])
ax_sub  = fig.add_subplot(gs_a[0, 0])
ax_blk  = fig.add_subplot(gs_a[0, 1])

# ── left: sub-basin proportional-width bars ───────────────────────────────────
sub_s = sub.sort_values('n_test', ascending=False).reset_index(drop=True)
total_n   = sub_s['n_test'].sum()
span      = 8.0
log_n     = np.log(sub_s['n_test'].astype(float))
bw        = log_n / log_n.sum() * span   # width ∝ log(n_test) per spec
bc        = np.cumsum(bw) - bw / 2
bar_cols  = [SUB_COLOR_MAP.get(f, '#7F8C8D') for f in sub_s['fold_id']]
edgecols  = ['#880000' if f == 'r27' else 'white' for f in sub_s['fold_id']]
edgewidths= [2.0        if f == 'r27' else 0.5    for f in sub_s['fold_id']]

ax_sub.bar(bc, sub_s['spearman'], width=bw * 0.88,
           color=bar_cols, edgecolor=edgecols, linewidth=edgewidths,
           zorder=3, alpha=0.90)

# Fold labels + % annotation below bar
for xi, row in zip(bc, sub_s.itertuples()):
    pct = 100 * row.n_test / total_n
    ax_sub.text(xi, -0.045, row.fold_id, ha='center', va='top', fontsize=8)
    ax_sub.text(xi, -0.075, f'{pct:.0f}%', ha='center', va='top',
                fontsize=7, color='#555555')

# Reference lines
for val, col, ls, lbl in [
    (WTD_SUB,  '#C8102E', '-',  f'Weighted mean  ρ={WTD_SUB:.3f}'),
    (UWTD_SUB, '#2E86AB', '--', f'Unweighted mean ρ={UWTD_SUB:.3f}'),
    (POOL_SUB, '#228B22', ':',  f'Pooled ρ={POOL_SUB:.3f}'),
]:
    ax_sub.axhline(val, lw=1.8, color=col, ls=ls, label=lbl, zorder=4)

# r27 annotation
r27_idx  = sub_s[sub_s.fold_id == 'r27'].index[0]
r27_x    = float(bc[r27_idx])
r27_rho  = float(sub_s.loc[r27_idx, 'spearman'])
ax_sub.annotate(
    f'r27 = {R27_PCT:.1f}% of test data\nρ = {R27_RHO:.2f}',
    xy=(r27_x, r27_rho), xytext=(r27_x + 1.4, r27_rho - 0.10),
    fontsize=8, color='#C8102E',
    arrowprops=dict(arrowstyle='->', lw=1.0, color='#C8102E'),
    bbox=dict(boxstyle='round,pad=0.25', fc='#FFF0F0', ec='#C8102E', alpha=0.9),
)

ax_sub.set_xlim(-0.3, span + 0.3)
ax_sub.set_ylim(-0.09, max(sub_s.spearman) + 0.14)
ax_sub.set_xticks([])
ax_sub.set_ylabel('Spearman ρ')
ax_sub.set_title('(A-left)  Sub-basin CV — 7 folds\nBar width ∝ log(n_test)  |  % = share of held-out data',
                 fontsize=10)
ax_sub.yaxis.grid(True, lw=0.4, color='#EEEEEE', zorder=0)
ax_sub.legend(fontsize=7.5, loc='upper left', framealpha=0.92, edgecolor='#CCCCCC')

# ── right: block CV sorted bars ───────────────────────────────────────────────
blk_s = blk.sort_values('spearman', ascending=False).reset_index(drop=True)
x_blk = np.arange(len(blk_s))
b_cols = [CMAP20(block_color_idx(str(f))) for f in blk_s['fold_id']]

ax_blk.bar(x_blk, blk_s['spearman'], color=b_cols, alpha=0.80,
           width=0.85, zorder=3, linewidth=0)

for val, col, ls, lbl in [
    (WTD_BLK,  '#C8102E', '-',  f'Weighted mean  ρ={WTD_BLK:.3f}'),
    (UWTD_BLK, '#2E86AB', '--', f'Unweighted mean ρ={UWTD_BLK:.3f}'),
    (POOL_BLK, '#228B22', ':',  f'Pooled ρ={POOL_BLK:.3f}'),
]:
    ax_blk.axhline(val, lw=1.8, color=col, ls=ls, label=lbl, zorder=4)

ax_blk.set_xlim(-1, len(blk_s))
ax_blk.set_xticks([])
ax_blk.set_xlabel(f'Block fold (1–{len(blk_s)}, sorted by ρ descending)', fontsize=9)
ax_blk.set_ylabel('Spearman ρ')
ax_blk.set_title(f'(A-right)  Block CV — {len(blk_s)} folds\n'
                 f'n_test: min={blk_s.n_test.min()}  '
                 f'median={blk_s.n_test.median():.0f}  '
                 f'max={blk_s.n_test.max()}',
                 fontsize=10)
ax_blk.yaxis.grid(True, lw=0.4, color='#EEEEEE', zorder=0)
ax_blk.legend(fontsize=7.5, loc='lower right', framealpha=0.92, edgecolor='#CCCCCC')

# Summary stats box
stats_txt = (
    f'std(ρ) = {blk_s.spearman.std():.3f}\n'
    f'range  [{blk_s.spearman.min():.3f}, {blk_s.spearman.max():.3f}]'
)
ax_blk.text(0.98, 0.97, stats_txt, transform=ax_blk.transAxes,
            ha='right', va='top', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', alpha=0.9))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Geographic maps
# ══════════════════════════════════════════════════════════════════════════════
print('Loading geometry ...')
gs_b       = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                              wspace=0.04)
ax_map_sub = fig.add_subplot(gs_b[0, 0])
ax_map_blk = fig.add_subplot(gs_b[0, 1])

gdf = gpd.read_file(GPKG, columns=['parcel_id','fold_subbasin','fold_block'])
gdf['fold_subbasin'] = gdf['fold_subbasin'].astype(str).replace('nan', None)
gdf['fold_block']    = gdf['fold_block'].astype(str).replace('nan', None)

# NHD streams
nhd = gpd.read_file(NHD_PATH).to_crs('EPSG:26915') if NHD_PATH.exists() else None

bounds     = gdf.total_bounds          # [xmin, ymin, xmax, ymax]
xmin, ymin, xmax, ymax = bounds
span_x     = xmax - xmin
span_y     = ymax - ymin

def add_scalebar_north(ax, bounds):
    xmn, ymn, xmx, ymx = bounds
    sx   = xmn + (xmx-xmn)*0.05
    sy   = ymn + (ymx-ymn)*0.04
    slen = 2000
    ax.plot([sx, sx+slen], [sy, sy], 'k-', lw=2.5, solid_capstyle='butt', zorder=9)
    ax.plot([sx,sx],           [sy-(ymx-ymn)*0.004, sy+(ymx-ymn)*0.004], 'k-', lw=1.5, zorder=9)
    ax.plot([sx+slen,sx+slen], [sy-(ymx-ymn)*0.004, sy+(ymx-ymn)*0.004], 'k-', lw=1.5, zorder=9)
    ax.text(sx+slen/2, sy+(ymx-ymn)*0.015, '2 km', ha='center', va='bottom',
            fontsize=8, fontweight='bold', zorder=9)
    # North arrow
    nx = xmx - (xmx-xmn)*0.07
    ny = ymn + (ymx-ymn)*0.07
    al = (ymx-ymn)*0.04
    ax.annotate('', xy=(nx, ny+al), xytext=(nx, ny),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'), zorder=9)
    ax.text(nx, ny+al*1.35, 'N', ha='center', va='bottom',
            fontsize=9, fontweight='bold', zorder=9)

# ── left map: sub-basin folds ─────────────────────────────────────────────────
# Unassigned grey
gdf[gdf['fold_subbasin'].isna()].plot(
    ax=ax_map_sub, color='#E0E0E0', linewidth=0, rasterized=True)

# Per-fold coloring + centroid labels
spearman_map = dict(zip(sub['fold_id'], sub['spearman']))

for fold in SUB_FOLDS:
    sub_gdf = gdf[gdf['fold_subbasin'] == fold]
    if len(sub_gdf) == 0:
        continue
    sub_gdf.plot(ax=ax_map_sub, color=SUB_COLOR_MAP[fold],
                 linewidth=0, alpha=0.88, rasterized=True)
    # Centroid label (mean of parcel centroids — avoids dissolve topology errors)
    cx = sub_gdf.geometry.centroid.x.mean()
    cy = sub_gdf.geometry.centroid.y.mean()
    rho = spearman_map.get(fold, float('nan'))
    ax_map_sub.text(cx, cy, f'{fold}\nρ={rho:.2f}',
                    ha='center', va='center', fontsize=7.5, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc=SUB_COLOR_MAP[fold],
                              ec='white', alpha=0.85, lw=0.7),
                    zorder=6)

if nhd is not None:
    nhd.plot(ax=ax_map_sub, color='#6EC6F0', linewidth=1.4, zorder=7)

add_scalebar_north(ax_map_sub, bounds)
ax_map_sub.set_title('(B-left)  Sub-basin CV — 7 folds', fontsize=10, pad=4)
ax_map_sub.set_axis_off()

# ── right map: block folds ────────────────────────────────────────────────────
gdf[gdf['fold_block'].isna()].plot(
    ax=ax_map_blk, color='#E0E0E0', linewidth=0, rasterized=True)

for cidx in range(20):
    mask = gdf['fold_block'].apply(
        lambda f: block_color_idx(str(f)) == cidx
                  if (f is not None and '_' in str(f)) else False)
    if mask.any():
        gdf[mask].plot(ax=ax_map_blk, color=CMAP20(cidx),
                       linewidth=0, alpha=0.85, rasterized=True)

if nhd is not None:
    nhd.plot(ax=ax_map_blk, color='#6EC6F0', linewidth=1.4, zorder=7)

add_scalebar_north(ax_map_blk, bounds)
n_blk_folds = gdf['fold_block'].nunique()
ax_map_blk.set_title(f'(B-right)  2-km Block CV — {n_blk_folds} folds\n'
                     '(colors cycle; ~3–4 folds per shade)', fontsize=10, pad=4)
ax_map_blk.set_axis_off()

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Spearman vs n_test scatter
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[2])

# Block dots (small, grey-toned)
ax_c.scatter(blk['n_test'], blk['spearman'],
             c=[CMAP20(block_color_idx(str(f))) for f in blk['fold_id']],
             s=30, alpha=0.55, linewidths=0, zorder=3,
             label=f'Block folds (n={len(blk)})')

# Sub-basin circles (large, colored)
for _, row in sub.iterrows():
    col  = SUB_COLOR_MAP.get(row.fold_id, '#7F8C8D')
    ms   = 180 if row.fold_id == 'r27' else 90
    edge = '#880000' if row.fold_id == 'r27' else 'white'
    ax_c.scatter(row.n_test, row.spearman,
                 c=col, s=ms, zorder=5,
                 edgecolors=edge, linewidths=1.5)

# Labels for sub-basin folds
offsets = {
    'r27' : ( 0.10, -0.055),
    'r75' : ( 0.08,  0.030),
    'r122': ( 0.08, -0.045),
    'r56' : (-0.30,  0.030),
    'r136': ( 0.08,  0.030),
    'r68' : (-0.35, -0.030),
    'r133': ( 0.08,  0.030),
}
for _, row in sub.iterrows():
    dx, dy = offsets.get(row.fold_id, (0.06, 0.02))
    ax_c.annotate(
        f'{row.fold_id}\n(n={row.n_test:,}, ρ={row.spearman:.3f})',
        xy=(row.n_test, row.spearman),
        xytext=(row.n_test * (10**dx), row.spearman + dy),
        fontsize=7.5, color=SUB_COLOR_MAP.get(row.fold_id, '#333333'),
        arrowprops=dict(arrowstyle='->', lw=0.8,
                        color=SUB_COLOR_MAP.get(row.fold_id, '#888888')),
    )

# r27 extra annotation
ax_c.annotate(
    f'r27: {R27_PCT:.1f}% of\nheld-out data',
    xy=(r27.n_test, r27.spearman),
    xytext=(r27.n_test * 0.18, r27.spearman + 0.09),
    fontsize=8, color='#C8102E', fontweight='bold',
    arrowprops=dict(arrowstyle='->', lw=1.2, color='#C8102E'),
    bbox=dict(boxstyle='round,pad=0.25', fc='#FFF0F0', ec='#C8102E', alpha=0.9),
)

# r133 outlier annotation
r133 = sub[sub.fold_id == 'r133'].iloc[0]
ax_c.annotate(
    'r133: small fold\n= noisy estimate',
    xy=(r133.n_test, r133.spearman),
    xytext=(r133.n_test * 3.0, r133.spearman - 0.10),
    fontsize=8, color='#555555',
    arrowprops=dict(arrowstyle='->', lw=0.9, color='#888888'),
    bbox=dict(boxstyle='round,pad=0.25', fc='#F5F5F5', ec='#AAAAAA', alpha=0.9),
)

# Reference lines
ax_c.axhline(WTD_BLK,  lw=1.6, color='#C8102E', ls='-',
             label=f'Block weighted mean  ρ={WTD_BLK:.3f}', zorder=4)
ax_c.axhline(WTD_SUB,  lw=1.6, color='#2E86AB', ls='--',
             label=f'Sub-basin weighted mean  ρ={WTD_SUB:.3f}', zorder=4)
ax_c.axhline(0, lw=0.8, color='black', ls='-', alpha=0.3, zorder=2)

# Legend: sub-basin circles as custom patches
sub_handles = [
    plt.scatter([], [], c=SUB_COLOR_MAP.get(f, '#333333'), s=90,
                edgecolors='white' if f != 'r27' else '#880000',
                linewidths=1.5, label=f)
    for f in SUB_FOLDS
]
sub_legend = ax_c.legend(handles=sub_handles,
                          title='Sub-basin folds', title_fontsize=8,
                          fontsize=7.5, loc='lower right',
                          framealpha=0.92, edgecolor='#CCCCCC',
                          ncol=2)
ax_c.add_artist(sub_legend)
ax_c.legend(fontsize=8.5, loc='upper left', framealpha=0.92, edgecolor='#CCCCCC')

ax_c.set_xscale('log')
ax_c.set_xlabel('n_test (log scale)', fontsize=10)
ax_c.set_ylabel('Spearman ρ', fontsize=10)
ax_c.set_title('(C)  Spearman ρ vs fold size\n'
               'Large dot = sub-basin fold  |  small dot = block fold',
               fontsize=10)
ax_c.yaxis.grid(True, lw=0.4, color='#EEEEEE', zorder=0)
ax_c.xaxis.grid(True, lw=0.4, color='#EEEEEE', zorder=0, which='both')

# ══════════════════════════════════════════════════════════════════════════════
# Overall title
# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    'Supplementary Figure S1 — Per-fold cross-validation diagnostics\n'
    'Ridge × target_max_log  |  7 sub-basin folds  |  75 block folds',
    fontsize=12, fontweight='bold', y=0.985,
)

# ── save ──────────────────────────────────────────────────────────────────────
print('Saving figures ...')
fig.savefig(OUT_PNG,     dpi=300, bbox_inches='tight')
fig.savefig(OUT_PDF,     bbox_inches='tight')
# Backward-compatible copies for embed_figures.py
import shutil
shutil.copy2(OUT_PNG, OUT_PNG_CV1)
shutil.copy2(OUT_PDF, OUT_PDF_CV1)
plt.close(fig)

png_kb  = OUT_PNG.stat().st_size // 1024
pdf_kb  = OUT_PDF.stat().st_size // 1024
print()
print('=' * 60)
print('DONE')
print(f'  PNG  : {OUT_PNG}  ({png_kb:,} KB)')
print(f'  PDF  : {OUT_PDF}  ({pdf_kb:,} KB)')
print(f'  alias: {OUT_PNG_CV1}')
print(f'  alias: {OUT_PDF_CV1}')
print()
print('Panel summary:')
print(f'  A: Bar charts — sub-basin {len(sub)} folds (proportional width) + '
      f'block {len(blk)} folds (sorted by ρ)')
print(f'  B: Maps — {gdf.fold_subbasin.nunique()} sub-basin regions + '
      f'{gdf.fold_block.nunique()} block cells')
print(f'  C: Scatter — ρ vs n_test (log), r27 and r133 annotated')
print()
print('→  Now re-run embed_figures.py to embed S1 into the paper draft.')
print('=' * 60)
