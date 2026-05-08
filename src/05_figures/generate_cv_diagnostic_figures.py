#!/usr/bin/env python3
"""
generate_cv_diagnostic_figures.py
Publication-ready CV diagnostic figures for the Methods section.

CV1 — Per-fold Spearman bar charts (sub-basin + block)
CV2 — Geographic fold maps
CV3 — n_test vs Spearman scatter (dominance / reviewer-defence figure)

Outputs → outputs/figures/paper/cv_diagnostics/   PDF + PNG @ 300 DPI
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
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
TAB_DIR = BASE / 'outputs' / 'tables'
FIG_OUT = BASE / 'outputs' / 'figures' / 'paper' / 'cv_diagnostics'
GPKG    = BASE / 'outputs' / 'results' / 'parcel_archetypes_v2.gpkg'
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ── global style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.size'        : 9,
    'axes.titlesize'   : 10,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 9,
    'legend.fontsize'  : 8,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
})

SUB_PALETTE = {          # per fold, sub-basin CV
    'r27' : '#1B3A6B',   # navy   — dominant fold
    'r75' : '#2E86AB',
    'r122': '#A8DADC',
    'r56' : '#E9C46A',
    'r136': '#F4A261',
    'r68' : '#C8102E',
    'r133': '#8E8E8E',   # grey   — tiny fold
}

def save_fig(fig, stem):
    """Save PDF + PNG 300 DPI, print sizes."""
    pdf = FIG_OUT / f'{stem}.pdf'
    png = FIG_OUT / f'{stem}.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    print(f'  [OK] {stem}.pdf  ({pdf.stat().st_size/1024:.0f} KB)')
    print(f'  [OK] {stem}.png  ({png.stat().st_size/1024:.0f} KB)')

# ─────────────────────────────────────────────────────────────────────────
# Load per-fold metrics
# ─────────────────────────────────────────────────────────────────────────
print('=' * 65)
print('Loading per-fold metrics …')
df_all = pd.read_csv(TAB_DIR / 'per_fold_metrics_ridge.csv')

sub_df = df_all[df_all.fold_scheme == 'fold_subbasin'].copy()
blk_df = df_all[df_all.fold_scheme == 'fold_block'].copy()

# sort sub-basin by n_test descending (largest bar first)
sub_df = sub_df.sort_values('n_test', ascending=False).reset_index(drop=True)
# sort block CV by Spearman descending
blk_df = blk_df.sort_values('spearman', ascending=False).reset_index(drop=True)

# ── aggregate stats (from verify_spearman_scope) ─────────────────────────
def wtd(df):   return float(np.average(df.spearman, weights=df.n_test))
def unwtd(df): return float(df.spearman.mean())

# recompute pooled quickly from stored values is not possible without
# raw predictions; use the numbers from the verify run terminal output
STATS = {
    'sub': {'weighted': wtd(sub_df), 'unweighted': unwtd(sub_df), 'pooled': 0.3932},
    'blk': {'weighted': wtd(blk_df), 'unweighted': unwtd(blk_df), 'pooled': 0.4308},
}
print(f'  Sub-basin: weighted={STATS["sub"]["weighted"]:.4f}  '
      f'unweighted={STATS["sub"]["unweighted"]:.4f}  pooled={STATS["sub"]["pooled"]:.4f}')
print(f'  Block:     weighted={STATS["blk"]["weighted"]:.4f}  '
      f'unweighted={STATS["blk"]["unweighted"]:.4f}  pooled={STATS["blk"]["pooled"]:.4f}')

# ─────────────────────────────────────────────────────────────────────────
# Load geographic data (slim read for speed)
# ─────────────────────────────────────────────────────────────────────────
print('Loading parcel geometries …')
t0  = time.time()
gdf = gpd.read_file(GPKG, columns=['parcel_id','fold_subbasin','fold_block','geometry'])
print(f'  {len(gdf):,} parcels loaded in {time.time()-t0:.0f}s  CRS={gdf.crs}')

# ── per-fold annotation centroids (mean of parcel centroids — no dissolve) ─
print('Computing fold centroids …')
cx = gdf.geometry.centroid.x
cy = gdf.geometry.centroid.y
gdf_c = gdf.assign(cx=cx, cy=cy)
sub_cents = (gdf_c[gdf_c.fold_subbasin.notna()]
             .groupby('fold_subbasin')[['cx','cy']].mean().reset_index())
blk_unique_folds = (gdf_c[gdf_c.fold_block.notna()]['fold_block'].unique())
print(f'  {len(sub_cents)} sub-basin folds  |  {len(blk_unique_folds)} block folds')

# ─────────────────────────────────────────────────────────────────────────
# Helpers — bar chart with width ∝ n_test
# ─────────────────────────────────────────────────────────────────────────

def proportional_bars(ax, df, col_color, stats, title, annotate_r27=False,
                      fold_key='fold_id'):
    """Draw Spearman bars with width proportional to n_test."""
    total_n   = df['n_test'].sum()
    span      = 1.0              # normalised total x-width
    widths    = df['n_test'] / total_n * span
    centers   = (np.cumsum(widths) - widths / 2).values
    heights   = df['spearman'].values
    fids      = df[fold_key].values
    colors    = [col_color.get(f, '#999999') for f in fids]

    bars = ax.bar(centers, heights, width=widths * 0.88,
                  color=colors, alpha=0.92, zorder=3, linewidth=0)

    # n_test label inside each bar (if wide enough)
    for c, h, w, n in zip(centers, heights, widths, df['n_test']):
        if w * span > 0.04:   # skip tiny bars
            ax.text(c, max(h - 0.04, 0.02), f'n={n:,}',
                    ha='center', va='top', fontsize=6.5, color='white',
                    fontweight='bold', rotation=90 if w < 0.10 else 0)

    # reference lines
    ax.axhline(stats['weighted'],   ls='-',  lw=2.0, color='#C8102E',
               label=f'Weighted mean  ρ = {stats["weighted"]:.3f}', zorder=4)
    ax.axhline(stats['unweighted'], ls='--', lw=1.8, color='#2E86AB',
               label=f'Unweighted mean ρ = {stats["unweighted"]:.3f}', zorder=4)
    ax.axhline(stats['pooled'],     ls=':',  lw=1.8, color='#2CA02C',
               label=f'Pooled ρ = {stats["pooled"]:.3f}', zorder=4)

    ax.set_xticks(centers)
    ax.set_xticklabels(fids, fontsize=8.5)
    ax.text(0.5, -0.10, 'bar width ∝ n_test (% of held-out data)',
            transform=ax.transAxes, ha='center', fontsize=7.5, color='#666666',
            style='italic')
    ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
    ax.set_title(title, pad=6)
    ax.set_ylabel('Spearman ρ')
    leg = ax.legend(fontsize=8, loc='upper right', framealpha=0.9,
                    edgecolor='#CCCCCC')

    # annotate r27 if present
    if annotate_r27 and 'r27' in list(fids):
        idx = list(fids).index('r27')
        cx, cy = centers[idx], heights[idx]
        pct = 100 * df.loc[df[fold_key]=='r27', 'n_test'].values[0] / total_n
        ax.annotate(
            f'r27\n{pct:.0f}% of data\nρ = {cy:.3f}',
            xy=(cx, cy), xytext=(cx + 0.18, cy + 0.06),
            fontsize=7.5, color=SUB_PALETTE['r27'],
            arrowprops=dict(arrowstyle='->', lw=0.9,
                            color=SUB_PALETTE['r27']),
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=SUB_PALETTE['r27'], alpha=0.9),
        )
    return bars

# ═════════════════════════════════════════════════════════════════════════
# FIGURE CV1 — Per-fold Spearman bar charts
# ═════════════════════════════════════════════════════════════════════════
print('\n─── Figure CV1 …')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── CV1a: sub-basin ───────────────────────────────────────────────────────
ax = axes[0]
proportional_bars(
    ax, sub_df, SUB_PALETTE, STATS['sub'],
    title='(a) Sub-basin CV  —  7 hydrological folds',
    annotate_r27=True,
)
ax.set_ylim(-0.05, 0.62)

# ── CV1b: block CV ────────────────────────────────────────────────────────
ax = axes[1]

# color block bars by Spearman value (diverging around weighted mean)
blk_rho    = blk_df['spearman'].values
blk_norm   = mcolors.TwoSlopeNorm(
    vmin=blk_rho.min(), vcenter=STATS['blk']['weighted'], vmax=blk_rho.max())
blk_cmap   = mcm.RdYlGn
blk_colors = [blk_cmap(blk_norm(r)) for r in blk_rho]

xpos   = np.arange(len(blk_df))
width  = 0.85
bars   = ax.bar(xpos, blk_df['spearman'], width=width,
                color=blk_colors, alpha=0.90, zorder=3, linewidth=0)

ax.axhline(STATS['blk']['weighted'],   ls='-',  lw=2.0, color='#C8102E',
           label=f'Weighted mean  ρ = {STATS["blk"]["weighted"]:.3f}', zorder=4)
ax.axhline(STATS['blk']['unweighted'], ls='--', lw=1.8, color='#2E86AB',
           label=f'Unweighted mean ρ = {STATS["blk"]["unweighted"]:.3f}', zorder=4)
ax.axhline(STATS['blk']['pooled'],     ls=':',  lw=1.8, color='#2CA02C',
           label=f'Pooled ρ = {STATS["blk"]["pooled"]:.3f}', zorder=4)

ax.set_xticks([])
ax.set_xlabel(f'Block fold (sorted by ρ,  n = {len(blk_df)} blocks shown)')
ax.set_ylabel('Spearman ρ')
ax.set_title(f'(b) Block CV  —  {len(blk_df)} × 2-km spatial blocks', pad=6)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')

# colorbar
sm = plt.cm.ScalarMappable(cmap=blk_cmap, norm=blk_norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
cb.ax.set_ylabel('Spearman ρ', fontsize=8)

# summary inset
blk_stats_txt = (
    f'n_test/fold:  min={blk_df.n_test.min():,}  '
    f'median={blk_df.n_test.median():.0f}  '
    f'max={blk_df.n_test.max():,}\n'
    f'std(ρ) = {blk_df.spearman.std():.3f}   '
    f'range [{blk_df.spearman.min():+.3f}, {blk_df.spearman.max():+.3f}]'
)
ax.text(0.02, 0.04, blk_stats_txt, transform=ax.transAxes,
        ha='left', va='bottom', fontsize=7.5, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#CCCCCC', alpha=0.92))

fig.suptitle('Per-fold Ridge performance under two spatial CV schemes\n'
             'Ridge regression, target = log(1 + max flood depth)  |  12 features',
             fontsize=10.5, y=1.02)
fig.tight_layout(w_pad=3)
save_fig(fig, 'CV1_per_fold_spearman')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════
# FIGURE CV2 — Geographic fold maps
# ═════════════════════════════════════════════════════════════════════════
print('\n─── Figure CV2 …')

# block-fold color index: checkerboard so adjacent folds have different colors
def block_color_idx(fold_id_str, n=20):
    parts = fold_id_str.split('_')
    r, c = int(parts[0]), int(parts[1])
    return (r * 3 + c * 7) % n   # prime multipliers → minimal adjacent repeats

blk_cmap20 = mcm.get_cmap('tab20', 20)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# ── CV2a: sub-basin fold map ───────────────────────────────────────────────
ax = axes[0]
# draw un-assigned parcels first
gdf[gdf.fold_subbasin.isna()].plot(ax=ax, color='#E8E8E8', linewidth=0)
# draw each fold
for fid, color in SUB_PALETTE.items():
    gdf[gdf.fold_subbasin == fid].plot(
        ax=ax, color=color, linewidth=0, alpha=0.88, rasterized=True)

# annotate at mean centroid of each fold
total_n_sub = sub_df['n_test'].sum()
for _, crow in sub_cents.iterrows():
    fid = crow['fold_subbasin']
    if fid not in sub_df['fold_id'].values:
        continue
    n   = sub_df.loc[sub_df.fold_id==fid, 'n_test'].values[0]
    rho = sub_df.loc[sub_df.fold_id==fid, 'spearman'].values[0]
    pct = 100 * n / total_n_sub
    clr = SUB_PALETTE.get(fid, '#333333')
    ax.annotate(
        f'{fid}\nn={n:,} ({pct:.0f}%)\nρ={rho:.3f}',
        xy=(crow.cx, crow.cy), xytext=(crow.cx, crow.cy),
        fontsize=6.8, ha='center', va='center',
        color='white' if fid in ('r27', 'r68', 'r136') else '#222222',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc=clr, ec='white',
                  alpha=0.85, linewidth=0.8),
    )

ax.set_axis_off()
ax.set_title('(a) Sub-basin CV — 7 hydrological folds', pad=6)
legend_patches = [mpatches.Patch(color=SUB_PALETTE[f], label=f)
                  for f in sub_df['fold_id']]
ax.legend(handles=legend_patches, fontsize=7.5, loc='lower right',
          framealpha=0.9, edgecolor='#CCCCCC', title='Fold ID', title_fontsize=7.5)

# ── CV2b: block fold map ───────────────────────────────────────────────────
ax = axes[1]
# assign a cycling color index to each parcel via its block fold ID
blk_cidx_parcel = gdf['fold_block'].map(
    lambda f: block_color_idx(f) if isinstance(f, str) and '_' in str(f) else -1)
# draw unassigned parcels first
gdf[blk_cidx_parcel == -1].plot(ax=ax, color='#E8E8E8', linewidth=0, rasterized=True)
# draw each block fold color group
for cidx in range(20):
    mask = blk_cidx_parcel == cidx
    if mask.any():
        gdf[mask].plot(ax=ax, color=blk_cmap20(cidx),
                       linewidth=0, alpha=0.85, rasterized=True)

ax.set_axis_off()
ax.set_title(f'(b) Block CV — {len(blk_df)} × 2-km blocks  (colors cycle, no annotations)',
             pad=6)
# note box
ax.text(0.02, 0.04,
        f'n_test: min={blk_df.n_test.min():,}  median={blk_df.n_test.median():.0f}  '
        f'max={blk_df.n_test.max():,}\n'
        f'Weighted ρ = {STATS["blk"]["weighted"]:.3f}  '
        f'Pooled ρ = {STATS["blk"]["pooled"]:.3f}',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#CCCCCC', alpha=0.93))

fig.suptitle('Spatial cross-validation fold geometry — Brays Bayou watershed\n'
             'EPSG:26915  |  118,119 residential parcels',
             fontsize=10.5, y=1.01)
fig.tight_layout(w_pad=2)
save_fig(fig, 'CV2_fold_geometry_map')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════
# FIGURE CV3 — n_test vs Spearman scatter (reviewer-defence)
# ═════════════════════════════════════════════════════════════════════════
print('\n─── Figure CV3 …')
fig, ax = plt.subplots(figsize=(8.5, 5.5))

# ── block CV (small dots, unlabeled) ──────────────────────────────────────
blk_rho_arr = blk_df['spearman'].values
blk_n_arr   = blk_df['n_test'].values
sc_blk = ax.scatter(blk_n_arr, blk_rho_arr,
                    s=32, c='#2E86AB', alpha=0.55, zorder=3,
                    linewidths=0, label=f'Block CV  ({len(blk_df)} folds)')

# ── sub-basin (large dots, labeled) ──────────────────────────────────────
for _, row in sub_df.iterrows():
    fid  = row['fold_id']
    clr  = SUB_PALETTE.get(fid, '#999999')
    size = 220 if fid == 'r27' else (60 if fid == 'r133' else 130)
    ax.scatter(row.n_test, row.spearman,
               s=size, c=clr, zorder=6, linewidths=1.2, edgecolors='white')

# ── annotations ───────────────────────────────────────────────────────────
for _, row in sub_df.iterrows():
    fid = row['fold_id']
    pct = 100 * row.n_test / sub_df['n_test'].sum()

    if fid == 'r27':
        offset = (0.08, 0.045)
        txt = (f'r27\n{pct:.0f}% of held-out data\n'
               f'ρ = {row.spearman:.3f}  →  drives weighted mean')
        fc  = SUB_PALETTE['r27']
        tc  = 'white'
    elif fid == 'r133':
        offset = (-0.15, 0.055)
        txt = f'r133  (n={row.n_test:,})\nρ = {row.spearman:.3f}  "small fold = noisy"'
        fc  = '#F0F0F0'
        tc  = '#333333'
    else:
        offset = (0.04, 0.02)
        txt = f'{fid}\nρ={row.spearman:.3f}'
        fc  = 'white'
        tc  = '#333333'

    ax.annotate(
        txt,
        xy=(row.n_test, row.spearman),
        xytext=(row.n_test * (1 + offset[0]), row.spearman + offset[1]),
        fontsize=7.5, color=tc,
        arrowprops=dict(arrowstyle='->', lw=0.9, color='#666666'),
        bbox=dict(boxstyle='round,pad=0.28', fc=fc,
                  ec=SUB_PALETTE.get(fid, '#AAAAAA'), alpha=0.92, linewidth=0.8),
        zorder=7,
    )

# ── reference line at block weighted mean ─────────────────────────────────
ax.axhline(STATS['blk']['weighted'], ls='--', lw=1.8, color='#C8102E', zorder=4,
           label=f'Block CV weighted mean  ρ = {STATS["blk"]["weighted"]:.3f}')
ax.axhline(0, ls='-', lw=0.8, color='#AAAAAA', zorder=2)

# ── axis config ───────────────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_xlabel('n_test  (log scale — parcels held out in that fold)', fontsize=9.5)
ax.set_ylabel('Spearman ρ', fontsize=9.5)
ax.set_title('Per-fold Spearman vs fold size — does block CV give more uniform coverage?\n'
             'Ridge × target_max_log  |  sub-basin (7 large folds) vs block (75 × 2-km folds)',
             pad=8)
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax.xaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0, which='both')

# ── custom legend ─────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#1B3A6B',
           markersize=11, label='Sub-basin CV (7 folds)  — size ∝ n_test'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#2E86AB',
           markersize=6,  label=f'Block CV  ({len(blk_df)} folds)'),
    Line2D([0],[0], ls='--', lw=1.8, color='#C8102E',
           label=f'Block weighted mean  ρ = {STATS["blk"]["weighted"]:.3f}'),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc='lower right',
          framealpha=0.93, edgecolor='#CCCCCC')

# variance annotation box
var_txt = (
    f'Sub-basin std(ρ) = {sub_df.spearman.std():.3f}   '
    f'range [{sub_df.spearman.min():+.3f}, {sub_df.spearman.max():+.3f}]\n'
    f'Block CV   std(ρ) = {blk_df.spearman.std():.3f}   '
    f'range [{blk_df.spearman.min():+.3f}, {blk_df.spearman.max():+.3f}]'
)
ax.text(0.02, 0.04, var_txt, transform=ax.transAxes,
        ha='left', va='bottom', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#CCCCCC', alpha=0.93))

fig.tight_layout()
save_fig(fig, 'CV3_n_test_vs_spearman')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('DONE')
print(f'  Output directory: {FIG_OUT}')
print()
print('Recommended placement:')
print('  CV1_per_fold_spearman  → Methods supplementary Fig S-CV1')
print('  CV2_fold_geometry_map  → Methods supplementary Fig S-CV2')
print('  CV3_n_test_vs_spearman → Main paper Fig 2 (or Methods panel)')
print()
print('Caption talking points:')
print('  • Block CV distributes test data more uniformly (std=0.184 vs 0.131 sub-basin)')
print(f'  • r27 holds {100*sub_df.loc[sub_df.fold_id=="r27","n_test"].values[0]/sub_df.n_test.sum():.0f}% '
      f'of held-out data; drives sub-basin weighted mean to '
      f'{STATS["sub"]["weighted"]:.3f} (unweighted = {STATS["sub"]["unweighted"]:.3f})')
print(f'  • Block and sub-basin pooled ρ agree within 0.04 '
      f'({STATS["sub"]["pooled"]:.3f} vs {STATS["blk"]["pooled"]:.3f}) → '
      f'generalization claim robust')
print('=' * 65)
