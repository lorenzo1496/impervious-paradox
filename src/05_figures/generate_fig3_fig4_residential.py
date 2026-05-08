#!/usr/bin/env python3
"""
generate_fig3_fig4_residential.py
Regenerate Figure 3 (archetype discovery) and Figure 4 (ISA paradox heatmap)
using residential-only parcels (NLCD 22-24) and residential archetypes.

Outputs — PDF + PNG @ 300 DPI
------------------------------
outputs/figures/paper/fig3_residential_archetypes.{pdf,png}
outputs/figures/paper/fig4_residential_isa_paradox.{pdf,png}
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
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import geopandas as gpd
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
RES_DIR = BASE / 'outputs' / 'results'
FIG_DIR = BASE / 'outputs' / 'figures' / 'paper'
PRE_DIR = FIG_DIR / 'preview'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRE_DIR.mkdir(parents=True, exist_ok=True)

RESIDENTIAL_CLASSES = {22, 23, 24}
N_BOOT              = 1_000
RNG                 = np.random.default_rng(42)

RES_ARCH_ORDER = [
    'Hotspot_res', 'Lowland_baseline_res',
    'Upland_baseline_res', 'Upland_shield_res',
]
RES_ARCH_SHORT = ['Hotspot', 'Lowland\nbaseline', 'Upland\nbaseline', 'Upland\nshield']
RES_ARCH_COLORS = {
    'Hotspot_res'         : '#C8102E',
    'Lowland_baseline_res': '#E9C46A',
    'Upland_baseline_res' : '#A8DADC',
    'Upland_shield_res'   : '#2E86AB',
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42,   # editable text in Illustrator
    'ps.fonttype' : 42,
})

def save_fig(fig, stem):
    for ext, dpi in [('pdf', None), ('png', 300)]:
        p = FIG_DIR / f'{stem}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
    p_pre = PRE_DIR / f'{stem}.png'
    fig.savefig(p_pre, dpi=150, bbox_inches='tight')
    pdf_kb  = (FIG_DIR / f'{stem}.pdf').stat().st_size // 1024
    png_kb  = (FIG_DIR / f'{stem}.png').stat().st_size // 1024
    print(f'  [OK] {stem}.pdf ({pdf_kb} KB)  /  {stem}.png ({png_kb} KB)')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load and merge all data
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print('generate_fig3_fig4_residential.py')
print('=' * 65)
print('\nLoading data ...')

fm = pd.read_csv(RES_DIR / 'feature_matrix_nlcd.csv')
ra = pd.read_csv(RES_DIR / 'residential_archetypes.csv',
                 usecols=['parcel_id', 'residential_archetype'])
pa = pd.read_csv(RES_DIR / 'parcel_archetypes_v2.csv',
                 usecols=['parcel_id', 'shap_slope', 'shap_HAND_min'])

# Residential filter
is_res  = fm['nlcd_class'].isin(RESIDENTIAL_CLASSES)
fm_res  = fm[is_res].copy().reset_index(drop=True)

# Master residential dataframe
df = (fm_res
      .merge(ra,  on='parcel_id', how='left')
      .merge(pa,  on='parcel_id', how='left'))

# ISA quintiles — global within residential subset
df['isa_q'] = pd.qcut(df['ISA_frac'], 5, labels=['Q1','Q2','Q3','Q4','Q5'])

N = len(df)
print(f'  Residential parcels : {N:,}')
for a in RES_ARCH_ORDER:
    n = (df['residential_archetype'] == a).sum()
    print(f'    {a:30s}: {n:,}  ({100*n/N:.1f}%)')

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Bootstrap Q5-Q1 CIs  (used in both Fig 4 and terminal report)
# ══════════════════════════════════════════════════════════════════════════════
print('\nBootstrapping Q5-Q1 deltas (1,000 resamples) ...')

ci_rows = []
for arch in RES_ARCH_ORDER:
    sub = df[df['residential_archetype'] == arch]
    q1  = sub.loc[sub['isa_q'] == 'Q1', 'target_mean'].values
    q5  = sub.loc[sub['isa_q'] == 'Q5', 'target_mean'].values
    delta = q5.mean() - q1.mean()
    boot_deltas = np.array([
        RNG.choice(q5, len(q5), replace=True).mean() -
        RNG.choice(q1, len(q1), replace=True).mean()
        for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boot_deltas, [2.5, 97.5])
    ci_rows.append(dict(arch=arch, delta=delta, lo=lo, hi=hi,
                        sig=(lo > 0 or hi < 0),
                        n_q1=len(q1), n_q5=len(q5)))

ci_df = pd.DataFrame(ci_rows).set_index('arch')

print('\n  Q5-Q1 deltas with 95% CI:')
for arch in RES_ARCH_ORDER:
    r   = ci_df.loc[arch]
    sig = 'SIGNIFICANT' if r.sig else 'not sig'
    print(f'    {arch:30s}: {r.delta:+.3f} ft  '
          f'[{r.lo:+.3f}, {r.hi:+.3f}]  {sig}')

# ══════════════════════════════════════════════════════════════════════════════
# 3.  ISA quintile cell sizes  (flag n < 100)
# ══════════════════════════════════════════════════════════════════════════════
count_pivot = (df.groupby(['residential_archetype', 'isa_q'], observed=False)
                 .size().unstack(fill_value=0))
count_pivot = count_pivot.reindex(RES_ARCH_ORDER)

n_flagged = int((count_pivot < 100).values.sum())
print(f'\n  Cells with n < 100: {n_flagged}')
if n_flagged:
    for arch in RES_ARCH_ORDER:
        for q in ['Q1','Q2','Q3','Q4','Q5']:
            n = count_pivot.loc[arch, q]
            if n < 100:
                print(f'    ! {arch} / {q}: n={n}')

mean_pivot = (df.groupby(['residential_archetype', 'isa_q'], observed=False)
                ['target_mean'].mean().unstack())
mean_pivot = mean_pivot.reindex(RES_ARCH_ORDER)

print('\n  Hotspot_res cell sample sizes:')
for q in ['Q1','Q2','Q3','Q4','Q5']:
    n = count_pivot.loc['Hotspot_res', q]
    flag = '  ← n<100' if n < 100 else ''
    print(f'    {q}: n={n:,}{flag}')

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Residential archetype discovery
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 3 ...')

fig3 = plt.figure(figsize=(16, 7.5))
gs3  = gridspec.GridSpec(1, 2, figure=fig3, wspace=0.10,
                         left=0.06, right=0.97, top=0.88, bottom=0.10)
ax3a = fig3.add_subplot(gs3[0, 0])
ax3b = fig3.add_subplot(gs3[0, 1])

# ── 3a: SHAP quadrant scatter ─────────────────────────────────────────────────
# Plot in reverse-priority order (Hotspot_res last, drawn on top)
alpha_map = {
    'Upland_shield_res'   : 0.12,
    'Upland_baseline_res' : 0.20,
    'Lowland_baseline_res': 0.22,
    'Hotspot_res'         : 0.55,
}
size_map = {
    'Upland_shield_res'   : 2,
    'Upland_baseline_res' : 3,
    'Lowland_baseline_res': 3,
    'Hotspot_res'         : 5,
}
draw_order = ['Upland_shield_res', 'Upland_baseline_res',
              'Lowland_baseline_res', 'Hotspot_res']

for arch in draw_order:
    sub = df[df['residential_archetype'] == arch]
    ax3a.scatter(
        sub['shap_slope'], sub['shap_HAND_min'],
        c=RES_ARCH_COLORS[arch], alpha=alpha_map[arch],
        s=size_map[arch], linewidths=0, rasterized=True,
    )

# Cluster centroids
for arch in RES_ARCH_ORDER:
    sub = df[df['residential_archetype'] == arch]
    cx  = sub['shap_slope'].mean()
    cy  = sub['shap_HAND_min'].mean()
    ax3a.scatter(cx, cy, c=RES_ARCH_COLORS[arch],
                 s=160, marker='D', edgecolors='white',
                 linewidths=1.4, zorder=6)
    # Label offset per archetype
    offsets = {
        'Hotspot_res'         : ( 0.12, -0.12),
        'Lowland_baseline_res': (-0.25,  0.10),
        'Upland_baseline_res' : ( 0.10,  0.10),
        'Upland_shield_res'   : (-0.35, -0.12),
    }
    dx, dy = offsets[arch]
    short = arch.replace('_res', '').replace('_', ' ')
    ax3a.text(cx + dx, cy + dy, short,
              fontsize=8.5, fontweight='bold',
              color=RES_ARCH_COLORS[arch],
              ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.2', fc='white',
                        ec=RES_ARCH_COLORS[arch], alpha=0.85, lw=0.8))

# Quadrant reference lines
ax3a.axvline(0, color='#888888', lw=0.9, ls='--', zorder=2)
ax3a.axhline(0, color='#888888', lw=0.9, ls='--', zorder=2)

# Axis labels
ax3a.set_xlabel(
    '← lower exposure    SHAP(slope)    higher exposure →',
    fontsize=9, labelpad=6)
ax3a.set_ylabel(
    '← higher protection    SHAP(HAND_min)    lower protection →',
    fontsize=9, labelpad=6)

# Legend
handles = [mpatches.Patch(color=RES_ARCH_COLORS[a],
           label=f'{a.replace("_res","").replace("_"," ")} (n={int((df.residential_archetype==a).sum()):,})')
           for a in RES_ARCH_ORDER]
ax3a.legend(handles=handles, fontsize=8.5, framealpha=0.92,
            loc='upper right', edgecolor='#CCCCCC')

ax3a.set_title('(a) SHAP quadrant — topographic position of residential archetypes',
               fontsize=10, pad=6)
ax3a.text(0.5, -0.13, f'n = {N:,} residential parcels',
          transform=ax3a.transAxes, ha='center', fontsize=8.5,
          color='#666666', style='italic')

# ── 3b: Geographic choropleth ─────────────────────────────────────────────────
print('  Loading geometry ...')
gpkg_path = RES_DIR / 'parcel_archetypes_v2_nlcd.gpkg'
gdf_all   = gpd.read_file(gpkg_path)
gdf       = (gdf_all[gdf_all['parcel_id']
             .isin(df['parcel_id'])]
             .merge(ra, on='parcel_id', how='inner')
             .to_crs('EPSG:26915'))

# NHD streams
nhd_path = BASE / 'data' / 'raw' / 'nhd_streams_brays.gpkg'
nhd = gpd.read_file(nhd_path).to_crs('EPSG:26915') if nhd_path.exists() else None

# Plot each archetype
for arch in ['Upland_shield_res', 'Upland_baseline_res',
             'Lowland_baseline_res', 'Hotspot_res']:
    sub = gdf[gdf['residential_archetype'] == arch]
    if len(sub):
        sub.plot(ax=ax3b, color=RES_ARCH_COLORS[arch],
                 linewidth=0, alpha=0.85, rasterized=True)

# NHD overlay
if nhd is not None:
    nhd.plot(ax=ax3b, color='#B0D4F1', linewidth=1.4,
             zorder=5, label='Brays Bayou')

# ── scale bar (manual, EPSG:26915 in meters) ─────────────────────────────────
xmin, ymin, xmax, ymax = gdf.total_bounds
sb_len   = 2000          # 2 km in meters
sb_x0    = xmin + (xmax - xmin) * 0.06
sb_y0    = ymin + (ymax - ymin) * 0.04
sb_y_txt = sb_y0 + (ymax - ymin) * 0.018
ax3b.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
          color='black', lw=3, solid_capstyle='butt', zorder=8)
ax3b.plot([sb_x0, sb_x0], [sb_y0 - (ymax-ymin)*0.005, sb_y0 + (ymax-ymin)*0.005],
          color='black', lw=1.5, zorder=8)
ax3b.plot([sb_x0+sb_len, sb_x0+sb_len],
          [sb_y0 - (ymax-ymin)*0.005, sb_y0 + (ymax-ymin)*0.005],
          color='black', lw=1.5, zorder=8)
ax3b.text(sb_x0 + sb_len/2, sb_y_txt, '2 km',
          ha='center', va='bottom', fontsize=8.5, fontweight='bold', zorder=8)

# ── north arrow ───────────────────────────────────────────────────────────────
na_x = xmax - (xmax - xmin) * 0.07
na_y = ymin + (ymax - ymin) * 0.08
arr_len = (ymax - ymin) * 0.04
ax3b.annotate('', xy=(na_x, na_y + arr_len), xytext=(na_x, na_y),
              arrowprops=dict(arrowstyle='->', lw=2, color='black'), zorder=8)
ax3b.text(na_x, na_y + arr_len * 1.3, 'N',
          ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=8)

# Legend
pct_map = {'Hotspot_res':1.7, 'Lowland_baseline_res':15.9,
           'Upland_baseline_res':21.7, 'Upland_shield_res':60.7}
handles3b = [mpatches.Patch(color=RES_ARCH_COLORS[a],
             label=f'{a.replace("_res","").replace("_"," ")} ({pct_map[a]:.1f}%)')
             for a in RES_ARCH_ORDER]
if nhd is not None:
    handles3b.append(plt.Line2D([0],[0], color='#B0D4F1', lw=2,
                                label='Brays Bayou channel'))
ax3b.legend(handles=handles3b, fontsize=8.5, framealpha=0.92,
            loc='lower right', edgecolor='#CCCCCC')

ax3b.set_title('(b) Spatial distribution of residential archetypes',
               fontsize=10, pad=6)
ax3b.set_axis_off()
ax3b.text(0.5, -0.04, 'Spatial distribution of archetypes  |  EPSG:26915',
          transform=ax3b.transAxes, ha='center', fontsize=8.5,
          color='#666666', style='italic')

fig3.suptitle(
    'Four residential parcel archetypes defined by topographic position',
    fontsize=12, fontweight='bold', y=0.97)

save_fig(fig3, 'fig3_residential_archetypes')

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — The ISA paradox
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 4 ...')

# ── layout ────────────────────────────────────────────────────────────────────
fig4 = plt.figure(figsize=(13, 10))
gs4  = gridspec.GridSpec(
    2, 3,
    figure=fig4,
    width_ratios=[5, 0.05, 1.4],   # heatmap | gap | delta column
    height_ratios=[3.2, 1.0],      # heatmap | inset
    hspace=0.40, wspace=0.08,
    left=0.10, right=0.97, top=0.89, bottom=0.09,
)
ax_hm    = fig4.add_subplot(gs4[0, 0])    # main heatmap
ax_delta = fig4.add_subplot(gs4[0, 2])    # Q5-Q1 delta column
ax_inset = fig4.add_subplot(gs4[1, :])    # bottom inset bar chart

# ── colormap scale: 5th–95th percentile of all cell means ────────────────────
all_vals = mean_pivot.values.flatten()
vmin = np.nanpercentile(all_vals, 5)
vmax = np.nanpercentile(all_vals, 95)

# ── main heatmap ──────────────────────────────────────────────────────────────
# Short row labels
row_labels = [a.replace('_res', '').replace('_', '\n') for a in RES_ARCH_ORDER]

im = ax_hm.imshow(
    mean_pivot.values,
    aspect='auto', cmap='YlOrRd',
    vmin=vmin, vmax=vmax,
)

# Cell annotations + hatching for n<100
for i, arch in enumerate(RES_ARCH_ORDER):
    for j, q in enumerate(['Q1','Q2','Q3','Q4','Q5']):
        val = mean_pivot.loc[arch, q]
        n   = count_pivot.loc[arch, q]
        # Text color: white for dark cells
        txt_color = 'white' if val > (vmin + 0.6*(vmax-vmin)) else 'black'
        ax_hm.text(j, i, f'{val:.2f}',
                   ha='center', va='center',
                   fontsize=10.5, fontweight='bold', color=txt_color, zorder=3)
        ax_hm.text(j, i + 0.30, f'n={n:,}',
                   ha='center', va='center',
                   fontsize=7.0, color=txt_color, alpha=0.85, zorder=3)
        # Hatch n<100
        if n < 100:
            rect = mpatches.FancyBboxPatch(
                (j - 0.5, i - 0.5), 1, 1,
                boxstyle='square,pad=0',
                fill=False, hatch='////', edgecolor='#333333',
                linewidth=0, zorder=4)
            ax_hm.add_patch(rect)
            ax_hm.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='#333333', linewidth=2.0, zorder=5))

# Axis ticks
ax_hm.set_xticks(range(5))
ax_hm.set_xticklabels(['Q1\n(lowest ISA)', 'Q2', 'Q3', 'Q4', 'Q5\n(highest ISA)'],
                       fontsize=9.5)
ax_hm.set_yticks(range(4))
ax_hm.set_yticklabels(row_labels, fontsize=9.5, ha='right')
ax_hm.tick_params(length=0)
# Color each y-tick label
for ytick, arch in zip(ax_hm.get_yticklabels(), RES_ARCH_ORDER):
    ytick.set_color(RES_ARCH_COLORS[arch])
    ytick.set_fontweight('bold')

# Colorbar
cb = fig4.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.01)
cb.ax.set_ylabel('Mean flood depth (ft)', fontsize=8.5)
cb.ax.tick_params(labelsize=8)

ax_hm.set_title(
    'Within every residential archetype, the most impervious quintile\n'
    'flooded less than the least  —  ISA paradox',
    fontsize=10.5, pad=8, fontweight='bold')
ax_hm.set_xlabel('ISA quintile (low → high impervious surface fraction)', fontsize=9.5)

# ── delta column ──────────────────────────────────────────────────────────────
ax_delta.set_xlim(0, 1)
ax_delta.set_ylim(-0.5, 3.5)
ax_delta.set_yticks([])
ax_delta.set_xticks([])
ax_delta.spines[:].set_visible(False)

# Header
ax_delta.text(0.5, 3.72, 'Q5 − Q1\nΔ (ft)  [95% CI]',
              ha='center', va='bottom', fontsize=8.5, fontweight='bold',
              transform=ax_delta.get_xaxis_transform())

for i, arch in enumerate(RES_ARCH_ORDER):
    r   = ci_df.loc[arch]
    sig = '*' if r.sig else ''
    y_data = 3 - i   # flip: Hotspot at top (i=0) → y=3
    txt = f'{r.delta:+.2f}{sig}\n[{r.lo:+.2f}, {r.hi:+.2f}]'
    ax_delta.text(0.5, y_data, txt,
                  ha='center', va='center',
                  fontsize=9.0, fontweight='bold',
                  color=RES_ARCH_COLORS[arch])

ax_delta.text(0.5, -0.48,
              '* CI excludes 0', ha='center', va='bottom',
              fontsize=7.5, color='#666666', style='italic',
              transform=ax_delta.transData)

# ── bottom inset: collapsed bar chart across all archetypes ───────────────────
quintiles   = ['Q1','Q2','Q3','Q4','Q5']
all_q_means = [df.loc[df['isa_q']==q, 'target_mean'].mean() for q in quintiles]
all_q_errs  = [df.loc[df['isa_q']==q, 'target_mean'].sem() for q in quintiles]

x = np.arange(5)
bars = ax_inset.bar(x, all_q_means, color='#6B8DB5', edgecolor='white',
                    alpha=0.88, width=0.55, zorder=3)
ax_inset.errorbar(x, all_q_means, yerr=[1.96*e for e in all_q_errs],
                  fmt='none', ecolor='black', elinewidth=1.5,
                  capsize=5, capthick=1.5, zorder=4)

# Value labels
for bar, v in zip(bars, all_q_means):
    ax_inset.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.02,
                  f'{v:.3f}', ha='center', va='bottom',
                  fontsize=9, fontweight='bold', color='#1A3A5C')

# Delta annotation
overall_delta = all_q_means[4] - all_q_means[0]
ax_inset.annotate(
    f'Q5 − Q1 = {overall_delta:+.3f} ft\n(collapsed across all archetypes)',
    xy=(4, all_q_means[4]), xytext=(2.8, all_q_means[4] + 0.05),
    fontsize=8.5, color='#333333',
    arrowprops=dict(arrowstyle='->', lw=1.0, color='#555555'),
    bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCCCCC', alpha=0.9),
)

ax_inset.set_xticks(x)
ax_inset.set_xticklabels(
    ['Q1 (lowest ISA)', 'Q2', 'Q3', 'Q4', 'Q5 (highest ISA)'], fontsize=9.5)
ax_inset.set_ylabel('Mean flood depth (ft)', fontsize=9)
ax_inset.set_title(
    'Collapsed across all residential archetypes: '
    'Q5 (highest ISA) flooded less than Q1 (lowest ISA)',
    fontsize=9.5)
ax_inset.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax_inset.set_ylim(bottom=max(0, min(all_q_means) - 0.15))

fig4.suptitle(
    'Within every residential archetype, the most impervious quintile flooded less than the least\n'
    '(ISA paradox: higher impervious surface → lower mean flood depth)',
    fontsize=11, fontweight='bold', y=0.96)

save_fig(fig4, 'fig4_residential_isa_paradox')

# ══════════════════════════════════════════════════════════════════════════════
# Terminal summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 65)
print('Q5-Q1 DELTAS WITH BOOTSTRAP 95% CI')
print('=' * 65)
for arch in RES_ARCH_ORDER:
    r   = ci_df.loc[arch]
    n_q1, n_q5 = r.n_q1, r.n_q5
    sig = 'SIGNIFICANT (CI excludes zero)' if r.sig else 'inconclusive'
    print(f'  {arch}')
    print(f'    delta = {r.delta:+.3f} ft  [{r.lo:+.3f}, {r.hi:+.3f}]')
    print(f'    n_Q1={int(n_q1):,}  n_Q5={int(n_q5):,}  → {sig}')
print()
print('HOTSPOT_RES CELL SIZES:')
for q in ['Q1','Q2','Q3','Q4','Q5']:
    n = count_pivot.loc['Hotspot_res', q]
    flag = '  ← n<100  [FLAGGED]' if n < 100 else ''
    print(f'  {q}: n={n:,}{flag}')

print()
print('FIGURES SAVED:')
for stem in ['fig3_residential_archetypes', 'fig4_residential_isa_paradox']:
    pdf_kb = (FIG_DIR / f'{stem}.pdf').stat().st_size // 1024
    png_kb = (FIG_DIR / f'{stem}.png').stat().st_size // 1024
    print(f'  {stem}.pdf  {pdf_kb} KB')
    print(f'  {stem}.png  {png_kb} KB')
print('=' * 65)
