#!/usr/bin/env python3
"""
extract_slide_panels.py
Extract three standalone slide-ready PNGs from existing figure sources.

Output 1  outputs/slides/slide7_archetype_map.png   — Fig 3 panel (b), 9"×6", transparent bg
Output 2  outputs/slides/slide8_isa_paradox.png     — copy of fig4_residential_isa_paradox.png
Output 3  outputs/slides/slide10_dose_response.png  — dose-response line plot, 9"×5", transparent bg
"""
import sys, shutil, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

BASE      = Path(__file__).resolve().parents[2]
RES_DIR   = BASE / 'outputs' / 'results'
FIG_DIR   = BASE / 'outputs' / 'figures' / 'paper'
SLIDE_DIR = BASE / 'outputs' / 'slides'
SLIDE_DIR.mkdir(parents=True, exist_ok=True)

RESIDENTIAL_CLASSES = {22, 23, 24}

RES_ARCH_ORDER = [
    'Hotspot_res', 'Lowland_baseline_res',
    'Upland_baseline_res', 'Upland_shield_res',
]
RES_ARCH_COLORS = {
    'Hotspot_res'         : '#C8102E',
    'Lowland_baseline_res': '#E9C46A',
    'Upland_baseline_res' : '#A8DADC',
    'Upland_shield_res'   : '#2E86AB',
}
# Exact percentages from residential archetype run
ARCH_PCT = {
    'Hotspot_res'         : 1.7,
    'Lowland_baseline_res': 15.9,
    'Upland_baseline_res' : 21.7,
    'Upland_shield_res'   : 60.7,
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype' : 42,
})


def report(path):
    kb  = path.stat().st_size // 1024
    img = PILImage.open(path)
    w, h = img.size
    print(f'  ✓ {path.name}  |  {kb:,} KB  |  {w} × {h} px')


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 1 — slide7_archetype_map.png
# Fig 3 panel (b): geographic choropleth of residential archetypes
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print('Output 1: slide7_archetype_map.png')
print('=' * 65)

print('  Loading residential archetype data ...')
fm = pd.read_csv(RES_DIR / 'feature_matrix_nlcd.csv')
ra = pd.read_csv(RES_DIR / 'residential_archetypes.csv',
                 usecols=['parcel_id', 'residential_archetype'])
is_res  = fm['nlcd_class'].isin(RESIDENTIAL_CLASSES)
res_ids = set(fm.loc[is_res, 'parcel_id'].values)

print('  Loading geometry ...')
gpkg_path = RES_DIR / 'parcel_archetypes_v2_nlcd.gpkg'
gdf_all   = gpd.read_file(gpkg_path)
gdf       = (gdf_all[gdf_all['parcel_id'].isin(res_ids)]
             .merge(ra, on='parcel_id', how='inner')
             .to_crs('EPSG:26915'))

nhd_path = BASE / 'data' / 'raw' / 'nhd_streams_brays.gpkg'
nhd = gpd.read_file(nhd_path).to_crs('EPSG:26915') if nhd_path.exists() else None

fig1, ax1 = plt.subplots(figsize=(9, 6))
fig1.patch.set_facecolor('white')
ax1.patch.set_facecolor('white')

# Plot archetypes (low-alpha background ones first, Hotspot on top)
draw_order = ['Upland_shield_res', 'Upland_baseline_res',
              'Lowland_baseline_res', 'Hotspot_res']
for arch in draw_order:
    sub = gdf[gdf['residential_archetype'] == arch]
    if len(sub):
        sub.plot(ax=ax1, color=RES_ARCH_COLORS[arch],
                 linewidth=0, alpha=0.88, rasterized=True)

if nhd is not None:
    nhd.plot(ax=ax1, color='#B0D4F1', linewidth=1.6, zorder=5)

# ── scale bar (EPSG:26915, meters) ────────────────────────────────────────────
xmin, ymin, xmax, ymax = gdf.total_bounds
sb_len   = 2000
sb_x0    = xmin + (xmax - xmin) * 0.06
sb_y0    = ymin + (ymax - ymin) * 0.04
sb_y_txt = sb_y0 + (ymax - ymin) * 0.018
tick_h   = (ymax - ymin) * 0.006
ax1.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
         color='black', lw=3, solid_capstyle='butt', zorder=8)
ax1.plot([sb_x0,          sb_x0],          [sb_y0 - tick_h, sb_y0 + tick_h],
         color='black', lw=1.5, zorder=8)
ax1.plot([sb_x0 + sb_len, sb_x0 + sb_len], [sb_y0 - tick_h, sb_y0 + tick_h],
         color='black', lw=1.5, zorder=8)
ax1.text(sb_x0 + sb_len / 2, sb_y_txt, '2 km',
         ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=8)

# ── north arrow ───────────────────────────────────────────────────────────────
na_x    = xmax - (xmax - xmin) * 0.07
na_y    = ymin + (ymax - ymin) * 0.08
arr_len = (ymax - ymin) * 0.04
ax1.annotate('', xy=(na_x, na_y + arr_len), xytext=(na_x, na_y),
             arrowprops=dict(arrowstyle='->', lw=2.2, color='black'), zorder=8)
ax1.text(na_x, na_y + arr_len * 1.35, 'N',
         ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=8)

# ── legend ────────────────────────────────────────────────────────────────────
handles = [mpatches.Patch(
               color=RES_ARCH_COLORS[a],
               label=f'{a.replace("_res","").replace("_"," ")} ({ARCH_PCT[a]:.1f}%)')
           for a in RES_ARCH_ORDER]
if nhd is not None:
    handles.append(plt.Line2D([0], [0], color='#B0D4F1', lw=2,
                              label='Brays Bayou channel'))
ax1.legend(handles=handles, fontsize=9.5, framealpha=0.92,
           loc='lower right', edgecolor='#CCCCCC',
           facecolor='white')

ax1.set_axis_off()

out1 = SLIDE_DIR / 'slide7_archetype_map.png'
fig1.savefig(out1, dpi=300, bbox_inches='tight')
plt.close(fig1)
report(out1)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 2 — slide8_isa_paradox.png  (direct copy)
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 65)
print('Output 2: slide8_isa_paradox.png')
print('=' * 65)

src2 = FIG_DIR / 'fig4_residential_isa_paradox.png'
out2 = SLIDE_DIR / 'slide8_isa_paradox.png'
shutil.copy(src2, out2)
report(out2)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 3 — slide10_dose_response.png
# Dose-response line plot: % parcels jumping ≥1 risk class vs ISA multiplier
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 65)
print('Output 3: slide10_dose_response.png')
print('=' * 65)

dr = pd.read_csv(RES_DIR / 'dose_response_summary.csv')

# Add baseline row (multiplier=1.0, pct_jumped=0) for each archetype
baselines = pd.DataFrame([
    {'multiplier': 1.0, 'archetype': a, 'pct_jumped': 0.0}
    for a in RES_ARCH_ORDER
])
dr_plot = (pd.concat([baselines, dr[['multiplier','archetype','pct_jumped']]],
                     ignore_index=True)
           .sort_values(['archetype', 'multiplier']))

x_ticks   = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
x_labels  = ['baseline\n(×1.0)', '+20%\n(×1.2)', '+40%\n(×1.4)',
              '+60%\n(×1.6)', '+80%\n(×1.8)', '+100%\n(×2.0)']
LINE_STYLE = {
    'Hotspot_res'         : dict(lw=2.8, ls='-',  marker='o', ms=7),
    'Lowland_baseline_res': dict(lw=2.0, ls='--', marker='s', ms=6),
    'Upland_baseline_res' : dict(lw=2.0, ls='-',  marker='^', ms=6),
    'Upland_shield_res'   : dict(lw=2.0, ls='-.',  marker='D', ms=6),
}
ARCH_LEGEND = {
    'Hotspot_res'         : 'Hotspot_res',
    'Lowland_baseline_res': 'Lowland_baseline_res',
    'Upland_baseline_res' : 'Upland_baseline_res',
    'Upland_shield_res'   : 'Upland_shield_res',
}

TXT = '#212121'   # dark gray for all text on white background

fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor('white')
ax3.patch.set_facecolor('white')

for arch in RES_ARCH_ORDER:
    sub = dr_plot[dr_plot['archetype'] == arch].sort_values('multiplier')
    ax3.plot(sub['multiplier'], sub['pct_jumped'],
             color=RES_ARCH_COLORS[arch],
             label=arch.replace('_res', '').replace('_', ' '),
             **LINE_STYLE[arch],
             zorder=4)

# Saturation vertical line at ×1.4 (+40%)
ax3.axvline(1.4, color='#888888', lw=1.3, ls='--', zorder=3)

x_labels_clean = ['+0%', '+20%', '+40%', '+60%', '+80%', '+100%']
ax3.set_xticks(x_ticks)
ax3.set_xticklabels(x_labels_clean, fontsize=10, color=TXT)
ax3.tick_params(axis='y', labelcolor=TXT, labelsize=10)
ax3.set_ylabel('% parcels jumping ≥1 risk class', fontsize=11, color=TXT)
ax3.set_xlabel('ISA development (×baseline)', fontsize=11, color=TXT)
ax3.set_title('Risk recruitment vs ISA development',
              fontsize=13, fontweight='bold', pad=10, color=TXT)
ax3.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax3.set_xlim(0.95, 2.05)
ax3.set_ylim(bottom=0)
ax3.spines['bottom'].set_color(TXT)
ax3.spines['left'].set_color(TXT)
ax3.tick_params(colors=TXT)
ax3.legend(fontsize=9.5, framealpha=0.92, edgecolor='#CCCCCC',
           facecolor='white', loc='upper left',
           labelcolor=TXT)

# Saturation annotation placed after ylim is finalised
y_top = ax3.get_ylim()[1]
ax3.text(1.405, y_top * 0.98, 'saturation\n(+40%)',
         va='top', ha='left', fontsize=8.5, color='#666666', style='italic')

out3 = SLIDE_DIR / 'slide10_dose_response.png'
fig3.savefig(out3, dpi=300, bbox_inches='tight')
plt.close(fig3)
report(out3)

# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 65)
print('All slide panels saved to:', SLIDE_DIR)
print('=' * 65)
