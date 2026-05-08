#!/usr/bin/env python3
"""
regenerate_supp_s3.py
Supplementary Figure S3 — Six-scenario ISA-reduction intervention diagnostic.

Single compact panel (10" × 4.5"):
  X-axis: 3 interventions (depave, permeable, bioswales)
  Paired bars: Hotspot_res (cherry red) and Top-10% residential (shield blue)
  Y-axis: 0–100% fixed — the flat bars are the story

Output:
  outputs/figures/paper/supp_s3_intervention_diagnostic.{png,pdf}
  outputs/figures/paper/fig5_residential/step7_supplementary_interventions.png  (alias)
"""
import sys, shutil
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE     = Path(__file__).resolve().parents[2]
RES_CSV  = BASE / 'outputs' / 'results' / 'intervention_scenarios_residential.csv'
OUT_DIR  = BASE / 'outputs' / 'figures' / 'paper'
ALT_DIR  = OUT_DIR / 'fig5_residential'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG  = OUT_DIR / 'supp_s3_intervention_diagnostic.png'
OUT_PDF  = OUT_DIR / 'supp_s3_intervention_diagnostic.pdf'
ALT_PNG  = ALT_DIR / 'step7_supplementary_interventions.png'

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RES_CSV)
df['pct_escaped'] = df['escaped_top10_res'] / df['n_treated'] * 100

INTERVENTIONS = ['depave', 'permeable', 'bioswales']
GROUPS        = ['hotspot_res', 'top10_res']

INT_LABELS = {
    'depave'   : 'Depave\n(ISA → 0)',
    'permeable': 'Permeable paving\n(ISA × 0.7)',
    'bioswales': 'Bioswales\n(Cw_topo × 0.5)',
}
GROUP_COLORS = {
    'hotspot_res': '#C8102E',   # cherry red
    'top10_res'  : '#2E86AB',   # shield blue
}
GROUP_LABELS = {
    'hotspot_res': 'Hotspot_res (n = 2,020)',
    'top10_res'  : 'Top-10% residential (n = 11,582)',
}

# Build lookup: {(group, intervention): pct_escaped}
vals = {}
for _, row in df.iterrows():
    vals[(row['group'], row['intervention'])] = float(row['pct_escaped'])

# Print the six exact percentages
print('Six scenario percentages:')
all_below_1 = True
for intv in INTERVENTIONS:
    for grp in GROUPS:
        v = vals.get((grp, intv), float('nan'))
        flag = '  <-- ABOVE 1%' if v >= 1.0 else ''
        print(f'  {grp:15s} / {intv:10s}: {v:.4f}%{flag}')
        if v >= 1.0:
            all_below_1 = False
print()
print(f'All bars below 1%: {all_below_1}')
print()

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'        : 'sans-serif',
    'font.size'          : 10,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'pdf.fonttype'       : 42,
    'ps.fonttype'        : 42,
})

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.subplots_adjust(left=0.08, right=0.62, top=0.72, bottom=0.18)

x      = np.arange(len(INTERVENTIONS))
width  = 0.32
offset = [-width / 2, width / 2]

for k, grp in enumerate(GROUPS):
    bar_vals = [vals.get((grp, intv), 0.0) for intv in INTERVENTIONS]
    bars = ax.bar(
        x + offset[k], bar_vals, width=width * 0.90,
        color=GROUP_COLORS[grp], label=GROUP_LABELS[grp],
        edgecolor='white', linewidth=0.8,
        alpha=0.90, zorder=3,
    )
    for bar, v in zip(bars, bar_vals):
        label_y = max(v + 0.8, 2.2)
        ax.text(
            bar.get_x() + bar.get_width() / 2, label_y,
            f'{v:.2f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#222222',
        )

# Y-axis forced 0–100%
ax.set_ylim(0, 100)

# Subtle horizontal gridlines only
ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax.set_axisbelow(True)

# 50% majority-benefit reference line
ax.axhline(50, color='#999999', lw=1.3, ls='--', zorder=2)
ax.text(
    -0.38, 51.8,
    'majority benefit threshold',
    ha='left', va='bottom', fontsize=8.5, color='#888888', style='italic',
)

# X-axis labels (intervention names)
ax.set_xticks(x)
ax.set_xticklabels([INT_LABELS[i] for i in INTERVENTIONS], fontsize=10)
ax.set_ylabel('% of treated parcels with reduced predicted risk', fontsize=10)

# Legend
ax.legend(
    loc='upper left', fontsize=9,
    framealpha=0.92, edgecolor='#CCCCCC',
)

# Title (14pt bold, placed in figure coords) + subtitle (11pt italic)
fig.text(
    0.35, 0.95,
    'Simulated ISA-reduction interventions: % of parcels with reduced predicted risk',
    ha='center', va='bottom', fontsize=14, fontweight='bold',
    transform=fig.transFigure,
)
fig.text(
    0.35, 0.90,
    'Diagnostic confirmation of the impervious paradox — see §4.2 for interpretation',
    ha='center', va='bottom', fontsize=11, color='#444444', style='italic',
    transform=fig.transFigure,
)

# Callout text box (right of axes, rounded background)
fig.text(
    0.645, 0.50,
    'Across all six scenarios, fewer than 1% of treated\n'
    'parcels achieve risk reduction. This is the expected\n'
    'counterfactual signature of the impervious paradox\n'
    'documented in §3.3 — not a prediction of physical\n'
    'intervention outcomes.',
    ha='left', va='center', fontsize=9, color='#333333',
    transform=fig.transFigure,
    bbox=dict(
        boxstyle='round,pad=0.6',
        fc='#F7F7F7', ec='#CCCCCC', alpha=0.95,
    ),
)

# ── save ──────────────────────────────────────────────────────────────────────
print('Saving ...')
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
shutil.copy2(OUT_PNG, ALT_PNG)
plt.close(fig)

png_kb = OUT_PNG.stat().st_size // 1024
pdf_kb = OUT_PDF.stat().st_size // 1024
print(f'  PNG : {OUT_PNG}  ({png_kb:,} KB)')
print(f'  PDF : {OUT_PDF}  ({pdf_kb:,} KB)')
print(f'  alias: {ALT_PNG}')
print()
print(f'All bars below 1%: {all_below_1}')
print('Done.')
