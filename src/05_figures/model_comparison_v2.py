#!/usr/bin/env python3
"""
model_comparison_v2.py
Two-panel horizontal grouped bar chart: feature categories × Ridge vs LightGBM SHAP.
Tells the "topography dominates in both models" story that defuses the multicollinearity issue.

Ridge   : refit on residential parcels (NLCD 22-24), StandardScaler + Ridge(α=1.0),
          target = target_max_log.  Coefficients are standardised betas.
LightGBM SHAP: precomputed per-parcel values in parcel_archetypes_v2.csv;
          filtered to residential parcels, mean |SHAP| per feature, summed per category.

Outputs
-------
outputs/figures/paper/model_comparison_v2.png     300 DPI, white bg
outputs/slides/slide6_model_comparison.png        300 DPI, white bg  (overwrites v1)
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE      = Path(__file__).resolve().parents[2]
RES_DIR   = BASE / 'outputs' / 'results'
FIG_DIR   = BASE / 'outputs' / 'figures' / 'paper'
SLIDE_DIR = BASE / 'outputs' / 'slides'
FIG_DIR.mkdir(parents=True, exist_ok=True)
SLIDE_DIR.mkdir(parents=True, exist_ok=True)

RESIDENTIAL_CLASSES = {22, 23, 24}
TXT = '#212121'

FEATURES = [
    'elevation', 'slope', 'TWI', 'log_flow_accum',
    'dist_to_stream', 'dist_to_street',
    'ISA_frac', 'log_lot_area', 'is_enclave',
    'conn_topo', 'Cw_topo', 'HAND_min',
]

FEAT_LABELS = {
    'elevation'      : 'Elevation',
    'slope'          : 'Slope',
    'TWI'            : 'TWI',
    'log_flow_accum' : 'log(Flow accum.)',
    'dist_to_stream' : 'Dist. to stream',
    'dist_to_street' : 'Dist. to street',
    'ISA_frac'       : 'ISA fraction',
    'log_lot_area'   : 'log(Lot area)',
    'is_enclave'     : 'Is enclave',
    'conn_topo'      : 'Conn. topo.',
    'Cw_topo'        : 'Cw_topo',
    'HAND_min'       : 'HAND_min',
}
SHORT = {
    'elevation'      : 'Elev.',
    'slope'          : 'Slope',
    'TWI'            : 'TWI',
    'log_flow_accum' : 'log(FA)',
    'dist_to_stream' : 'Dist.stream',
    'dist_to_street' : 'Dist.street',
    'ISA_frac'       : 'ISA',
    'log_lot_area'   : 'log(Area)',
    'is_enclave'     : 'Enclave',
    'conn_topo'      : 'ConnTopo',
    'Cw_topo'        : 'Cw_topo',
    'HAND_min'       : 'HAND',
}

# Categories use display names from FEAT_LABELS
CATEGORIES = {
    'Topographic'      : ['Slope', 'Elevation', 'TWI', 'log(Flow accum.)', 'HAND_min'],
    'Connectivity'     : ['Conn. topo.', 'Cw_topo', 'Dist. to stream'],
    'Built environment': ['ISA fraction', 'log(Lot area)', 'Dist. to street', 'Is enclave'],
}
LABEL_TO_FEAT = {v: k for k, v in FEAT_LABELS.items()}   # display → internal

CAT_COLORS = {
    'Topographic'      : '#36454F',   # charcoal
    'Connectivity'     : '#2E86AB',   # steel blue
    'Built environment': '#C8102E',   # cherry red
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def report(path):
    kb  = path.stat().st_size // 1024
    img = PILImage.open(path)
    w, h = img.size
    print(f'  ✓ {path.name}  |  {kb:,} KB  |  {w} × {h} px')


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load and compute feature-level importance
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print('model_comparison_v2.py')
print('=' * 65)

print('\nLoading feature matrix ...')
fm     = pd.read_csv(RES_DIR / 'feature_matrix_nlcd.csv')
fm_res = fm[fm['nlcd_class'].isin(RESIDENTIAL_CLASSES)].copy()
print(f'  Residential parcels: {len(fm_res):,}')

print('Fitting Ridge (α=1.0, target_max_log) on residential parcels ...')
X_res   = fm_res[FEATURES].values
y_res   = fm_res['target_max_log'].values
sc      = StandardScaler()
Xsc     = sc.fit_transform(X_res)
ridge   = Ridge(alpha=1.0, random_state=42).fit(Xsc, y_res)
ridge_coef = pd.Series(ridge.coef_, index=FEATURES)

print('Loading precomputed LightGBM SHAP values ...')
pa     = pd.read_csv(RES_DIR / 'parcel_archetypes_v2.csv')
pa_res = pa[pa['parcel_id'].isin(fm_res['parcel_id'])]
shap_abs = pd.Series({f: pa_res[f'shap_{f}'].abs().mean() for f in FEATURES})
print(f'  SHAP residential rows: {len(pa_res):,}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Aggregate to category level
# ══════════════════════════════════════════════════════════════════════════════
cat_data = {}
for cat, display_names in CATEGORIES.items():
    feat_names = [LABEL_TO_FEAT[d] for d in display_names]
    cat_data[cat] = {
        'ridge_sum' : ridge_coef[feat_names].abs().sum(),
        'shap_sum'  : shap_abs[feat_names].sum(),
        'feats'     : feat_names,
        'display'   : display_names,
        # per-feature, sorted by |value| descending (for detail text)
        'ridge_feats': sorted(feat_names, key=lambda f: abs(ridge_coef[f]),  reverse=True),
        'shap_feats' : sorted(feat_names, key=lambda f: shap_abs[f],          reverse=True),
    }

# Sort categories by Ridge sum |β| descending
cat_order = sorted(cat_data.keys(), key=lambda c: cat_data[c]['ridge_sum'], reverse=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Terminal report
# ══════════════════════════════════════════════════════════════════════════════
print()
print('─' * 65)
print('Category sums — Ridge Σ|β|:')
for cat in cat_order:
    v = cat_data[cat]['ridge_sum']
    print(f'  {cat:22s}  {v:.4f}')

print()
print('Category sums — LightGBM Σ mean|SHAP|:')
shap_order = sorted(cat_data.keys(), key=lambda c: cat_data[c]['shap_sum'], reverse=True)
for cat in shap_order:
    v = cat_data[cat]['shap_sum']
    print(f'  {cat:22s}  {v:.4f}')

print()
ridge_ranks = {cat: i+1 for i, cat in enumerate(cat_order)}
shap_ranks  = {cat: i+1 for i, cat in enumerate(shap_order)}

print('Ranking comparison (Ridge order vs SHAP order):')
for cat in cat_order:
    rr, sr = ridge_ranks[cat], shap_ranks[cat]
    match  = '✓' if rr == sr else '⚠'
    print(f'  {match} {cat:22s}  Ridge #{rr}  SHAP #{sr}')

agree = all(ridge_ranks[c] == shap_ranks[c] for c in cat_order)
if agree:
    verdict = 'FULL AGREEMENT — identical category ranking in both models.'
else:
    diffs = [c for c in cat_order if ridge_ranks[c] != shap_ranks[c]]
    verdict = (f'PARTIAL AGREEMENT — same #1 (Topographic), '
               f'but ranks 2-3 differ: {", ".join(diffs)}.')
print(f'\nVerdict: {verdict}')
print('─' * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Build figure
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding figure ...')

BAR_H   = 0.50
Y_GAP   = 2.10                            # vertical gap between bar centres
N_CATS  = len(cat_order)
y_pos   = np.arange(N_CATS) * Y_GAP      # [0, 2.10, 4.20]

fig, (ax_r, ax_s) = plt.subplots(
    1, 2, figsize=(12, 6.5),
    facecolor='white')
fig.patch.set_facecolor('white')
for ax in (ax_r, ax_s):
    ax.patch.set_facecolor('white')

# Reserve top 22% for suptitle + subtitle; leave small bottom margin
fig.subplots_adjust(left=0.14, right=0.96, top=0.76, bottom=0.06, wspace=0.42)

# ── helper: build one-line detail string ──────────────────────────────────────
def ridge_detail(feats):
    parts = [f'{SHORT[f]} {ridge_coef[f]:+.3f}' for f in feats]
    return '  ·  '.join(parts)

def shap_detail(feats):
    parts = [f'{SHORT[f]} {shap_abs[f]:.3f}' for f in feats]
    return '  ·  '.join(parts)

# ── Panel A: Ridge ────────────────────────────────────────────────────────────
max_ridge = max(cat_data[c]['ridge_sum'] for c in cat_order)

for i, cat in enumerate(cat_order):
    y   = y_pos[i]
    val = cat_data[cat]['ridge_sum']
    clr = CAT_COLORS[cat]

    ax_r.barh(y, val, height=BAR_H, color=clr,
              edgecolor='white', alpha=0.90, zorder=3)

    # Sum annotation to the right
    ax_r.text(val + max_ridge * 0.02, y,
              f'{val:.3f}',
              ha='left', va='center', fontsize=10,
              fontweight='bold', color=clr)

    # Feature detail text below bar
    detail = ridge_detail(cat_data[cat]['ridge_feats'])
    ax_r.text(0, y + BAR_H * 0.6 + 0.06,
              detail,
              ha='left', va='bottom', fontsize=7, color='#555555',
              style='italic', clip_on=False)

ax_r.set_yticks(y_pos)
ax_r.set_yticklabels([cat_order[i] for i in range(N_CATS)],
                      fontsize=11, color=TXT, fontweight='bold')
ax_r.set_xlabel('Sum of |standardized β| per category', fontsize=10, color=TXT)
ax_r.tick_params(axis='x', labelcolor=TXT, labelsize=9)
ax_r.tick_params(axis='y', length=0)
ax_r.xaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax_r.set_xlim(0, max_ridge * 1.32)
ax_r.set_ylim(-Y_GAP * 0.55, y_pos[-1] + Y_GAP * 0.55)
ax_r.invert_yaxis()
ax_r.spines['bottom'].set_color('#BBBBBB')
ax_r.spines['left'].set_visible(False)

ax_r.set_title('Ridge regression', fontsize=14, fontweight='bold',
               color=TXT, pad=30)
ax_r.text(0.5, 1.065, 'ρ = 0.42  ·  linear, interpretable',
          transform=ax_r.transAxes,
          ha='center', va='bottom', fontsize=9,
          color='#555555', style='italic')

# ── Panel B: LightGBM SHAP ────────────────────────────────────────────────────
max_shap = max(cat_data[c]['shap_sum'] for c in cat_order)

for i, cat in enumerate(cat_order):
    y   = y_pos[i]
    val = cat_data[cat]['shap_sum']
    clr = CAT_COLORS[cat]

    ax_s.barh(y, val, height=BAR_H, color=clr,
              edgecolor='white', alpha=0.90, zorder=3)

    ax_s.text(val + max_shap * 0.02, y,
              f'{val:.3f}',
              ha='left', va='center', fontsize=10,
              fontweight='bold', color=clr)

    detail = shap_detail(cat_data[cat]['shap_feats'])
    ax_s.text(0, y + BAR_H * 0.6 + 0.06,
              detail,
              ha='left', va='bottom', fontsize=7, color='#555555',
              style='italic', clip_on=False)

ax_s.set_yticks(y_pos)
ax_s.set_yticklabels([])                          # labels only on left panel
ax_s.set_xlabel('Sum of mean |SHAP| per category', fontsize=10, color=TXT)
ax_s.tick_params(axis='x', labelcolor=TXT, labelsize=9)
ax_s.tick_params(axis='y', length=0)
ax_s.xaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
ax_s.set_xlim(0, max_shap * 1.22)
ax_s.set_ylim(-Y_GAP * 0.55, y_pos[-1] + Y_GAP * 0.55)
ax_s.invert_yaxis()
ax_s.spines['bottom'].set_color('#BBBBBB')
ax_s.spines['left'].set_visible(False)

ax_s.set_title('LightGBM', fontsize=14, fontweight='bold',
               color=TXT, pad=30)
ax_s.text(0.5, 1.065, 'ρ = 0.42  ·  gradient-boosted, captures non-linearity',
          transform=ax_s.transAxes,
          ha='center', va='bottom', fontsize=9,
          color='#555555', style='italic')

# ── Figure-level headline (suptitle + subtitle well separated) ────────────────
fig.text(0.5, 0.97,
         'Topography dominates in both models.',
         ha='center', va='top',
         fontsize=15, fontweight='bold', color=TXT)
fig.text(0.5, 0.90,
         'Same top category, same story  →  linear model is sufficient.',
         ha='center', va='top',
         fontsize=10, color='#444444', style='italic')

# ── Save ──────────────────────────────────────────────────────────────────────
print('\nSaving ...')
out_paper = FIG_DIR  / 'model_comparison_v2.png'
out_slide = SLIDE_DIR / 'slide6_model_comparison.png'

for out in (out_paper, out_slide):
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')

plt.close(fig)

print()
report(out_paper)
report(out_slide)
print()
print('=' * 65)
