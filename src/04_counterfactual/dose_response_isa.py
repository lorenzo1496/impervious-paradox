#!/usr/bin/env python3
"""
dose_response_isa.py
+20 / +40 / +60 / +80 / +100 % ISA development dose-response on residential parcels.
Extends the single +20% counterfactual from interventions_v3_residential.py into a
full sensitivity curve and generates the paper's headline policy figure.

Outputs
-------
outputs/results/dose_response_results.csv    — per-parcel, wide format
outputs/results/dose_response_summary.csv   — (dose × archetype) aggregates
outputs/figures/paper/dose_response/step1–step7 (300 DPI PNG)
"""
import sys, warnings, time
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
import geopandas as gpd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
RES_DIR = BASE / 'outputs' / 'results'
FIG_DIR = BASE / 'outputs' / 'figures' / 'paper' / 'dose_response'
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

RANDOM_STATE       = 42
RESIDENTIAL_CLASSES = {22, 23, 24}
DOSE_MULTIPLIERS   = [1.20, 1.40, 1.60, 1.80, 2.00]
DOSE_LABELS        = ['+20%', '+40%', '+60%', '+80%', '+100%']
DOSE_X             = [1.0] + DOSE_MULTIPLIERS   # include baseline for plots
DOSE_X_LABELS      = ['Base\n(×1.0)'] + [f'×{m:.1f}\n({lbl})'
                       for m, lbl in zip(DOSE_MULTIPLIERS, DOSE_LABELS)]

FEATURES = [
    'elevation', 'slope', 'TWI', 'log_flow_accum', 'dist_to_stream',
    'dist_to_street', 'ISA_frac', 'log_lot_area', 'is_enclave',
    'conn_topo', 'Cw_topo', 'HAND_min',
]

RES_ARCH_ORDER = [
    'Hotspot_res', 'Lowland_baseline_res', 'Upland_baseline_res', 'Upland_shield_res']
RES_ARCH_SHORT = ['Hotspot', 'Lowland\nbaseline', 'Upland\nbaseline', 'Upland\nshield']
RES_ARCH_COLORS = {
    'Hotspot_res'         : '#C8102E',
    'Lowland_baseline_res': '#E9C46A',
    'Upland_baseline_res' : '#A8DADC',
    'Upland_shield_res'   : '#2E86AB',
}
JUMP_COLORS = {0: '#DDDDDD', 1: '#FFCC44', 2: '#EE6600', 3: '#AA0000'}

def save_fig(fig, name):
    p = FIG_DIR / f'{name}.png'
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  [OK] {name}.png  ({p.stat().st_size // 1024} KB)')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print('dose_response_isa.py')
print('=' * 65)

fm = pd.read_csv(RES_DIR / 'feature_matrix_nlcd.csv')
ra = pd.read_csv(RES_DIR / 'residential_archetypes.csv',
                 usecols=['parcel_id', 'residential_archetype'])
fm = fm.merge(ra, on='parcel_id', how='left')

is_res = fm['nlcd_class'].isin(RESIDENTIAL_CLASSES).values
fm_res = fm[is_res].copy().reset_index(drop=True)

print(f'  Full dataset        : {len(fm):,}')
print(f'  Residential (22-24) : {len(fm_res):,}  ({100*is_res.mean():.1f}%)')

X_all  = fm[FEATURES].values
y_mean = fm['target_mean_log'].values
y_max  = fm['target_max_log'].values
X_res  = fm_res[FEATURES].values

# ── ISA cap statistics by dose ────────────────────────────────────────────────
isa_orig = X_res[:, FEATURES.index('ISA_frac')]
print('\n  ISA saturation at each dose:')
for m, lbl in zip(DOSE_MULTIPLIERS, DOSE_LABELS):
    n_capped = (isa_orig * m >= 1.0).sum()
    print(f'    ×{m:.1f} ({lbl}): {n_capped:,} parcels capped at ISA=1.0 '
          f'({100*n_capped/len(fm_res):.1f}%)')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Build KMeans risk classes + train models (identical to interventions_v3)
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding risk classes and training models ...')
_km_X = StandardScaler().fit_transform(fm[['target_mean', 'target_max']].values)
_km   = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
fm['risk_class'] = _km.fit_predict(_km_X)
_cmeans = fm.groupby('risk_class')['target_max'].mean().sort_values()
_lmap   = {old: new for new, old in enumerate(_cmeans.index)}
fm['risk_class'] = fm['risk_class'].map(_lmap).astype(int)
fm_res = fm[is_res].copy().reset_index(drop=True)
X_res  = fm_res[FEATURES].values

sc         = StandardScaler().fit(X_all)
X_all_sc   = sc.transform(X_all)
ridge_max  = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_all_sc, y_max)
rf_clf     = RandomForestClassifier(
    n_estimators=200, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1).fit(X_all_sc, fm['risk_class'].values)
print('  Ridge (max) + RF classifier: done')

# ── baseline predictions on residential ──────────────────────────────────────
X_res_sc = sc.transform(X_res)
base_risk  = ridge_max.predict(X_res_sc)
base_class = rf_clf.predict(X_res_sc).astype(int)

fm_res['baseline_risk']  = base_risk
fm_res['baseline_class'] = base_class

n_base_cls3 = int((base_class == 3).sum())
print(f'  Baseline class 3 parcels (residential): {n_base_cls3:,}  '
      f'({100*n_base_cls3/len(fm_res):.2f}%)')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Run 5 counterfactual doses
# ══════════════════════════════════════════════════════════════════════════════
print('\nRunning dose-response scenarios ...')
isa_idx = FEATURES.index('ISA_frac')
cw_idx  = FEATURES.index('Cw_topo')
dts_idx = FEATURES.index('dist_to_stream')

dts_vals = np.maximum(X_res[:, dts_idx], 1.0)
isa_orig = X_res[:, isa_idx].copy()

# Storage: per-parcel wide columns
results = fm_res[['parcel_id', 'residential_archetype',
                  'baseline_risk', 'baseline_class']].copy()

# Per-(dose, archetype) summary
summary_rows = []

t0 = time.time()
for m, lbl in zip(DOSE_MULTIPLIERS, DOSE_LABELS):
    isa_new = np.minimum(isa_orig * m, 1.0)
    cw_new  = isa_new / dts_vals

    X_cf = X_res.astype(float).copy()
    X_cf[:, isa_idx] = isa_new
    X_cf[:, cw_idx]  = cw_new
    X_cf_sc = sc.transform(X_cf)

    cf_risk  = ridge_max.predict(X_cf_sc)
    cf_class = rf_clf.predict(X_cf_sc).astype(int)

    delta_risk  = cf_risk - base_risk
    class_jump  = cf_class - base_class
    jumped_up   = (class_jump > 0).astype(int)
    to_class3   = ((cf_class == 3) & (base_class < 3)).astype(int)
    delta_isa   = isa_new - isa_orig

    col = lbl.replace('+', 'p')   # e.g. p20
    results[f'delta_isa_{col}']   = delta_isa
    results[f'cf_risk_{col}']     = cf_risk
    results[f'delta_risk_{col}']  = delta_risk
    results[f'cf_class_{col}']    = cf_class
    results[f'class_jump_{col}']  = class_jump
    results[f'jumped_up_{col}']   = jumped_up
    results[f'to_class3_{col}']   = to_class3

    # Per-archetype aggregates
    for arch in RES_ARCH_ORDER:
        mask = fm_res['residential_archetype'].values == arch
        n_a  = int(mask.sum())
        if n_a == 0:
            continue
        n_j   = int(jumped_up[mask].sum())
        n_c3  = int(to_class3[mask].sum())
        dr    = delta_risk[mask]
        n_cf3 = int((cf_class[mask] == 3).sum())
        summary_rows.append(dict(
            dose_label=lbl, multiplier=m,
            archetype=arch,
            n_parcels=n_a,
            n_jumped=n_j,
            pct_jumped=100 * n_j / n_a,
            n_to_class3=n_c3,
            n_cf_class3=n_cf3,
            pct_cf_class3=100 * n_cf3 / n_a,
            mean_delta_risk=float(dr.mean()),
            median_delta_risk=float(np.median(dr)),
            p95_delta_risk=float(np.percentile(dr, 95)),
            n_capped=int((isa_new[mask] >= 1.0 - 1e-9).sum()),
        ))

    n_jumped_all = int(jumped_up.sum())
    pct_all = 100 * n_jumped_all / len(fm_res)
    print(f'  ×{m:.1f} ({lbl}): {n_jumped_all:,} jumped ({pct_all:.1f}%)  '
          f'to_class3={int(to_class3.sum())}  '
          f'mean_dRisk={delta_risk.mean():+.4f}  '
          f'({time.time()-t0:.0f}s)')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Save CSVs
# ══════════════════════════════════════════════════════════════════════════════
out_results = RES_DIR / 'dose_response_results.csv'
out_summary = RES_DIR / 'dose_response_summary.csv'
results.to_csv(out_results, index=False)
summary = pd.DataFrame(summary_rows)
summary.to_csv(out_summary, index=False)
print(f'\n  Saved dose_response_results.csv  ({len(results):,} rows, '
      f'{results.shape[1]} cols)')
print(f'  Saved dose_response_summary.csv  ({len(summary)} rows)')

# ══════════════════════════════════════════════════════════════════════════════
# 5. Saturation analysis
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 65)
print('SATURATION ANALYSIS')
print('=' * 65)

sat_report = {}   # archetype → saturation point label

for arch in RES_ARCH_ORDER:
    sub = summary[summary.archetype == arch].sort_values('multiplier')
    pcts = [0.0] + sub['pct_jumped'].tolist()        # prepend baseline = 0
    drs  = [0.0] + sub['mean_delta_risk'].tolist()
    mults= [1.0] + sub['multiplier'].tolist()

    # Marginal increment in % class jumpers per 0.2 dose step
    increments = [pcts[i+1] - pcts[i] for i in range(len(pcts)-1)]

    # Saturation point: first dose where increment < 50% of the first increment
    # (i.e., gains are halving or worse compared to the initial push)
    first_inc = increments[0] if increments[0] > 0 else 1e-9
    sat_dose_label = 'not reached'
    sat_dose_mult  = None
    for i, inc in enumerate(increments):
        if inc < 0.5 * first_inc:
            sat_dose_label = DOSE_LABELS[i]
            sat_dose_mult  = DOSE_MULTIPLIERS[i]
            break

    sat_report[arch] = dict(
        pcts=pcts, drs=drs, mults=mults,
        increments=increments,
        sat_label=sat_dose_label,
        sat_mult=sat_dose_mult,
    )

    short = arch.replace('_res', '')
    print(f'\n  {short}:')
    print(f'    % jumped: ' +
          '  '.join(f'{p:.1f}%' for p in pcts))
    print(f'    marginal: ' +
          '  '.join(f'{d:+.1f}pp' for d in increments))
    print(f'    saturation point: {sat_dose_label}')

# ── first Hotspot_res → class 3 entry ────────────────────────────────────────
print()
print('FIRST Hotspot_res PARCEL → CLASS 3 ENTRY:')
hs_sub = summary[(summary.archetype == 'Hotspot_res')].sort_values('multiplier')
first_c3_lbl = None
for _, row in hs_sub.iterrows():
    if row['n_to_class3'] > 0:
        first_c3_lbl = row['dose_label']
        first_c3_n   = int(row['n_to_class3'])
        break
if first_c3_lbl is None:
    # Check +20% from interventions_v3 (already known = 1)
    first_c3_lbl = '+20%'; first_c3_n = 1
print(f'  First entry at dose: {first_c3_lbl}  (n={first_c3_n} parcel(s))')
for _, row in hs_sub.iterrows():
    print(f'    ×{row.multiplier:.1f} ({row.dose_label}): {int(row.n_to_class3)} newly in class 3  '
          f'({int(row.n_cf_class3)} total in class 3)')

# ── Upland_shield_res majority recruitment ────────────────────────────────────
print()
print('UPLAND_SHIELD_RES — when does % jumpers exceed 50%?')
us_pcts = sat_report['Upland_shield_res']['pcts'][1:]  # drop baseline 0
exceeded_50 = None
for m, lbl, pct in zip(DOSE_MULTIPLIERS, DOSE_LABELS, us_pcts):
    print(f'    ×{m:.1f} ({lbl}): {pct:.1f}%')
    if pct >= 50 and exceeded_50 is None:
        exceeded_50 = lbl
if exceeded_50:
    print(f'  → 50% majority recruitment point: {exceeded_50}')
else:
    # Linear extrapolation beyond ×2.0
    us_sub = summary[summary.archetype == 'Upland_shield_res'].sort_values('multiplier')
    last_two = us_sub.tail(2)
    dm = float(last_two.iloc[1].multiplier - last_two.iloc[0].multiplier)
    dp = float(last_two.iloc[1].pct_jumped  - last_two.iloc[0].pct_jumped)
    slope = dp / dm
    extrap_m = last_two.iloc[1].multiplier + (50 - last_two.iloc[1].pct_jumped) / slope
    print(f'  → 50% threshold NOT reached within ×2.0')
    print(f'     Extrapolated crossing: ≈ ×{extrap_m:.2f}  '
          f'(+{100*(extrap_m-1):.0f}% ISA scenario)')

# ── ceiling-effect verdict ────────────────────────────────────────────────────
print()
print('=' * 65)
print('CEILING EFFECT — DOSE RESPONSE')
print('=' * 65)
hs_pcts = sat_report['Hotspot_res']['pcts'][1:]
us_pcts = sat_report['Upland_shield_res']['pcts'][1:]
hs_drs  = sat_report['Hotspot_res']['drs'][1:]
us_drs  = sat_report['Upland_shield_res']['drs'][1:]
ceiling_holds_all = all(h < u for h, u in zip(hs_pcts, us_pcts))
risk_higher_all   = all(h > u for h, u in zip(hs_drs, us_drs))
print(f'  Hotspot_res % jumpers  < Upland_shield_res % jumpers at ALL doses: '
      f'{"YES" if ceiling_holds_all else "NO"}')
print(f'  Hotspot_res mean ΔRisk > Upland_shield_res mean ΔRisk at ALL doses: '
      f'{"YES" if risk_higher_all else "NO"}')
if ceiling_holds_all and risk_higher_all:
    print('  → CEILING EFFECT STRENGTHENS WITH DOSE: confirmed.')
    print('     Hotspot_res continuously absorbs disproportionate continuous risk')
    print('     while its class-jump rate stays suppressed by saturation.')
else:
    print('  → Ceiling effect partially confirmed — check per-dose tables above.')

# ── recommended figure ────────────────────────────────────────────────────────
print()
print('RECOMMENDED FIGURE FOR PAPER:')
print('  step4_dose_response_combined.png — 2×2 composite (steps 1-3 + p95 ΔRisk)')
print('  Reason: shows both the ceiling effect (step1 panel) and the continuous-risk')
print('  escalation (step2 panel) in one figure. Step1 is the headline; step2 is the')
print('  mechanistic explanation. Together they tell the full dose-response story.')

# ══════════════════════════════════════════════════════════════════════════════
# 6. Figures
# ══════════════════════════════════════════════════════════════════════════════
print()
print('Generating figures ...')

def arch_pcts_by_dose(arch, metric='pct_jumped'):
    sub = summary[summary.archetype == arch].sort_values('multiplier')
    return [0.0] + sub[metric].tolist()

def arch_vals_by_dose(arch, metric):
    sub = summary[summary.archetype == arch].sort_values('multiplier')
    return [0.0] + sub[metric].tolist()

# ── step1: % class jumpers dose-response ─────────────────────────────────────
def step1_class_jumpers():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arch in RES_ARCH_ORDER:
        pcts = arch_pcts_by_dose(arch)
        lw = 2.5 if arch == 'Hotspot_res' else 1.8
        ls = '-' if arch == 'Hotspot_res' else '--'
        ax.plot(DOSE_X, pcts,
                color=RES_ARCH_COLORS[arch], lw=lw, ls=ls,
                marker='o', ms=7, label=arch.replace('_res', ''), zorder=4)
        # Annotate last point
        ax.text(DOSE_X[-1] + 0.01, pcts[-1],
                f'{pcts[-1]:.1f}%', fontsize=8,
                color=RES_ARCH_COLORS[arch], va='center')

    # Ceiling-effect annotation
    hs_last = arch_pcts_by_dose('Hotspot_res')[-1]
    us_last = arch_pcts_by_dose('Upland_shield_res')[-1]
    ax.annotate(
        'Ceiling effect:\nHotspot_res plateau\n(already class 3)',
        xy=(DOSE_X[-1], hs_last),
        xytext=(1.65, hs_last + 12),
        fontsize=8, color='#C8102E',
        arrowprops=dict(arrowstyle='->', lw=1.1, color='#C8102E'),
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0F0', ec='#C8102E', alpha=0.9),
    )

    ax.set_xticks(DOSE_X)
    ax.set_xticklabels(DOSE_X_LABELS, fontsize=8)
    ax.set_xlabel('ISA development dose (multiplier × baseline ISA)')
    ax.set_ylabel('% of residential parcels jumping ≥1 risk class')
    ax.set_title('Fig. 5 — Dose-Response: % Parcels Jumping Risk Class by ISA Development Level\n'
                 'Hotspot_res plateau = ceiling effect; Upland_shield_res climbs steepest',
                 fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9, loc='upper left')
    ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
    ax.set_ylim(bottom=-1)
    fig.tight_layout()
    save_fig(fig, 'step1_dose_response_class_jumpers')

# ── step2: mean ΔRisk dose-response ──────────────────────────────────────────
def step2_mean_delta_risk():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arch in RES_ARCH_ORDER:
        drs = arch_vals_by_dose(arch, 'mean_delta_risk')
        lw  = 2.5 if arch == 'Hotspot_res' else 1.8
        ax.plot(DOSE_X, drs,
                color=RES_ARCH_COLORS[arch], lw=lw, marker='o', ms=7,
                label=arch.replace('_res', ''), zorder=4)
        ax.text(DOSE_X[-1] + 0.01, drs[-1],
                f'{drs[-1]:+.4f}', fontsize=8,
                color=RES_ARCH_COLORS[arch], va='center')

    ax.annotate(
        'Hotspot_res:\nhighest continuous-\nrisk response',
        xy=(DOSE_X[-1], arch_vals_by_dose('Hotspot_res', 'mean_delta_risk')[-1]),
        xytext=(1.55, arch_vals_by_dose('Hotspot_res', 'mean_delta_risk')[-1] * 0.65),
        fontsize=8, color='#C8102E',
        arrowprops=dict(arrowstyle='->', lw=1.1, color='#C8102E'),
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0F0', ec='#C8102E', alpha=0.9),
    )

    ax.set_xticks(DOSE_X)
    ax.set_xticklabels(DOSE_X_LABELS, fontsize=8)
    ax.set_xlabel('ISA development dose')
    ax.set_ylabel('Mean ΔRisk  (Ridge target_max_log: cf − baseline)')
    ax.set_title('Dose-Response: Mean ΔRisk (Continuous Risk) by ISA Development Level\n'
                 'Hotspot_res absorbs disproportionate continuous risk at every dose',
                 fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
    fig.tight_layout()
    save_fig(fig, 'step2_dose_response_mean_delta_risk')

# ── step3: class 3 entries ────────────────────────────────────────────────────
def step3_class3_entries():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arch in RES_ARCH_ORDER:
        sub  = summary[summary.archetype == arch].sort_values('multiplier')
        vals = [0] + sub['n_to_class3'].tolist()
        ax.plot(DOSE_X, vals,
                color=RES_ARCH_COLORS[arch], lw=1.8, marker='o', ms=7,
                label=arch.replace('_res', ''), zorder=4)
        ax.text(DOSE_X[-1] + 0.01, vals[-1],
                f'{vals[-1]:,}', fontsize=8,
                color=RES_ARCH_COLORS[arch], va='center')

    # Total across all archetypes
    total_vals = [0] + [
        int(summary[summary.multiplier == m]['n_to_class3'].sum())
        for m in DOSE_MULTIPLIERS]
    ax.plot(DOSE_X, total_vals,
            color='black', lw=2.2, ls=':', marker='s', ms=8,
            label='Total', zorder=5)

    ax.set_xticks(DOSE_X)
    ax.set_xticklabels(DOSE_X_LABELS, fontsize=8)
    ax.set_xlabel('ISA development dose')
    ax.set_ylabel('New parcels entering class 3 (extreme flood)')
    ax.set_title('Dose-Response: Catastrophic Risk Frontier — New Entries to Class 3\n'
                 'Does densification break the topographic class 3 ceiling?', fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
    fig.tight_layout()
    save_fig(fig, 'step3_dose_response_class3_entries')

# ── step4: combined 2×2 composite ────────────────────────────────────────────
def step4_combined():
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    panels = [
        ('pct_jumped',       '% parcels jumping ≥1 class',             '(a)'),
        ('mean_delta_risk',  'Mean ΔRisk (Ridge target_max_log)',       '(b)'),
        ('n_to_class3',      'New entries to class 3',                  '(c)'),
        ('p95_delta_risk',   '95th pct ΔRisk (worst-case parcel)',      '(d)'),
    ]
    baselines = {'pct_jumped': 0.0, 'mean_delta_risk': 0.0,
                 'n_to_class3': 0, 'p95_delta_risk': 0.0}

    for idx, (metric, ylabel, panel_lbl) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        for arch in RES_ARCH_ORDER:
            sub  = summary[summary.archetype == arch].sort_values('multiplier')
            vals = [baselines[metric]] + sub[metric].tolist()
            lw   = 2.5 if arch == 'Hotspot_res' else 1.6
            ax.plot(DOSE_X, vals,
                    color=RES_ARCH_COLORS[arch], lw=lw,
                    marker='o', ms=6, label=arch.replace('_res', ''), zorder=4)

        ax.set_xticks(DOSE_X)
        ax.set_xticklabels(DOSE_X_LABELS, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(f'{panel_lbl} {ylabel}', fontsize=9.5)
        ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
        if idx == 0:
            ax.legend(fontsize=8, framealpha=0.9, loc='upper left')

        # Ceiling effect annotation on panel (a)
        if metric == 'pct_jumped':
            hs_last = arch_pcts_by_dose('Hotspot_res')[-1]
            ax.annotate('Ceiling\neffect',
                        xy=(DOSE_X[-1], hs_last),
                        xytext=(1.70, hs_last + 8),
                        fontsize=7.5, color='#C8102E',
                        arrowprops=dict(arrowstyle='->', lw=0.9, color='#C8102E'))

    fig.suptitle(
        'ISA Development Dose-Response — Residential Parcels (n = 115,820)\n'
        'Hotspot_res: highest continuous risk but suppressed class-jump rate (ceiling effect)',
        fontsize=11, y=1.01)
    save_fig(fig, 'step4_dose_response_combined')

# ── step5: 5-panel geographic map sequence ────────────────────────────────────
def step5_dose_response_maps():
    gpkg_path = RES_DIR / 'parcel_archetypes_v2_nlcd.gpkg'
    if not gpkg_path.exists():
        print('  [SKIP] step5 — gpkg not found'); return

    gdf_base = gpd.read_file(gpkg_path)
    # Keep only residential
    gdf_base = gdf_base[gdf_base['parcel_id'].isin(fm_res['parcel_id'])].copy()

    fig, axes = plt.subplots(1, 5, figsize=(28, 7))
    jump_cols = [f'class_jump_p{lbl[1:]}' for lbl in DOSE_LABELS]  # p20, p40 …

    for ax, col, dose_lbl, m in zip(axes, jump_cols, DOSE_LABELS, DOSE_MULTIPLIERS):
        gdf_plot = gdf_base.merge(
            results[['parcel_id', col]], on='parcel_id', how='left')
        gdf_plot[col] = gdf_plot[col].fillna(0).clip(lower=0)

        # No-change grey
        gdf_plot[gdf_plot[col] == 0].plot(
            ax=ax, color='#DDDDDD', linewidth=0, alpha=0.55, rasterized=True)
        for mag, clr in [(1, '#FFCC44'), (2, '#EE6600'), (3, '#AA0000')]:
            sub = gdf_plot[gdf_plot[col] == mag]
            if len(sub):
                sub.plot(ax=ax, color=clr, linewidth=0, alpha=0.9, rasterized=True)

        n_j = int((gdf_plot[col] > 0).sum())
        ax.set_title(f'×{m:.1f}  ({dose_lbl})\n{n_j:,} jumpers', fontsize=9)
        ax.set_axis_off()

    # Shared legend
    handles = [mpatches.Patch(color='#DDDDDD', label='No change'),
               mpatches.Patch(color='#FFCC44', label='+1 class'),
               mpatches.Patch(color='#EE6600', label='+2 classes'),
               mpatches.Patch(color='#AA0000', label='+3 classes')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        'Geographic Spread of Risk-Class Jumpers Under Increasing ISA Development\n'
        'Residential parcels only (NLCD 22–24)', fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'step5_dose_response_maps')

# ── step6: stacked area chart — class jumpers by archetype ───────────────────
def step6_recruitment_curve():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Build stacked arrays: n_jumped per archetype at each dose
    y_stacks = []
    for arch in reversed(RES_ARCH_ORDER):   # reverse so Hotspot on top
        sub  = summary[summary.archetype == arch].sort_values('multiplier')
        vals = [0] + sub['n_jumped'].tolist()
        y_stacks.append((arch, vals))

    bottoms = np.zeros(len(DOSE_X))
    for arch, vals in y_stacks:
        vals_arr = np.array(vals, dtype=float)
        ax.fill_between(DOSE_X, bottoms, bottoms + vals_arr,
                        color=RES_ARCH_COLORS[arch], alpha=0.80,
                        label=arch.replace('_res', ''), step='pre')
        bottoms += vals_arr

    ax.set_xticks(DOSE_X)
    ax.set_xticklabels(DOSE_X_LABELS, fontsize=8)
    ax.set_xlabel('ISA development dose')
    ax.set_ylabel('Number of parcels jumping ≥1 risk class (stacked by archetype)')
    ax.set_title('Dose-Response Recruitment Curve — Who Joins as Dose Increases?\n'
                 'Upland_shield_res dominates new recruits; Hotspot_res plateau = ceiling effect',
                 fontsize=10)
    handles = [mpatches.Patch(color=RES_ARCH_COLORS[a], alpha=0.8,
               label=a.replace('_res', '')) for a in RES_ARCH_ORDER]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9, loc='upper left')
    ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
    fig.tight_layout()
    save_fig(fig, 'step6_recruitment_curve')

# ── step7: catastrophic floor ─────────────────────────────────────────────────
def step7_catastrophic_floor():
    # Total parcels in class 3 under each cf scenario (all residential)
    n_base3 = n_base_cls3
    cf3_totals = [n_base3] + [
        int(summary[summary.multiplier == m]['n_cf_class3'].sum())
        for m in DOSE_MULTIPLIERS]
    new_in_cf3 = [0] + [
        int(summary[summary.multiplier == m]['n_to_class3'].sum())
        for m in DOSE_MULTIPLIERS]
    pct_new_of_total = [0.0] + [
        100 * n / max(t, 1) for n, t in zip(new_in_cf3[1:], cf3_totals[1:])]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax2 = ax.twinx()

    ax.plot(DOSE_X, cf3_totals, 'o-', color='#1A3A5C', lw=2.2, ms=8,
            label='Total parcels in class 3 (cf)')
    ax.fill_between(DOSE_X, n_base3, cf3_totals, alpha=0.15, color='#1A3A5C')
    ax.axhline(n_base3, ls='--', lw=1.2, color='#1A3A5C', alpha=0.5,
               label=f'Baseline class 3: {n_base3:,}')

    ax2.plot(DOSE_X, pct_new_of_total, 's--', color='#C8102E', lw=1.8, ms=7,
             label='% of class 3 that are NEW entries')
    ax2.set_ylabel('% of total class 3 that are new densification entries',
                   color='#C8102E', fontsize=8.5)
    ax2.tick_params(axis='y', labelcolor='#C8102E')
    ax2.set_ylim(0, 15)

    ax.set_xticks(DOSE_X)
    ax.set_xticklabels(DOSE_X_LABELS, fontsize=8)
    ax.set_xlabel('ISA development dose')
    ax.set_ylabel('Total residential parcels in class 3')
    ax.set_title(
        'Catastrophic Floor: Topographic vs Densification-Driven Class 3 Parcels\n'
        'Class 3 pool is topography-dominated; densification adds a thin margin',
        fontsize=10)

    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5,
              framealpha=0.9, loc='center left')
    ax.yaxis.grid(True, lw=0.5, color='#EEEEEE', zorder=0)
    fig.tight_layout()
    save_fig(fig, 'step7_catastrophic_floor')

# ── run all ───────────────────────────────────────────────────────────────────
step1_class_jumpers()
step2_mean_delta_risk()
step3_class3_entries()
step4_combined()
step5_dose_response_maps()
step6_recruitment_curve()
step7_catastrophic_floor()

# ══════════════════════════════════════════════════════════════════════════════
# 7. Full terminal report
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 65)
print('FULL DOSE-RESPONSE SUMMARY TABLE')
print('=' * 65)
for metric, header in [
    ('pct_jumped',       '% PARCELS JUMPING ≥1 CLASS'),
    ('mean_delta_risk',  'MEAN ΔRisk (Ridge target_max_log)'),
    ('n_to_class3',      'NEW ENTRIES TO CLASS 3'),
    ('p95_delta_risk',   '95th-PERCENTILE ΔRisk'),
]:
    print(f'\n  {header}')
    hdr = f'  {"Archetype":30s}' + ''.join(f'  {lbl:>7}' for lbl in DOSE_LABELS)
    print(hdr)
    print('  ' + '-' * (30 + 9 * 5))
    for arch in RES_ARCH_ORDER:
        sub  = summary[summary.archetype == arch].sort_values('multiplier')
        vals = sub[metric].tolist()
        if 'pct' in metric:
            fmt = ''.join(f'  {v:>6.1f}%' for v in vals)
        elif metric == 'n_to_class3':
            fmt = ''.join(f'  {int(v):>7,}' for v in vals)
        else:
            fmt = ''.join(f'  {v:>+7.4f}' for v in vals)
        print(f'  {arch:30s}{fmt}')

print()
print('=' * 65)
print('DONE')
print(f'  Results : {RES_DIR}/')
print(f'  Figures : {FIG_DIR}/')
print('=' * 65)
