#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flood-risk Modeling v2 - 12 features (+ HAND_min)
==================================================
Features (12): elevation, slope, TWI, log_flow_accum, dist_to_stream,
               dist_to_street, ISA_frac, log_lot_area, is_enclave,
               conn_topo, Cw_topo, HAND_min
Dropped: HAND_90m, HAND (10m mean)
16 experiments: 4 models x 2 targets x 2 CV schemes
"""

import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE / 'data' / 'processed'
RESULTS_DIR = BASE / 'outputs' / 'results'
FIG_DIR     = BASE / 'outputs' / 'figures' / 'modeling_v2'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
TOP_K_PCT     = 10
MIN_FOLD_SIZE = 100
LGB_ROUNDS    = 500

FEATURES = [
    'elevation', 'slope', 'TWI', 'log_flow_accum',
    'dist_to_stream', 'dist_to_street',
    'ISA_frac', 'log_lot_area', 'is_enclave',
    'conn_topo', 'Cw_topo', 'HAND_min',
]

MODELS  = ['median', 'cw_only', 'ridge', 'lightgbm']
TARGETS = ['target_mean_log', 'target_max_log']
FOLDS   = ['fold_subbasin', 'fold_block']

# 11-feature baseline Spearman from existing cv_summary.csv
BASELINE = {
    ('median',   'target_mean_log', 'fold_subbasin'): np.nan,
    ('cw_only',  'target_mean_log', 'fold_subbasin'): 0.162027,
    ('ridge',    'target_mean_log', 'fold_subbasin'): 0.344515,
    ('lightgbm', 'target_mean_log', 'fold_subbasin'): np.nan,
    ('median',   'target_mean_log', 'fold_block'):    np.nan,
    ('cw_only',  'target_mean_log', 'fold_block'):    0.134782,
    ('ridge',    'target_mean_log', 'fold_block'):    0.344743,
    ('lightgbm', 'target_mean_log', 'fold_block'):    np.nan,
    ('median',   'target_max_log',  'fold_subbasin'): np.nan,
    ('cw_only',  'target_max_log',  'fold_subbasin'): 0.165149,
    ('ridge',    'target_max_log',  'fold_subbasin'): 0.401490,
    ('lightgbm', 'target_max_log',  'fold_subbasin'): np.nan,
    ('median',   'target_max_log',  'fold_block'):    np.nan,
    ('cw_only',  'target_max_log',  'fold_block'):    0.136907,
    ('ridge',    'target_max_log',  'fold_block'):    0.397559,
    ('lightgbm', 'target_max_log',  'fold_block'):    np.nan,
}

LGB_PARAMS = {
    'objective':       'lambdarank',
    'metric':          'ndcg',
    'ndcg_at':         [10, 50, 100],
    'learning_rate':   0.05,
    'num_leaves':      63,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq':    5,
    'verbose':         -1,
    'random_state':    RANDOM_STATE,
    'max_position':    1000,
}

FOLD_COLORS   = {'fold_subbasin': '#3B4992', 'fold_block': '#EE9900'}
FOLD_LABELS   = {'fold_subbasin': 'Sub-basin CV', 'fold_block': '2 km Block CV'}
TARGET_LABELS = {'target_mean_log': 'Mean log depth', 'target_max_log': 'Max log depth'}
MODEL_LABELS  = {'median': 'Median', 'cw_only': 'Cw_topo only',
                 'ridge': 'Ridge', 'lightgbm': 'LightGBM-LR'}

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                     'axes.titlesize': 12, 'axes.labelsize': 11})

KEY_RIDGE_MAX_SUB = 'ridge__target_max_log__fold_subbasin'

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print('Flood-risk Modeling v2  (12 features, +HAND_min)')
print('=' * 65)

df = pd.read_csv(DATA_DIR / 'feature_matrix.csv')
FEATURES = [f for f in FEATURES if f in df.columns]

print(f'Parcels  : {len(df):,}')
print(f'Features : ({len(FEATURES)}) {", ".join(FEATURES)}')
print(f'Sub-basin folds: {df["fold_subbasin"].nunique()}  |  '
      f'Block folds: {df["fold_block"].nunique()}')

# K-means risk classes on raw flood depth (FloodGenome approach)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_km_X  = StandardScaler().fit_transform(df[['target_mean', 'target_max']].values)
_km    = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
df['risk_class'] = _km.fit_predict(_km_X)
# Re-label so 0 = lowest flood depth, 3 = highest
_cmeans = df.groupby('risk_class')['target_max'].mean().sort_values()
_lmap   = {old: new for new, old in enumerate(_cmeans.index)}
df['risk_class'] = df['risk_class'].map(_lmap)
_vc = df['risk_class'].value_counts().sort_index()
print(f'K-means risk classes (0=low .. 3=high): {_vc.to_dict()}')


# ══════════════════════════════════════════════════════════════════════════════
# 2. Evaluation helpers  (identical to baseline notebook)
# ══════════════════════════════════════════════════════════════════════════════
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, ndcg_score


def spearman_r(y_true, y_pred):
    return spearmanr(y_true, y_pred).statistic


def ndcg_at_top_k(y_true, y_pred, k_pct=TOP_K_PCT):
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    y_rel = np.asarray(y_true, dtype=float)
    if y_rel.min() < 0:          # shift so ndcg_score gets non-negative relevance
        y_rel = y_rel - y_rel.min()
    return float(ndcg_score([y_rel], [np.asarray(y_pred, dtype=float)], k=k))


def scale_to_rank_labels(y, n_bins=32):
    ranks = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates='drop')
    return ranks.fillna(0).astype(int).values


def weighted_avg(df_, col):
    if len(df_) == 0:
        return np.nan
    vals = df_[col].values.astype(float)
    wts  = df_['n_test'].values.astype(float)
    mask = ~np.isnan(vals)
    return np.average(vals[mask], weights=wts[mask]) if mask.any() else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 3. Spatial-CV runner  (same logic as baseline notebook)
# ══════════════════════════════════════════════════════════════════════════════
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def run_spatial_cv(df, features, target_col, fold_col, model_type,
                   collect_preds=False):
    """Leave-one-fold-out CV.  Returns (metrics_df, pred_list)."""
    fold_sizes  = df[fold_col].value_counts()
    valid_folds = fold_sizes[fold_sizes >= MIN_FOLD_SIZE].index.tolist()

    rows  = []
    preds = []

    for fold_id in valid_folds:
        test_mask  = df[fold_col] == fold_id
        train_mask = ~test_mask

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, target_col].values
        X_te = df.loc[test_mask,  features].values
        y_te = df.loc[test_mask,  target_col].values

        if len(np.unique(y_te)) < 2:
            continue

        if model_type == 'median':
            y_pred = np.full(len(y_te), np.median(y_tr))

        elif model_type == 'cw_only':
            if 'Cw_topo' not in list(features):
                continue
            idx    = list(features).index('Cw_topo')
            y_pred = X_te[:, idx]

        elif model_type == 'ridge':
            sc     = StandardScaler()
            Xtr_s  = sc.fit_transform(X_tr)
            Xte_s  = sc.transform(X_te)
            m      = Ridge(alpha=1.0, random_state=RANDOM_STATE)
            m.fit(Xtr_s, y_tr)
            y_pred = m.predict(Xte_s)

        elif model_type == 'lightgbm':
            y_tr_r = scale_to_rank_labels(y_tr)
            # LightGBM 4.x caps query size at 10,000; chunk into groups of 9,000
            chunk  = 9000
            n_tr   = len(y_tr)
            groups = [chunk] * (n_tr // chunk)
            if n_tr % chunk:
                groups.append(n_tr % chunk)
            dtrain = lgb.Dataset(X_tr, label=y_tr_r, group=groups,
                                 feature_name=list(features))
            m      = lgb.train(LGB_PARAMS, dtrain, num_boost_round=LGB_ROUNDS)
            y_pred = m.predict(X_te)

        else:
            raise ValueError(f'Unknown model: {model_type!r}')

        n_te = int(test_mask.sum())
        rows.append({
            'fold':       fold_id,
            'n_test':     n_te,
            'spearman':   spearman_r(y_te, y_pred),
            'ndcg_top_k': ndcg_at_top_k(y_te, y_pred),
            'mae_log':    mean_absolute_error(y_te, y_pred)
                          if model_type in ('ridge', 'cw_only') else np.nan,
        })
        if collect_preds:
            preds.append((y_te.copy(), y_pred.copy(), fold_id, n_te))

    return pd.DataFrame(rows), preds


# ══════════════════════════════════════════════════════════════════════════════
# 3b. RF classifier CV runner  (FloodGenome approach)
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix as sk_cm, roc_curve


def run_rf_cv(df, features, label_col, fold_col, collect_preds=False):
    """Leave-one-fold-out CV for RF risk classifier.
    Returns (metrics_df, agg_conf_matrix, y_true_list, y_proba_list).
    """
    fold_sizes  = df[fold_col].value_counts()
    valid_folds = fold_sizes[fold_sizes >= MIN_FOLD_SIZE].index.tolist()
    N_CLS       = 4

    rows         = []
    conf_matrix  = np.zeros((N_CLS, N_CLS), dtype=int)
    y_true_list  = []
    y_proba_list = []

    for fold_id in valid_folds:
        test_mask  = df[fold_col] == fold_id
        train_mask = ~test_mask

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, label_col].values.astype(int)
        X_te = df.loc[test_mask,  features].values
        y_te = df.loc[test_mask,  label_col].values.astype(int)

        if len(np.unique(y_te)) < 2:
            continue

        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(X_tr)
        Xte_s = sc.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(Xtr_s, y_tr)

        # Always return (n_te, 4) proba — handles folds missing a class
        raw_proba    = rf.predict_proba(Xte_s)
        y_proba_full = np.zeros((len(y_te), N_CLS))
        for i, c in enumerate(rf.classes_):
            y_proba_full[:, int(c)] = raw_proba[:, i]

        y_pred = rf.predict(Xte_s)

        try:
            auc_per = roc_auc_score(
                y_te, y_proba_full, multi_class='ovr', average=None,
                labels=list(range(N_CLS)),
            )
        except ValueError:
            auc_per = [np.nan] * N_CLS

        conf_matrix += sk_cm(y_te, y_pred, labels=list(range(N_CLS)))
        n_te = int(test_mask.sum())
        row  = {'fold': fold_id, 'n_test': n_te,
                'auc_macro': float(np.nanmean(auc_per))}
        for c in range(N_CLS):
            row[f'auc_class{c}'] = float(auc_per[c])
        rows.append(row)

        if collect_preds:
            y_true_list.append(y_te.copy())
            y_proba_list.append(y_proba_full.copy())

    return pd.DataFrame(rows), conf_matrix, y_true_list, y_proba_list


# ══════════════════════════════════════════════════════════════════════════════
# 4. Run all 16 experiments
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 65)
print('Running 16 experiments  (4 models × 2 targets × 2 CV schemes)')
print('─' * 65)

summary_rows   = []
per_fold_store = {}
pred_store     = {}
t0             = time.time()

for target in TARGETS:
    for fold in FOLDS:
        for model in MODELS:
            key     = f'{model}__{target}__{fold}'
            collect = (key == KEY_RIDGE_MAX_SUB)
            t1      = time.time()
            print(f'  ▶ {key}', end=' ... ', flush=True)

            metrics, preds = run_spatial_cv(
                df, FEATURES, target, fold, model,
                collect_preds=collect,
            )
            per_fold_store[key] = metrics
            if collect:
                pred_store[key] = preds

            wt_sp = weighted_avg(metrics, 'spearman')
            wt_nd = weighted_avg(metrics, 'ndcg_top_k')
            summary_rows.append({
                'model':         model,
                'target':        target,
                'fold_scheme':   fold,
                'n_folds':       len(metrics),
                'spearman_mean': wt_sp,
                'spearman_std':  metrics['spearman'].std() if len(metrics) else np.nan,
                'ndcg_top_k':    wt_nd,
            })
            bl        = BASELINE.get((model, target, fold), np.nan)
            delta_str = (f'  Δ={wt_sp - bl:+.3f}' if (not np.isnan(bl) and
                         not np.isnan(wt_sp)) else '')
            print(f'Spearman={wt_sp:+.3f}  NDCG@10%={wt_nd:.3f}  '
                  f'({time.time() - t1:.0f}s){delta_str}')

summary = pd.DataFrame(summary_rows)
out_csv = RESULTS_DIR / 'cv_summary_v2.csv'
summary.to_csv(out_csv, index=False)
print(f'\nSaved: {out_csv}')
print(f'Total CV runtime: {time.time() - t0:.0f}s')

# Per-fold Ridge metrics — required by generate_cv_diagnostic_figures.py and compose_supp_s1.py
TAB_DIR = BASE / 'outputs' / 'tables'
TAB_DIR.mkdir(parents=True, exist_ok=True)
pf_rows = []
for _key, _mdf in per_fold_store.items():
    _model, _target, _fold_scheme = _key.split('__')
    if _model == 'ridge' and _target == 'target_max_log':
        _tmp = _mdf[['fold', 'n_test', 'spearman', 'ndcg_top_k']].copy()
        _tmp.insert(0, 'fold_scheme', _fold_scheme)
        _tmp.rename(columns={'fold': 'fold_id', 'ndcg_top_k': 'ndcg_top10pct'}, inplace=True)
        pf_rows.append(_tmp)
pf_df = pd.concat(pf_rows, ignore_index=True)
pf_out = TAB_DIR / 'per_fold_metrics_ridge.csv'
pf_df.to_csv(pf_out, index=False)
print(f'Saved: {pf_out}  ({len(pf_df)} rows)')

# ── RF classifier (2 CV schemes) ──────────────────────────────────────────────
print('\n' + '─' * 65)
print('Running RF classifier (FloodGenome)  (2 CV schemes)')
print('─' * 65)

rf_store       = {}
rf_summary_rows = []

for fold in FOLDS:
    key = f'rf_classifier__risk_class__{fold}'
    print(f'  ▶ {key}', end=' ... ', flush=True)
    t1 = time.time()
    rf_metrics, cm, y_true_list, y_proba_list = run_rf_cv(
        df, FEATURES, 'risk_class', fold, collect_preds=True,
    )
    rf_store[fold] = (y_true_list, y_proba_list, cm)

    wt_auc  = weighted_avg(rf_metrics, 'auc_macro')
    per_cls = [weighted_avg(rf_metrics, f'auc_class{c}') for c in range(4)]
    rf_summary_rows.append({
        'model': 'rf_classifier', 'target': 'risk_class_kmeans',
        'fold_scheme': fold, 'n_folds': len(rf_metrics),
        'spearman_mean': np.nan, 'spearman_std': np.nan, 'ndcg_top_k': np.nan,
        'auc_macro':  wt_auc,
        'auc_class0': per_cls[0], 'auc_class1': per_cls[1],
        'auc_class2': per_cls[2], 'auc_class3': per_cls[3],
    })
    print(f'AUC-macro={wt_auc:.3f}  '
          f'per-class=[{per_cls[0]:.3f}, {per_cls[1]:.3f}, '
          f'{per_cls[2]:.3f}, {per_cls[3]:.3f}]  ({time.time()-t1:.0f}s)')

    print('  Confusion matrix (rows=true, cols=pred):')
    print(cm)

# Append RF rows to summary and re-save
rf_summary = pd.DataFrame(rf_summary_rows)
summary_full = pd.concat([summary, rf_summary], ignore_index=True)
summary_full.to_csv(out_csv, index=False)
print(f'Updated: {out_csv}  ({len(summary_full)} rows total)')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Terminal summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('SPEARMAN RESULTS + ΔSpearman vs 11-feature baseline')
print('=' * 65)

for model_name in ['ridge', 'lightgbm']:
    print(f'\n{model_name.upper()}:')
    for target in TARGETS:
        for fold in FOLDS:
            row = summary.query(
                'model == @model_name and target == @target and fold_scheme == @fold'
            )
            if row.empty:
                continue
            sp = row.iloc[0]['spearman_mean']
            bl = BASELINE.get((model_name, target, fold), np.nan)
            if not np.isnan(bl) and not np.isnan(sp):
                d_str = f'   Δ={sp - bl:+.3f} vs 11-feat'
            elif np.isnan(bl):
                d_str = '   (no baseline)'
            else:
                d_str = ''
            print(f'  {target:20s} / {fold:15s}:  {sp:+.4f}{d_str}')

print('\n  ► KEY: Ridge target_max_log/fold_subbasin baseline = 0.401')

print('\nRF CLASSIFIER (FloodGenome — k-means k=4 on [target_mean, target_max]):')
for r in rf_summary_rows:
    fold = r['fold_scheme']
    print(f'  {fold:15s}: AUC-macro={r["auc_macro"]:.3f}  '
          f'per-class=[{r["auc_class0"]:.3f}, {r["auc_class1"]:.3f}, '
          f'{r["auc_class2"]:.3f}, {r["auc_class3"]:.3f}]')


# ══════════════════════════════════════════════════════════════════════════════
# 6. Final models on full dataset  (target_max_log, for SHAP + map)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 65)
print('Training final models on full dataset (target_max_log) …')

X_all     = df[FEATURES].values
y_all_max = df['target_max_log'].values

sc_all      = StandardScaler()
X_all_sc    = sc_all.fit_transform(X_all)
ridge_final = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge_final.fit(X_all_sc, y_all_max)
ridge_pred_all = ridge_final.predict(X_all_sc)
print(f'  Ridge intercept: {ridge_final.intercept_:.4f}')

y_max_rank  = scale_to_rank_labels(y_all_max)
chunk_f     = 9000
n_all       = len(y_all_max)
groups_full = [chunk_f] * (n_all // chunk_f) + ([n_all % chunk_f] if n_all % chunk_f else [])
dtrain_full = lgb.Dataset(X_all, label=y_max_rank, group=groups_full,
                          feature_name=FEATURES)
lgb_final   = lgb.train(LGB_PARAMS, dtrain_full, num_boost_round=LGB_ROUNDS)
print(f'  LightGBM: {LGB_ROUNDS} rounds')

# SHAP sample
import shap
sample_n   = min(5000, len(df))
rng        = np.random.RandomState(RANDOM_STATE)
s_idx      = rng.choice(len(df), sample_n, replace=False)
X_samp     = X_all[s_idx]
X_samp_sc  = X_all_sc[s_idx]

print('  Ridge SHAP (LinearExplainer) …', end=' ', flush=True)
ridge_expl = shap.LinearExplainer(ridge_final, X_all_sc,
                                   feature_perturbation='interventional')
shap_ridge = ridge_expl.shap_values(X_samp_sc)
print('done')

print('  LGB SHAP (TreeExplainer) …', end=' ', flush=True)
lgb_expl   = shap.TreeExplainer(lgb_final)
shap_lgb_r = lgb_expl.shap_values(X_samp)
if isinstance(shap_lgb_r, list):
    shap_lgb = shap_lgb_r[0]
elif np.ndim(shap_lgb_r) == 3:
    shap_lgb = shap_lgb_r[:, :, 0]
else:
    shap_lgb = shap_lgb_r
print('done')


# ══════════════════════════════════════════════════════════════════════════════
# 7. Figures
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 65)
print('Generating figures …')


# ── step1: CV performance ─────────────────────────────────────────────────────
def step1_cv_performance():
    # x-axis: model × target  (8 groups); bars: subbasin vs block
    scenario_keys   = [(m, t) for m in MODELS for t in TARGETS]
    scenario_labels = [f'{MODEL_LABELS[m]}\n{TARGET_LABELS[t]}'
                       for m, t in scenario_keys]
    x   = np.arange(len(scenario_keys))
    w   = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for ax, metric, ylabel, title_suffix in zip(
        axes,
        ['spearman_mean', 'ndcg_top_k'],
        ['Spearman ρ (weighted by fold size)',
         f'NDCG@{TOP_K_PCT}% (weighted by fold size)'],
        ['Spearman ρ', f'NDCG@{TOP_K_PCT}%'],
    ):
        for offset, (fold, label) in enumerate(FOLD_LABELS.items()):
            vals, errs = [], []
            for m, t in scenario_keys:
                row = summary.query(
                    'model == @m and target == @t and fold_scheme == @fold'
                )
                v = row.iloc[0][metric] if not row.empty else np.nan
                e = (row.iloc[0]['spearman_std']
                     if metric == 'spearman_mean' and not row.empty else 0.0)
                vals.append(v)
                errs.append(0.0 if np.isnan(e) else e)
            ax.bar(x + (offset - 0.5) * w, vals, w,
                   label=label, color=FOLD_COLORS[fold],
                   alpha=0.85, edgecolor='white',
                   yerr=errs if metric == 'spearman_mean' else None,
                   capsize=3, error_kw={'linewidth': 1.0})

        ax.set_ylabel(ylabel)
        ax.axhline(0, color='black', linewidth=0.7, zorder=0)
        ax.legend(framealpha=0.9, loc='upper left')
        # Light background bands to separate models (every 2 scenarios)
        for mi in range(len(MODELS)):
            if mi % 2 == 0:
                ax.axvspan(mi * 2 - 0.5, mi * 2 + 1.5, alpha=0.06,
                           color='steelblue', zorder=0)

    axes[0].set_title('Step 1 — CV Performance: All 16 Experiments\n'
                      '(4 models × 2 targets × 2 CV schemes)',
                      fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenario_labels, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step1_cv_performance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step1_cv_performance.png')


# ── step2: baseline comparison ────────────────────────────────────────────────
def step2_baseline_comparison():
    rows = []
    for model in ['cw_only', 'ridge']:          # models with baseline values
        for target in TARGETS:
            for fold in FOLDS:
                sub = summary.query(
                    'model == @model and target == @target and fold_scheme == @fold'
                )
                v2_sp = sub.iloc[0]['spearman_mean'] if not sub.empty else np.nan
                bl_sp = BASELINE.get((model, target, fold), np.nan)
                if np.isnan(v2_sp) or np.isnan(bl_sp):
                    continue
                rows.append({
                    'label': (f'{MODEL_LABELS[model]}\n'
                              f'{TARGET_LABELS[target]}\n'
                              f'{FOLD_LABELS[fold]}'),
                    'v1':    bl_sp,
                    'v2':    v2_sp,
                    'delta': v2_sp - bl_sp,
                })
    cdf = pd.DataFrame(rows)
    if cdf.empty:
        print('  ✗ step2: no paired baseline data')
        return

    n, w = len(cdf), 0.38
    x    = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 6))
    ax.bar(x - w / 2, cdf['v1'], w, label='11-feat baseline',
           color='#aaaaaa', edgecolor='white', alpha=0.85)
    ax.bar(x + w / 2, cdf['v2'], w, label='12-feat (+ HAND_min)',
           color='#3B7DC8', edgecolor='white', alpha=0.85)

    for i, row in cdf.iterrows():
        clr = '#228833' if row['delta'] >= 0 else '#CC3311'
        top = max(row['v1'], row['v2']) + 0.009
        ax.text(i, top, f"Δ={row['delta']:+.3f}",
                ha='center', va='bottom', fontsize=8.5,
                color=clr, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cdf['label'], fontsize=8.5)
    ax.set_ylabel('Spearman ρ (weighted by fold size)')
    ax.set_title('Step 2 — 11-feature Baseline vs 12-feature (+ HAND_min)',
                 fontsize=13)
    ax.legend(framealpha=0.9)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step2_baseline_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step2_baseline_comparison.png')


# ── step3: per-fold Spearman scatter ─────────────────────────────────────────
def step3_per_fold_spearman():
    mdf = per_fold_store.get(KEY_RIDGE_MAX_SUB, pd.DataFrame())
    if mdf.empty:
        print('  ✗ step3: missing Ridge/max/subbasin per-fold data')
        return

    wt_mean = weighted_avg(mdf, 'spearman')
    smax    = mdf['n_test'].max()

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        mdf['n_test'], mdf['spearman'],
        s=mdf['n_test'] / smax * 300 + 30,
        c=mdf['spearman'], cmap='RdYlGn', vmin=-0.1, vmax=0.8,
        alpha=0.85, edgecolors='white', linewidth=0.6, zorder=3,
    )
    ax.axhline(wt_mean, color='#3B4992', linestyle='--', linewidth=1.6,
               label=f'Weighted mean ρ = {wt_mean:.3f}', zorder=2)

    for _, row in mdf.iterrows():
        ax.annotate(str(row['fold']),
                    (row['n_test'], row['spearman']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=8, alpha=0.75)

    plt.colorbar(sc, ax=ax, label='Spearman ρ', shrink=0.85)
    ax.set_xlabel('Fold size (n_test parcels)')
    ax.set_ylabel('Spearman ρ (per fold)')
    ax.set_title('Step 3 — Per-fold Spearman: Ridge · target_max_log · Sub-basin CV\n'
                 'Dot size ∝ fold size', fontsize=12)
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step3_per_fold_spearman.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step3_per_fold_spearman.png')


# ── step4: Ridge coefficients ─────────────────────────────────────────────────
def step4_ridge_coefficients():
    coefs        = pd.Series(ridge_final.coef_, index=FEATURES)
    coefs_sorted = coefs.reindex(coefs.abs().sort_values().index)
    colors       = ['#EE4444' if f == 'HAND_min' else '#4477AA'
                    for f in coefs_sorted.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(coefs_sorted)), coefs_sorted.values,
            color=colors, edgecolor='white', alpha=0.88)
    ax.set_yticks(range(len(coefs_sorted)))
    ax.set_yticklabels(coefs_sorted.index, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.9)

    # Annotate HAND_min value inline
    idx_hand = coefs_sorted.index.tolist().index('HAND_min')
    val_hand = coefs['HAND_min']
    offset   = 0.012 if val_hand >= 0 else -0.012
    ax.text(val_hand + offset, idx_hand,
            f'  {val_hand:.3f}  ← HAND_min (new)',
            va='center', ha='left' if val_hand >= 0 else 'right',
            color='#EE4444', fontsize=9.5, fontweight='bold')

    patches = [
        mpatches.Patch(color='#EE4444', label='HAND_min  (new feature)'),
        mpatches.Patch(color='#4477AA', label='Original features'),
    ]
    ax.legend(handles=patches, framealpha=0.9, loc='lower right')
    ax.set_xlabel('Standardised coefficient (Ridge, α=1.0)')
    ax.set_title('Step 4 — Ridge Regression Coefficients\n'
                 'target_max_log · full dataset · sorted by |coef|', fontsize=12)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step4_ridge_coefficients.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step4_ridge_coefficients.png')


# ── step5: SHAP beeswarm Ridge ───────────────────────────────────────────────
def step5_shap_ridge():
    import shap as _shap
    _shap.summary_plot(
        shap_ridge, X_samp, feature_names=FEATURES,
        plot_type='dot', show=False,
        plot_size=(10, 7), max_display=len(FEATURES),
    )
    fig = plt.gcf()
    fig.axes[0].set_title(
        'Step 5 — SHAP Beeswarm: Ridge Regression\n'
        f'target_max_log · full-data model · n={sample_n:,} sample',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step5_shap_ridge.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step5_shap_ridge.png')


# ── step6: SHAP beeswarm LightGBM ───────────────────────────────────────────
def step6_shap_lgb():
    import shap as _shap
    _shap.summary_plot(
        shap_lgb, X_samp, feature_names=FEATURES,
        plot_type='dot', show=False,
        plot_size=(10, 7), max_display=len(FEATURES),
    )
    fig = plt.gcf()
    fig.axes[0].set_title(
        'Step 6 — SHAP Beeswarm: LightGBM-LambdaRank\n'
        f'target_max_log · full-data model · n={sample_n:,} sample',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step6_shap_lgb.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step6_shap_lgb.png')


# ── step7: predicted vs actual ────────────────────────────────────────────────
def step7_predicted_vs_actual():
    key = KEY_RIDGE_MAX_SUB
    if key in pred_store and pred_store[key]:
        y_true = np.concatenate([p[0] for p in pred_store[key]])
        y_pred = np.concatenate([p[1] for p in pred_store[key]])
        subtitle = 'Sub-basin CV hold-out predictions (concatenated folds)'
    else:
        y_true   = y_all_max
        y_pred   = ridge_pred_all
        subtitle = 'Full-dataset predictions (in-sample)'

    sp = spearman_r(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.12, s=4,
               c='#3B4992', rasterized=True, label='Parcels')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, label='1 : 1 line')
    ax.annotate(f'ρ = {sp:.3f}', xy=(0.05, 0.91), xycoords='axes fraction',
                fontsize=14, fontweight='bold', color='#228833')
    ax.set_xlabel('Actual  target_max_log  (log flood depth, ft)')
    ax.set_ylabel('Predicted  target_max_log')
    ax.set_title(f'Step 7 — Predicted vs Actual: Ridge · target_max_log\n'
                 f'{subtitle}', fontsize=11)
    ax.legend(framealpha=0.9, markerscale=5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step7_predicted_vs_actual.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step7_predicted_vs_actual.png')


# ── step8: risk quintile map ──────────────────────────────────────────────────
def step8_risk_map():
    import geopandas as gpd

    gpkg_path = DATA_DIR / 'feature_matrix.gpkg'
    if not gpkg_path.exists():
        print(f'  ✗ step8: gpkg not found at {gpkg_path}')
        return

    gdf = gpd.read_file(gpkg_path)
    risk_df = pd.DataFrame({'parcel_id': df['parcel_id'],
                             'risk_pred': ridge_pred_all})
    gdf = gdf.merge(risk_df, on='parcel_id', how='left')

    # Quintile assignment (Q1 = lowest risk, Q5 = highest)
    gdf['quintile'] = pd.qcut(
        gdf['risk_pred'].rank(method='first', na_option='bottom'),
        5, labels=['Q1\n(lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(highest)'],
    )

    try:
        cmap5 = plt.colormaps['RdYlBu_r'].resampled(5)
    except AttributeError:
        cmap5 = plt.cm.get_cmap('RdYlBu_r', 5)

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        column='quintile', cmap=cmap5, ax=ax,
        legend=True,
        legend_kwds={'title': 'Risk Quintile', 'loc': 'lower right',
                     'framealpha': 0.9},
        linewidth=0,
        missing_kwds={'color': '#cccccc', 'label': 'No data'},
    )
    ax.set_title(
        'Step 8 — Predicted Flood Risk Quintiles  (Ridge · target_max_log)\n'
        'Q1 = lowest predicted risk  ·  Q5 = highest predicted risk',
        fontsize=13,
    )
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step8_top_bottom_parcels_map.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step8_top_bottom_parcels_map.png')


# ── step9: RF ROC curves ──────────────────────────────────────────────────────
def step9_rf_roc_curves():
    cls_colors = ['#3B4992', '#228833', '#EE9900', '#CC3311']
    cls_labels = ['Class 0 (lowest)', 'Class 1', 'Class 2', 'Class 3 (highest)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, fold in zip(axes, FOLDS):
        if fold not in rf_store:
            ax.set_visible(False)
            continue

        y_true_list, y_proba_list, _ = rf_store[fold]
        y_true_all  = np.concatenate(y_true_list)
        y_proba_all = np.vstack(y_proba_list)

        for c in range(4):
            fpr, tpr, _ = roc_curve(
                (y_true_all == c).astype(int), y_proba_all[:, c],
            )
            auc = float(np.trapz(tpr, fpr))
            ax.plot(fpr, tpr, color=cls_colors[c], linewidth=2,
                    label=f'{cls_labels[c]}  AUC={auc:.3f}')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('False Positive Rate')
        ax.set_title(FOLD_LABELS[fold], fontsize=12)
        ax.legend(framealpha=0.9, fontsize=9, loc='lower right')

    axes[0].set_ylabel('True Positive Rate')
    fig.suptitle(
        'Step 9 — Random Forest ROC Curves (One-vs-Rest per risk class)\n'
        'Classes from k-means (k=4) on [target_mean, target_max]',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step9_rf_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ step9_rf_roc_curves.png')


# ── run all ───────────────────────────────────────────────────────────────────
step1_cv_performance()
step2_baseline_comparison()
step3_per_fold_spearman()
step4_ridge_coefficients()
step5_shap_ridge()
step6_shap_lgb()
step7_predicted_vs_actual()
step8_risk_map()
step9_rf_roc_curves()

print('\n' + '=' * 65)
print('DONE')
print(f'  Results : {RESULTS_DIR / "cv_summary_v2.csv"}')
print(f'  Figures : {FIG_DIR}/')
print('=' * 65)
