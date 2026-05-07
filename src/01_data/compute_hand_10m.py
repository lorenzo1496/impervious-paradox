#!/usr/bin/env python3
"""
compute_hand_10m.py  —  Custom 10m HAND from local DEM (pysheds 0.4)

Pipeline
  1   Inspect 10m DEM
  2   Hydrological conditioning: fill_pits → fill_depressions → resolve_flats
  3   D8 flow direction
  4   Flow accumulation + stream extraction (threshold = 1 000 cells = 0.1 km²)
  5   HAND computation (Height Above Nearest Drainage)
  6   Zonal mean HAND per parcel via exactextract
  7   Update feature_matrix.csv  (old HAND → HAND_90m, new HAND = 10m version)
  8   Update feature_matrix.gpkg
  9   Validation printout
  Viz 8 figures  →  ./outputs/figures/10m/
"""

import sys
import io
import time
import warnings
from pathlib import Path

# Force UTF-8 output on Windows (default console is cp1252)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np

# ---------------------------------------------------------------------------
# pysheds 0.4 calls np.can_cast(python_scalar, dtype) which NumPy 2.0 removed
# (NEP 50). Patch it before pysheds is imported.
# ---------------------------------------------------------------------------
if int(np.__version__.split(".")[0]) >= 2:
    _orig_can_cast = np.can_cast
    def _can_cast_compat(from_, to, casting="unsafe", **kw):
        if isinstance(from_, (int, float, complex)):
            try:
                from_ = np.dtype(to).type(from_)
            except (OverflowError, ValueError):
                return False
        return _orig_can_cast(from_, to, casting=casting, **kw)
    np.can_cast = _can_cast_compat

import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

from pysheds.grid import Grid
from exactextract import exact_extract

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DEM_IN      = ROOT / "data" / "raw" / "dem.tif"
DATA        = ROOT / "data" / "processed"
CSV         = DATA / "feature_matrix.csv"
GPKG        = DATA / "feature_matrix.gpkg"

INTER       = ROOT / "outputs" / "intermediate"
FIGS        = ROOT / "outputs" / "figures" / "10m"
STREAMS_OUT = INTER / "streams_10m.tif"
HAND_OUT    = INTER / "hand_10m.tif"

FLOW_THRESH = 1_000          # cells  ×  100 m²/cell  =  0.1 km²  contributing area
DIRMAP      = (64, 128, 1, 2, 4, 8, 16, 32)   # D8: NW N NE E SE S SW W
DPI         = 150

# ---------------------------------------------------------------------------
# Validate inputs / create output dirs
# ---------------------------------------------------------------------------
for p in [DEM_IN, CSV, GPKG]:
    if not p.exists():
        sys.exit(f"ERROR: missing file: {p}")

INTER.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib style — publication-ready
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tick(label, t0):
    print(f"      {time.time()-t0:6.1f}s  {label}")
    return time.time()


def write_raster(path, arr, ref_profile, nodata=np.nan):
    """Save arr as float32 GeoTIFF, inheriting CRS/transform from ref_profile."""
    p = {
        "driver": "GTiff",
        "dtype":  "float32",
        "count":  1,
        "crs":    ref_profile["crs"],
        "transform": ref_profile["transform"],
        "nodata": float(nodata),
        "width":  arr.shape[1],
        "height": arr.shape[0],
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)


def read_for_plot(path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        ext = [src.bounds.left, src.bounds.right,
               src.bounds.bottom, src.bounds.top]
    return arr, ext


def km_fmt(ax):
    """Format UTM metre axes as 'NNNk'."""
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))


# ===========================================================================
# 1. Inspect DEM
# ===========================================================================
print("=" * 65)
print("1. Input 10m DEM")
print("=" * 65)

with rasterio.open(DEM_IN) as src:
    dem_profile   = src.profile.copy()
    dem_crs       = src.crs
    dem_transform = src.transform
    dem_nodata    = src.nodata
    dem_arr_raw   = src.read(1)
    dem_extent    = [src.bounds.left, src.bounds.right,
                     src.bounds.bottom, src.bounds.top]

dem_vals = dem_arr_raw.ravel()
dem_vals = dem_vals[np.isfinite(dem_vals)]

print(f"   Shape:    {dem_arr_raw.shape[0]:,} × {dem_arr_raw.shape[1]:,} = "
      f"{dem_arr_raw.size/1e6:.2f}M pixels")
print(f"   CRS:      {dem_crs}")
print(f"   Res:      {abs(dem_transform.a):.0f} × {abs(dem_transform.e):.0f} m")
print(f"   Elev:     {dem_vals.min():.2f} – {dem_vals.max():.2f} m  "
      f"(mean {dem_vals.mean():.2f} m)")

# ===========================================================================
# 2. Hydrological conditioning
# ===========================================================================
print()
print("=" * 65)
print("2. Hydrological conditioning")
print("=" * 65)

t_pipe = time.time()
t0 = time.time()

grid = Grid.from_raster(str(DEM_IN))
dem  = grid.read_raster(str(DEM_IN))
t0 = _tick("DEM loaded into pysheds", t0)

pit_filled = grid.fill_pits(dem)
t0 = _tick("fill_pits", t0)

flooded    = grid.fill_depressions(pit_filled)
t0 = _tick("fill_depressions", t0)

inflated   = grid.resolve_flats(flooded)
t0 = _tick("resolve_flats", t0)

# ===========================================================================
# 3. D8 flow direction
# ===========================================================================
print()
print("=" * 65)
print("3. D8 flow direction")
print("=" * 65)
t0 = time.time()

fdir = grid.flowdir(inflated, dirmap=DIRMAP)
t0 = _tick("flowdir", t0)

# ===========================================================================
# 4. Flow accumulation + stream extraction
# ===========================================================================
print()
print("=" * 65)
print(f"4. Flow accumulation + stream extraction  (threshold = {FLOW_THRESH:,} cells)")
print("=" * 65)
t0 = time.time()

acc = grid.accumulation(fdir, dirmap=DIRMAP)
t0 = _tick("accumulation", t0)

max_acc = float(np.asarray(acc).max())
print(f"      Max acc: {max_acc:,.0f} cells = {max_acc*100/1e6:.1f} km²")

streams   = acc > FLOW_THRESH
n_stream  = int(np.asarray(streams).sum())
pct_str   = n_stream / dem_arr_raw.size * 100
print(f"      Stream pixels: {n_stream:,}  ({pct_str:.2f}% of DEM,  "
      f"{n_stream*100/1e6:.2f} km² stream area)")

streams_np = np.asarray(streams, dtype=np.float32)
write_raster(STREAMS_OUT, streams_np, dem_profile, nodata=255)
print(f"      Saved: {STREAMS_OUT}")

# ===========================================================================
# 5. HAND computation
# ===========================================================================
print()
print("=" * 65)
print("5. Computing HAND  (pysheds compute_hand, algorithm=iterative)")
print("=" * 65)
t0 = time.time()

# Use pit_filled elevations (not inflated) for physically correct height diffs.
# fdir from inflated ensures proper routing over flats;
# pit_filled DEM gives accurate elevation values at stream pixels.
hand = grid.compute_hand(
    fdir, pit_filled, streams,
    dirmap=DIRMAP, nodata_out=np.nan, algorithm="iterative",
)
t0 = _tick("compute_hand", t0)

hand_np = np.asarray(hand, dtype=np.float64)
hand_np = np.maximum(hand_np, 0.0)   # clamp fp noise below zero

h_valid = hand_np[np.isfinite(hand_np)]
print(f"      Valid pixels: {len(h_valid):,}  ({len(h_valid)/hand_np.size*100:.1f}%)")
print(f"      Range:  {h_valid.min():.2f} – {h_valid.max():.2f} m")
print(f"      Mean:   {h_valid.mean():.2f} m   Median: {np.median(h_valid):.2f} m")

write_raster(HAND_OUT, hand_np, dem_profile, nodata=np.nan)
print(f"      Saved: {HAND_OUT}")
print(f"\n      Total pipeline: {time.time()-t_pipe:.0f}s")

# ===========================================================================
# 6. Zonal stats — exactextract
# ===========================================================================
print()
print("=" * 65)
print("6. Zonal stats — exactextract")
print("=" * 65)

with rasterio.open(HAND_OUT) as src:
    hand_crs = src.crs

gdf = gpd.read_file(GPKG)
print(f"   Parcels: {len(gdf):,}  CRS: {gdf.crs}")

if str(gdf.crs) != str(hand_crs):
    gdf_proj = gdf.to_crs(hand_crs)
    print(f"   Reprojected to {hand_crs}")
else:
    gdf_proj = gdf
    print("   CRS match — no reprojection needed")

t0 = time.time()
result = exact_extract(
    rast=str(HAND_OUT),
    vec=gdf_proj,
    ops=["mean", "min"],
    output="pandas",
    include_cols=["parcel_id"],
)
print(f"   exactextract: {time.time()-t0:.1f}s")
print(f"   Result columns: {result.columns.tolist()}")

# Normalise: find mean and min columns regardless of band-name prefix
def _find_col(df, suffix):
    col = next((c for c in df.columns if c == suffix or c.endswith(f"_{suffix}")), None)
    if col is None:
        nc = [c for c in df.columns
              if c != "parcel_id" and pd.api.types.is_numeric_dtype(df[c])]
        col = nc[0] if nc else None
    return col

mean_col = _find_col(result, "mean")
min_col  = _find_col(result, "min")
if mean_col is None:
    sys.exit("ERROR: cannot identify mean column in exactextract output")
if min_col is None:
    sys.exit("ERROR: cannot identify min column in exactextract output")

result = result.rename(columns={mean_col: "HAND_10m", min_col: "HAND_10m_min"})

for col_name, label in [("HAND_10m", "mean"), ("HAND_10m_min", "min")]:
    n_ok   = result[col_name].notna().sum()
    n_miss = result[col_name].isna().sum()
    med    = result[col_name].median()
    print(f"   HAND {label:4s}: {n_ok:,} ok  |  {n_miss:,} missing  "
          f"(median = {med:.2f} m)")
    if n_miss > 0:
        result[col_name] = result[col_name].fillna(med)

med_10     = result["HAND_10m"].median()
med_10_min = result["HAND_10m_min"].median()

# ===========================================================================
# 7. Update feature_matrix.csv
# ===========================================================================
print()
print("=" * 65)
print("7. Updating feature_matrix.csv")
print("=" * 65)

df = pd.read_csv(CSV)
print(f"   Existing: {len(df):,} rows × {len(df.columns)} cols")

# Archive 90m as HAND_90m; install 10m mean as HAND, 10m min as HAND_min
if "HAND" in df.columns and "HAND_90m" not in df.columns:
    df = df.rename(columns={"HAND": "HAND_90m"})
    print("   Renamed HAND → HAND_90m  (kept for comparison)")
elif "HAND" in df.columns:
    df = df.drop(columns=["HAND"])   # HAND_90m already present; drop old primary

for stale in ["HAND_min"]:
    if stale in df.columns:
        df = df.drop(columns=[stale])

new_cols = (result[["parcel_id", "HAND_10m", "HAND_10m_min"]]
            .rename(columns={"HAND_10m": "HAND", "HAND_10m_min": "HAND_min"}))
df_out = df.merge(new_cols, on="parcel_id", how="left")

for col_name, med in [("HAND", med_10), ("HAND_min", med_10_min)]:
    unmatched = df_out[col_name].isna().sum()
    if unmatched:
        df_out[col_name] = df_out[col_name].fillna(med)
        print(f"   Imputed {unmatched:,} unmatched {col_name} rows")

assert len(df_out) == len(df), "Row count changed!"
df_out.to_csv(CSV, index=False)
print(f"   Saved: {CSV}  ({len(df_out.columns)} cols)")

# ===========================================================================
# 8. Update feature_matrix.gpkg
# ===========================================================================
print()
print("=" * 65)
print("8. Updating feature_matrix.gpkg")
print("=" * 65)

for col in ["HAND", "HAND_min", "HAND_90m"]:
    if col in gdf.columns:
        gdf = gdf.drop(columns=[col])

keep_cols = (["parcel_id", "HAND", "HAND_min"]
             + (["HAND_90m"] if "HAND_90m" in df_out.columns else []))
gdf_out = gdf.merge(df_out[keep_cols], on="parcel_id", how="left")
for col_name, med in [("HAND", med_10), ("HAND_min", med_10_min)]:
    if gdf_out[col_name].isna().any():
        gdf_out[col_name] = gdf_out[col_name].fillna(med)

gdf_out.to_file(GPKG, driver="GPKG")
print(f"   Saved: {GPKG}  ({len(gdf_out.columns)} cols, CRS: {gdf_out.crs})")

# ===========================================================================
# 9. Validation
# ===========================================================================
print()
print("=" * 65)
print("VALIDATION — 10m HAND distribution  (per-parcel)")
print("=" * 65)
print(f"   {'stat':<8}  {'HAND mean (m)':>14}  {'HAND min (m)':>14}")
print(f"   {'-'*8}  {'-'*14}  {'-'*14}")
for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
    dm = df_out["HAND"].describe()[stat]
    dk = df_out["HAND_min"].describe()[stat]
    fmt = "{:>14,.0f}" if stat == "count" else "{:>14.3f}"
    print(f"   {stat:<8}  {fmt.format(dm)}  {fmt.format(dk)}")

print()
print("=" * 65)
print("VALIDATION — HAND variants vs flood targets  (Pearson r)")
print("=" * 65)
hdr = f"   {'Target':<26}  {'r(90m mean)':>12}  {'r(10m mean)':>12}  {'r(10m min)':>12}"
print(hdr)
print("   " + "-" * (len(hdr) - 3))
for col in ["target_mean", "target_mean_log", "target_max_log"]:
    if col not in df_out.columns:
        continue
    r90  = df_out["HAND_90m"].corr(df_out[col]) if "HAND_90m" in df_out.columns else np.nan
    r10m = df_out["HAND"].corr(df_out[col])
    r10k = df_out["HAND_min"].corr(df_out[col])
    r90s = f"{r90:+.4f}" if not np.isnan(r90) else "     n/a"
    print(f"   {col:<26}  {r90s:>12}  {r10m:>+12.4f}  {r10k:>+12.4f}")

print()
print("=" * 65)
print("VALIDATION — Feature ranking vs target_mean_log")
print("   (HAND = 10m mean,  HAND_min = 10m minimum per parcel)")
print("=" * 65)
ranked = []
if "target_mean_log" in df_out.columns:
    exclude = ({"parcel_id"}
               | {c for c in df_out.columns if c.startswith("target")}
               | {"HAND_90m"})
    feat_cols = [c for c in df_out.select_dtypes(include=[np.number]).columns
                 if c not in exclude]
    corrs = {c: df_out[c].corr(df_out["target_mean_log"]) for c in feat_cols}
    corrs = {k: v for k, v in corrs.items() if pd.notna(v)}
    ranked = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for rank, (feat, r) in enumerate(ranked, 1):
        tag = (" << HAND mean (10m)" if feat == "HAND"
               else " << HAND min  (10m)" if feat == "HAND_min"
               else "")
        print(f"   {rank:2d}.  {feat:<30}  {r:+.4f}{tag}")

print()
print("Done with data pipeline.")

# ===========================================================================
# VISUALIZATIONS
# ===========================================================================
print()
print("=" * 65)
print(f"Generating 8 figures  →  {FIGS}")
print("=" * 65)

# ---------------------------------------------------------------------------
# Fig 1 — Input 10m DEM
# ---------------------------------------------------------------------------
print("  [1/8] 10m DEM...")
dem_plot, dem_ext = read_for_plot(DEM_IN)
dv = dem_plot.compressed()

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(dem_plot, cmap="terrain", extent=dem_ext, aspect="equal",
               vmin=float(np.percentile(dv, 1)),
               vmax=float(np.percentile(dv, 99)),
               origin="upper")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("Elevation (m)")
ax.set_title("Input 10m DEM — Brays Bayou Watershed")
ax.set_xlabel("Easting (m, UTM 15N)")
ax.set_ylabel("Northing (m)")
km_fmt(ax)
plt.tight_layout()
fig.savefig(FIGS / "step1_dem_10m.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2 — Stream network overlaid on DEM
# ---------------------------------------------------------------------------
print("  [2/8] Stream network...")
streams_plot, streams_ext = read_for_plot(STREAMS_OUT)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(dem_plot, cmap="Greys_r", extent=dem_ext, aspect="equal",
          vmin=float(np.percentile(dv, 2)),
          vmax=float(np.percentile(dv, 98)),
          origin="upper", alpha=0.65)

s_arr = np.array(streams_plot)
rgba = np.zeros((*s_arr.shape, 4))
rgba[..., 2] = 0.5               # blue
rgba[..., 3] = np.where(s_arr > 0.5, 0.85, 0.0)
ax.imshow(rgba, extent=streams_ext, aspect="equal", origin="upper")

ax.set_title(
    f"Derived Stream Network  (D8 acc > {FLOW_THRESH:,} cells = 0.1 km²)\n"
    f"{n_stream:,} stream pixels  ·  {n_stream*100/1e6:.2f} km² total stream area"
)
ax.set_xlabel("Easting (m, UTM 15N)")
ax.set_ylabel("Northing (m)")
km_fmt(ax)
ax.legend(handles=[Patch(color=(0, 0, 0.5, 0.85),
                         label=f"Streams (acc > {FLOW_THRESH:,} cells)")],
          frameon=False)
plt.tight_layout()
fig.savefig(FIGS / "step2_stream_network.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3 — 10m HAND raster
# ---------------------------------------------------------------------------
print("  [3/8] HAND raster...")
hand_plot, hand_ext = read_for_plot(HAND_OUT)
hv = hand_plot.compressed()

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(hand_plot, cmap="YlOrRd_r", extent=hand_ext, aspect="equal",
               vmin=0, vmax=float(np.percentile(hv, 95)), origin="upper")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("HAND (m)")
ax.set_title(
    "10m HAND Raster — Brays Bayou Watershed\n"
    "Height Above Nearest Drainage  (D8, stream threshold = 0.1 km²)"
)
ax.set_xlabel("Easting (m, UTM 15N)")
ax.set_ylabel("Northing (m)")
km_fmt(ax)
plt.tight_layout()
fig.savefig(FIGS / "step3_hand_raster_10m.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4 — Per-parcel HAND distribution (histogram + boxplot)
# ---------------------------------------------------------------------------
print("  [4/8] HAND distribution...")
hvals  = df_out["HAND"].dropna()
h_mean = hvals.mean()
h_med  = hvals.median()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                gridspec_kw={"width_ratios": [3, 1]})
ax1.hist(hvals, bins=80, color="#4878CF", edgecolor="none", alpha=0.85)
ax1.axvline(h_mean, color="#C44E52", lw=2,
            label=f"Mean    {h_mean:.2f} m")
ax1.axvline(h_med,  color="#55A868", lw=2, ls="--",
            label=f"Median  {h_med:.2f} m")
ax1.set_xlabel("Mean HAND per parcel (m)")
ax1.set_ylabel("Parcel count")
ax1.set_title("Per-parcel 10m HAND Distribution")
ax1.legend(frameon=False)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax2.boxplot(
    hvals, vert=True, patch_artist=True, widths=0.5,
    boxprops=dict(facecolor="#4878CF", alpha=0.6),
    medianprops=dict(color="#55A868", lw=2),
    whiskerprops=dict(color="gray"),
    capprops=dict(color="gray"),
    flierprops=dict(marker=".", color="#4878CF", alpha=0.15, markersize=2),
)
ax2.set_ylabel("HAND (m)")
ax2.set_title("Boxplot")
ax2.set_xticks([])
ax2.spines["bottom"].set_visible(False)

plt.suptitle("10m HAND Distribution — 118,119 Parcels", y=1.02, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGS / "step4_hand_distribution_10m.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5 — Parcel choropleth (10m HAND)
# ---------------------------------------------------------------------------
print("  [5/8] Parcel choropleth (10m HAND) — ~90s for 118k polygons...")
fig, ax = plt.subplots(figsize=(13, 9))
gdf_out.plot(column="HAND", ax=ax, cmap="YlOrRd", linewidth=0,
             legend=True,
             legend_kwds={"label": "Mean HAND — 10m (m)", "shrink": 0.6})
ax.set_title("Mean 10m HAND per Parcel — Brays Bayou Watershed")
ax.set_axis_off()
plt.tight_layout()
fig.savefig(FIGS / "step5_hand_map_10m.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6 — Scatter: 10m HAND vs flood targets (3-panel)
# ---------------------------------------------------------------------------
print("  [6/8] Scatter plots...")
target_meta = [
    ("target_mean",     "Mean flood depth (ft)"),
    ("target_mean_log", "log(mean flood depth)  [ft]"),
    ("target_max_log",  "log(max flood depth)  [ft]"),
]
target_meta = [(c, l) for c, l in target_meta if c in df_out.columns]

fig, axes = plt.subplots(1, len(target_meta),
                         figsize=(5 * len(target_meta), 5))
if len(target_meta) == 1:
    axes = [axes]
for ax, (col, ylabel) in zip(axes, target_meta):
    ax.scatter(df_out["HAND"], df_out[col],
               alpha=0.10, s=1.5, color="#4878CF", rasterized=True)
    r = df_out["HAND"].corr(df_out[col])
    ax.text(0.97, 0.97, f"r = {r:+.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, fontweight="bold", color="#C44E52")
    ax.set_xlabel("Mean HAND — 10m (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"10m HAND vs {col}")
plt.suptitle("10m HAND vs Flood Depth Targets  (alpha = 0.1 for density)",
             y=1.02, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGS / "step6_hand_vs_targets_10m.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 7 — Side-by-side 90m vs 10m comparison choropleth
# ---------------------------------------------------------------------------
print("  [7/8] Comparison map (90m vs 10m) — ~2min for 2×118k polygons...")
if "HAND_90m" in gdf_out.columns and "target_mean_log" in df_out.columns:
    vmin = 0.0
    vmax = float(max(
        df_out["HAND_90m"].quantile(0.98),
        df_out["HAND"].quantile(0.98),
    ))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    r90 = df_out["HAND_90m"].corr(df_out["target_mean_log"])
    r10 = df_out["HAND"].corr(df_out["target_mean_log"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    for ax, col, title in [
        (ax1, "HAND_90m", f"MERIT 90m HAND\nr = {r90:+.4f} vs target_mean_log"),
        (ax2, "HAND",     f"Custom 10m HAND\nr = {r10:+.4f} vs target_mean_log"),
    ]:
        gdf_out.plot(column=col, ax=ax, cmap="YlOrRd",
                     linewidth=0, vmin=vmin, vmax=vmax, legend=False)
        ax.set_title(title)
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], fraction=0.012, pad=0.01,
                 label="Mean HAND (m)")
    fig.suptitle("HAND Resolution Comparison — Brays Bayou Watershed",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGS / "step7_hand_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
else:
    print("    (skipped — HAND_90m or target_mean_log not available)")

# ---------------------------------------------------------------------------
# Fig 8 — Feature correlation ranking (10m HAND highlighted)
# ---------------------------------------------------------------------------
print("  [8/8] Feature correlation bar chart...")
if ranked:
    feats  = [f for f, _ in ranked]
    cvals  = [r for _, r in ranked]

    def _bar_color(f):
        if f == "HAND":     return "#C44E52"   # red    — 10m mean
        if f == "HAND_min": return "#E07B39"   # orange — 10m min
        return "#4878CF"                        # blue   — other

    colors = [_bar_color(f) for f in feats]

    # Reverse so strongest |r| is at top
    feats_r, cvals_r, colors_r = (list(reversed(x))
                                   for x in (feats, cvals, colors))
    y = range(len(feats_r))

    fig, ax = plt.subplots(figsize=(9, max(5, 0.48 * len(feats))))
    bars = ax.barh(list(y), cvals_r, color=colors_r,
                   height=0.65, edgecolor="none")
    ax.set_yticks(list(y))
    ax.set_yticklabels(feats_r)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r  with  target_mean_log")
    ax.set_title("Feature Correlations with target_mean_log\n"
                 "(HAND mean = red,  HAND min = orange,  others = blue)")

    for bar, v in zip(bars, cvals_r):
        ha, off = ("left", 0.003) if v >= 0 else ("right", -0.003)
        ax.text(v + off, bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center", ha=ha, fontsize=9)

    ax.legend(
        handles=[
            Patch(facecolor="#C44E52", label="HAND mean — 10m"),
            Patch(facecolor="#E07B39", label="HAND min  — 10m"),
            Patch(facecolor="#4878CF", label="Other features"),
        ],
        frameon=False, loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(FIGS / "step8_feature_correlations_10m.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 65)
saved = sorted(FIGS.glob("step*.png"))
print(f"Saved {len(saved)} figures  →  {FIGS}")
for f in saved:
    print(f"  {f.name:<54}  {f.stat().st_size/1024:6.0f} KB")
print()
print("All done.")
