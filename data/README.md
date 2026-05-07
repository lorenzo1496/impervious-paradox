# Data Sources

Raw and processed data files are not distributed with this repository due to
file-size constraints. This document describes every data source used in the
pipeline: where to obtain it, its license, approximate file size, and any
preprocessing applied before it enters the feature matrix.

Place downloaded files at the paths shown under **Destination** before running
`src/01_data/compute_hand_10m.py`.

---

## 1. HCAD Parcel Data

| Field | Value |
|---|---|
| **Provider** | Harris County Appraisal District (HCAD) |
| **Download** | https://hcad.org/hcad-resources/hcad-shapefiles-and-data/ → "Real Property GIS Data" |
| **License** | Public record (Harris County, Texas) |
| **Acquired** | 2024 |
| **File size** | ~93 MB (raw) |
| **Destination** | `data/raw/hcad_parcels_raw.gpkg` |

**Variables used:** parcel geometry (polygon), lot area (`log_lot_area`), enclave
flag (`is_enclave`), land-use category.

**Preprocessing:** Clip to Brays Bayou watershed boundary. Reproject to
EPSG:26915 (UTM zone 15N). Deduplicate by account number. Flag parcels fully
enclosed by other parcels as enclaves.

**Use restrictions:** HCAD parcel data is public record under Texas open
records law but is provided for non-commercial use only. Commercial
redistribution requires permission from HCAD. This research project complies
with the non-commercial restriction.

---

## 2. FEMA Hurricane Harvey Flood Depth (3 m)

| Field | Value |
|---|---|
| **Provider** | FEMA / Texas Water Development Board |
| **Download** | FEMA Flood Map Service Center: https://msc.fema.gov/portal/home — search Harvey depth grids; or via TWDB Harvey data repository |
| **License** | U.S. Government public domain |
| **Event** | Hurricane Harvey, 25–31 August 2017 |
| **Resolution** | 3 m |
| **File size** | ~153 MB (raw TIF) |
| **Destination** | `data/raw/harvey_depth_brays.tif` |

**Variables used:** `target_mean` and `target_bin` — zonal mean flood depth (ft)
per parcel and binary flooded indicator.

**Preprocessing:** Reproject to EPSG:26915. Clip to Brays Bayou watershed.
Zonal mean extracted per parcel centroid via `exactextract`. Within the
FEMA-modeled extent, no-data pixels indicate non-flooded land and are
interpreted as depth = 0. Parcels falling entirely outside the FEMA extent
are excluded from the analysis. This is the resolution of the earlier
"nodata = dry" issue documented in the changelog.

---

## 3. GHSL Impervious Surface Area 2015

| Field | Value |
|---|---|
| **Provider** | European Commission Joint Research Centre (JRC) |
| **Product** | GHS-BUILT-S R2023A — Impervious Surface fraction |
| **Download** | https://ghsl.jrc.ec.europa.eu/download.php → GHS_BUILT_S_NRES_E2015_GLOBE_R2023A |
| **License** | CC BY 4.0 |
| **Resolution** | 10 m |
| **File size** | ~43 MB (clipped TIF) |
| **Destination** | `data/raw/ghsl_2015.tif` |

**Variables used:** `ISA_frac` — zonal mean impervious surface fraction per
parcel (0–1).

**Preprocessing:** Clip to study area extent. Reproject to EPSG:26915.
Zonal mean per parcel via `exactextract`.

---

## 4. USGS 3DEP 10 m Digital Elevation Model

| Field | Value |
|---|---|
| **Provider** | USGS 3D Elevation Program (3DEP) |
| **Download** | https://apps.nationalmap.gov/downloader/ → Elevation Products → 1/3 arc-second DEM; or `py3dep` Python package |
| **License** | U.S. Government public domain |
| **Resolution** | 10 m (1/3 arc-second) |
| **File size** | ~24 MB (clipped TIF) |
| **Destination** | `data/raw/dem.tif` |

**Variables used:** `elevation`, `slope`, `TWI` (topographic wetness index),
`log_flow_accum`, `HAND_min` (Height Above Nearest Drainage).

**Preprocessing:** All derived rasters computed in `src/01_data/compute_hand_10m.py`:
hydrological conditioning (fill pits → fill depressions → resolve flats),
D8 flow direction, flow accumulation, stream extraction at 1,000-cell threshold
(≈ 0.1 km²), HAND via nearest-drainage assignment. Zonal statistics per parcel
via `exactextract`.

---

## 5. NHD Flowlines (Brays Bayou watershed)

| Field | Value |
|---|---|
| **Provider** | USGS National Hydrography Dataset (NHD) |
| **Download** | https://www.usgs.gov/national-hydrography/access-national-hydrography-products → NHD Best Resolution, HUC-8 12040104 (Brays Bayou) |
| **License** | U.S. Government public domain |
| **File size** | ~0.1 MB (clipped GPKG) |
| **Destination** | `data/raw/nhd_streams_brays.gpkg` |

**Variables used:** Stream overlay for HAND computation; `dist_to_stream`
(Euclidean distance from parcel centroid to nearest NHD flowline).

**Note:** This file is small enough to distribute but is excluded from the
repository for consistency with the "no raw data" policy. Download as above and
clip to the study area boundary.

---

## 6. NLCD 2016 — National Land Cover Database

| Field | Value |
|---|---|
| **Provider** | Multi-Resolution Land Characteristics Consortium (MRLC) / USGS |
| **Download** | https://www.mrlc.gov/data → NLCD 2016 Land Cover CONUS |
| **License** | U.S. Government public domain |
| **Resolution** | 30 m |
| **File size** | ~2 MB (clipped and reprojected TIF) |
| **Destination** | `data/raw/nlcd_2016_brays.tif` |

**Variables used:** `nlcd_class` — integer land-cover code per parcel (modal
value). Residential filter applied in `src/01_data/add_nlcd_filter.py`:
classes 22 (developed, low intensity), 23 (developed, medium intensity),
24 (developed, high intensity).

**Preprocessing:** Clip to study area. Reproject to EPSG:26915
(`nlcd_2016_brays_26915.tif`). Zonal mode per parcel via `exactextract`.

---

## 7. OpenStreetMap Street Network

| Field | Value |
|---|---|
| **Provider** | OpenStreetMap contributors |
| **Download** | Via `osmnx` Python package: `osmnx.graph_from_place("Brays Bayou, Houston, TX")`, or Geofabrik Texas extract: https://download.geofabrik.de/north-america/us/texas.html |
| **License** | Open Database License (ODbL) |
| **File size** | ~22 MB |
| **Destination** | `data/raw/osm_streets.gpkg` |

**Variables used:** `dist_to_street` — Euclidean distance from parcel centroid
to nearest OSM street centerline.

**Preprocessing:** Extract road geometries for Harris County extent. Reproject
to EPSG:26915. Dissolve to single multilinestring for distance raster
computation.

**Attribution requirement (ODbL § 4.3):** Any derived work using these data
must include the notice *"© OpenStreetMap contributors, available under the
Open Database License (ODbL)"* — see https://www.openstreetmap.org/copyright.
The paper acknowledgements section carries this attribution.

---

## Processed data

`data/processed/` holds the feature matrix and intermediate rasters generated
by running `src/01_data/`. These files are gitignored due to size.

| File | Size | Generated by |
|---|---|---|
| `feature_matrix.gpkg` | ~47 MB | `compute_hand_10m.py` + `add_nlcd_filter.py` |
| `feature_matrix.csv` | ~27 MB | same |
| `harvey_depth_reproj.tif` | ~379 MB | `compute_hand_10m.py` |
| `hand_brays.tif` | ~0.4 MB | `compute_hand_10m.py` |

Regenerate by running Stage 1 of the pipeline (see root `README.md`).
