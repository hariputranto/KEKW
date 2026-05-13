# Adaptive Affinity Propagation — Python Port

Python port of the Adaptive Affinity Propagation (Adaptive AP) clustering algorithm, originally implemented in MATLAB by K. Wang et al. Also includes a utility script for extracting GPS coordinates from Google Maps links.

**Original paper:**
K. Wang, J. Zhang, D. Li, X. Zhang and T. Guo. *Adaptive Affinity Propagation Clustering.* Acta Automatica Sinica, 33(12):1242–1246, 2007.
([English preprint](http://arxiv.org/abs/0805.1096))

---

## Files

| File | Description |
|---|---|
| `adapt_apcluster.py` | Importable module — core algorithm only |
| `adapt_apcluster_script.py` | Standalone script — general-purpose pipeline with CSV input, visualisation, and result export |
| `adapt_apcluster_nofunc.py` | Sequential (no-function) version of the script — reference implementation used for comparison |
| `adapt_apcluster_spatial.py` | Spatial clustering script — optimised for projected (UTM) or geographic (lon/lat) coordinate data |
| `spatial_outlier_detect.py` | Spatial outlier detection — LOF, k-NN distance, and DBSCAN noise detection on coordinate data |
| `extract_coords.py` | Utility — extracts lat/long from Google Maps links in a CSV/Excel file |
| `export_similarity.py` | Utility — computes and exports the N×N similarity matrix |
| `adapt_apcluster.m` | Original MATLAB source |
| `wine.txt` / `ionosphere.txt` | Demo datasets |

---

## Requirements

```
numpy==2.4.4
scipy==1.17.1
pandas==3.0.2
scikit-learn==1.8.0
matplotlib==3.10.8
requests==2.33.1
openpyxl==3.1.5   # only needed for .xlsx input/output
```

Install all at once:

```bash
pip install numpy==2.4.4 scipy==1.17.1 pandas==3.0.2 scikit-learn==1.8.0 matplotlib==3.10.8 requests==2.33.1 openpyxl==3.1.5
```

---

## adapt_apcluster_script.py

End-to-end clustering pipeline. Edit the **PARAMETERS** block at the top of the file before running.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `data` | `load_data('wine.txt')` | Input data — `.txt` (whitespace-delimited) or `.csv` |
| `simatrix` | `False` | Set `True` if the input is a pre-computed similarity matrix |
| `sim_type` | `1` | `1` = Euclidean distance, `2` = Pearson similarity |
| `adapt` | `0` | Adaptiveness level (`0` or `1`; both enable adaptive mechanisms) |
| `nrun` | `50000` | Maximum iterations |
| `nconv` | `50` | Convergence window (larger = stricter) |
| `pstep` | `0.01` | Preference scan step size (smaller = finer search, slower) |
| `lam` | `0.5` | Initial damping factor |
| `cut` | `0` | Drop clusters with fewer than this many members |
| `output_csv` | `'ap_results.csv'` | Path for result CSV; `''` to skip |
| `output_plot` | `'ap_clusters.png'` | Path for cluster plot; `''` to skip |

### Running

```bash
python adapt_apcluster_script.py
```

### Output

- **Console** — progress per iteration, optimal number of clusters, Silhouette index
- **`ap_results.csv`** — one row per data point with columns: `feature_1..N`, `cluster`, `is_centre`, `centre_index`
- **`ap_clusters.png`** — scatter plot with cluster members, centres (stars), and connecting lines; PCA projection used automatically when data has more than 2 features

---

## adapt_apcluster.py

Importable module for use in your own scripts.

```python
from adapt_apcluster import adapt_apcluster

labels, labelid, NCs, Sil, Silmin, NCopt, Sid = adapt_apcluster(
    S,          # (N, N) similarity matrix or (N, d) data array
    simatrix=False,
    sim_type=1,
    adapt=0,
    nrun=50000,
    nconv=50,
    pstep=0.01,
    lam=0.5,
    cut=0,
)
```

**Returns:**

| Variable | Description |
|---|---|
| `labels` | `(N, n_solutions)` — cluster labels for each preference level scanned |
| `labelid` | `(N, n_solutions)` — exemplar (centre) index for each point |
| `NCs` | Number of clusters at each solution |
| `Sil` | Mean Silhouette index at each solution |
| `Silmin` | Min Silhouette index at each solution |
| `NCopt` | Optimal number of clusters |
| `Sid` | Column index of the optimal solution |

Optimal labels and exemplars:

```python
optimal_labels = labels[:, Sid]   # 1-based cluster assignments
optimal_centres = labelid[:, Sid] # 1-based exemplar indices
```

---

## extract_coords.py

Reads a CSV or Excel file containing Google Maps links and adds `lat`, `long`, and `longlat` columns.

### Parameters

Edit the **PARAMETERS** block at the top of the file:

| Parameter | Default | Description |
|---|---|---|
| `input_file` | `'places.csv'` | Input file path (`.csv` or `.xlsx`) |
| `url_column` | `'google_maps_link'` | Column name containing Google Maps URLs |
| `output_file` | `'places_coords.csv'` | Output file path; `''` to overwrite input |
| `delay` | `(15, 30)` | Random wait range (seconds) between HTTP requests |

### Running

```bash
python extract_coords.py
```

### Output columns added

| Column | Example | Description |
|---|---|---|
| `lat` | `3.148561` | Latitude |
| `long` | `101.686950` | Longitude |
| `longlat` | `101.686950,3.148561` | `longitude,latitude` string (suitable for GIS tools) |

### Supported URL formats

- Standard browser share URLs (`/@lat,lng,zoom`)
- Place URLs with embedded coordinates (`!3d{lat}!4d{lng}`)
- Direct coordinate search (`?q=lat,lng`)
- Short URLs (`maps.app.goo.gl`, `goo.gl/maps`) — resolved via HTTP redirect

### Bot-detection avoidance

- Random jitter delay between requests (15–30 s by default)
- Realistic browser headers (`Accept`, `Accept-Language`, `Referer`)
- Rotating User-Agent pool (Chrome, Firefox, Safari)

---

## export_similarity.py

Standalone script to compute, display, and export the N×N pairwise similarity matrix that `adapt_apcluster_script.py` uses internally.

### Parameters

Edit the **PARAMETERS** block at the top of the file:

| Parameter | Default | Description |
|---|---|---|
| `input_file` | `'wine.txt'` | Input data file — same file used in `adapt_apcluster_script.py` |
| `sim_type` | `'euclidean'` | `'euclidean'` = negative Euclidean distance, `'correlation'` = transformed Pearson correlation |
| `output_file` | `'similarity_matrix.csv'` | Output path; `''` to skip export |
| `print_limit` | `100` | Max rows/cols to print to console; `0` = print full matrix |

### Running

```bash
python export_similarity.py
```

### Output

- **Console** — matrix dimensions, off-diagonal min/max/mean/median, and a preview (or full matrix)
- **`similarity_matrix.csv`** — full N×N matrix with row and column labels `p1, p2, …`

### Similarity values

| `sim_type` | Formula | Range |
|---|---|---|
| `'euclidean'` | `S = −√(Σ(xᵢ−xⱼ)²)` | `(−∞, 0]`; 0 = identical |
| `'correlation'` | `S = −(0.5 − 0.5·r)` where `r` is Pearson correlation | `[−1, 0]`; 0 = identical |

Diagonal entries are set to `0.0` (the preference is assigned separately by `adapt_apcluster_script.py` before running).

---

## adapt_apcluster_spatial.py

Spatial clustering pipeline — a copy of `adapt_apcluster_script.py` optimised for geographic point data. Edit the **PARAMETERS** block before running.

**Key differences from `adapt_apcluster_script.py`:**

| Aspect | `script.py` | `spatial.py` |
|---|---|---|
| Distance | Squared Euclidean | Linear Euclidean (projected) or Haversine (geographic) |
| Initial preference | `pmedian × 0.5` | `pmedian` |
| Convergence window | 10 | 50 |
| Kvar escape step | `2 · √std` | `2 · std` (bolder) |
| Min cluster size | 3 | 1 |
| Plot aspect | arbitrary | `equal` (projected) or lat-corrected (geographic) |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `data_file` | `'your_data.csv'` | Path to input CSV |
| `x_col` | `'x'` | Column for easting (projected) or longitude (geographic) |
| `y_col` | `'y'` | Column for northing (projected) or latitude (geographic) |
| `crs_type` | `'projected'` | `'projected'` = UTM metres · `'geographic'` = decimal-degree lon/lat |
| `utm_zone` | `49` | UTM zone number 1–60; only used when `crs_type = 'projected'` |
| `utm_hemi` | `'S'` | `'N'` or `'S'`; only used when `crs_type = 'projected'` |
| `maxits` | `2000` | Maximum iterations |
| `convits` | `50` | Convergence window |
| `lam` | `0.7` | Initial damping factor |
| `folds` | `0.01` | Preference step factor |
| `cut` | `1` | Minimum cluster size (1 = allow singletons) |
| `truelabels` | `None` | True labels for validation, or `None` |
| `output_csv` | `'ap_spatial_results.csv'` | Output CSV path; `''` to skip |
| `output_plot` | `'ap_spatial_clusters.png'` | Output plot path; `''` to skip |

### Running

```bash
python adapt_apcluster_spatial.py
```

### Output

- **Console** — CRS info, distance range, per-iteration progress, Silhouette table
- **`ap_spatial_results.csv`** — all original columns preserved, plus `cluster`, `is_centre`, `centre_index`
- **`ap_spatial_clusters.png`** — geographic scatter plot with cluster members, exemplar centres (stars), and connecting spokes

---

## spatial_outlier_detect.py

Spatial outlier detection using three complementary methods. Edit the **PARAMETERS** block before running.

### Methods

| Method | What it detects |
|---|---|
| `lof` | Points in a locally sparse neighbourhood surrounded by denser areas (density contrast) |
| `knn` | Points absolutely far from their k-th nearest neighbour (geographic isolation) |
| `dbscan` | Points unreachable by any core point — DBSCAN noise label −1 (no cluster membership) |

Using all three methods and setting `vote_thresh` controls sensitivity: `1` = any method flags; `3` = unanimous agreement only.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `data_file` | `'your_data.csv'` | Path to input CSV |
| `x_col` | `'x'` | Column for easting or longitude |
| `y_col` | `'y'` | Column for northing or latitude |
| `crs_type` | `'projected'` | `'projected'` (UTM) or `'geographic'` (lon/lat) |
| `utm_zone` | `49` | UTM zone; only used when `crs_type = 'projected'` |
| `utm_hemi` | `'S'` | Hemisphere; only used when `crs_type = 'projected'` |
| `methods` | `['lof','knn','dbscan']` | Any subset of the three methods |
| `lof_k` | `20` | LOF neighbourhood size |
| `lof_thresh` | `1.5` | LOF score threshold for flagging |
| `knn_k` | `5` | k-th neighbour distance used as the score |
| `knn_thresh` | `None` | Auto: mean + 3σ of all k-NN distances |
| `dbscan_eps` | `None` | Auto: 95th-percentile of k-NN distances |
| `dbscan_minpts` | `5` | Minimum points to form a core point |
| `vote_thresh` | `1` | Minimum method votes to mark a point as outlier |
| `output_csv` | `'spatial_outliers.csv'` | Output CSV path; `''` to skip |
| `output_plot` | `'spatial_outliers.png'` | Output map path; `''` to skip |
| `output_kdist_plot` | `'kdist_plot.png'` | k-distance elbow plot to help choose `dbscan_eps`; `''` to skip |

### Running

```bash
python spatial_outlier_detect.py
```

### Output

- **Console** — flagged count per method, outlier index table with coordinates
- **`spatial_outliers.csv`** — all original columns preserved, plus `lof_score`, `knn_dist`, `dbscan_label`, `outlier_votes`, `is_outlier`
- **`spatial_outliers.png`** — geographic scatter plot with outliers marked as red X, annotated with vote count
- **`kdist_plot.png`** — sorted k-distance elbow plot with the auto eps threshold marked; use this to manually tune `dbscan_eps`

---

## Credits

Original MATLAB implementation by K. Wang et al., distributed under the BSD license.
Copyright © 2007–2008. Last modified: July 26, 2009.
Source: [MATLAB Central File Exchange #18244](http://www.mathworks.com/matlabcentral/fileexchange/18244)
