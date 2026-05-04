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
| `adapt_apcluster_script.py` | Standalone script — full pipeline with CSV input, visualization, and result export |
| `extract_coords.py` | Utility script — extracts lat/long from Google Maps links in a CSV/Excel file |
| `export_similarity.py` | Utility script — computes and exports the N×N similarity matrix |
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

## Credits

Original MATLAB implementation by K. Wang et al., distributed under the BSD license.
Copyright © 2007–2008. Last modified: July 26, 2009.
Source: [MATLAB Central File Exchange #18244](http://www.mathworks.com/matlabcentral/fileexchange/18244)
