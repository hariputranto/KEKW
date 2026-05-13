"""
Spatial Outlier Detection
=========================
Companion to adapt_apcluster_spatial.py.
Detects spatial outliers in projected (UTM) or geographic (lon/lat) point data
using three complementary methods:

  lof    — Local Outlier Factor: compares local density of a point against
            its neighbours; captures local anomalies that global methods miss.
  knn    — k-NN distance score: distance to the k-th nearest neighbour;
            large values indicate geographic isolation. Simple and interpretable.
  dbscan — DBSCAN noise detection: points that are not reachable by any core
            point at scale eps are labelled -1 (noise / outlier).

The three methods see different aspects of "outlier-ness":
  - LOF   → density contrast  (a point surrounded by dense neighbours, itself sparse)
  - k-NN  → absolute isolation (just far from everyone)
  - DBSCAN→ cluster membership (does not belong to any coherent group at given scale)

Output columns appended to the original data:
  lof_score    — raw LOF score (> lof_thresh → flagged)
  knn_dist     — distance to the k-th nearest neighbour
  dbscan_label — DBSCAN cluster label  (-1 = noise / outlier)
  outlier_votes — number of methods that flag the point
  is_outlier   — True if outlier_votes >= vote_thresh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

data_file  = 'your_data.csv'
x_col      = 'x'   # column for x  →  easting   (projected)  |  longitude  (geographic)
y_col      = 'y'   # column for y  →  northing  (projected)  |  latitude   (geographic)

# ── Coordinate Reference System ───────────────────────────────────────────────
#   'projected'  → coordinates in metres (UTM); Euclidean distance
#   'geographic' → decimal-degree lon/lat (WGS84); Haversine distance
crs_type   = 'projected'
utm_zone   = 49    # zone number 1–60; only used when crs_type = 'projected'
utm_hemi   = 'S'   # 'N' or 'S';       only used when crs_type = 'projected'

# ── Methods ───────────────────────────────────────────────────────────────────
methods    = ['lof', 'knn', 'dbscan']   # any subset of these three

# LOF
lof_k       = 20     # neighbourhood size
lof_thresh  = 1.5    # LOF score threshold; points above this are flagged

# k-NN distance
knn_k       = 5      # which neighbour's distance to use as the score
knn_thresh  = None   # None → auto: mean + 3 × std of all k-NN distances

# DBSCAN
dbscan_eps    = None  # None → auto: 95th-percentile of k-NN distances (same k as knn_k)
dbscan_minpts = 5     # minimum points to form a core point

# Consensus
vote_thresh = 1   # flag as outlier if this many methods agree (1 = any single method)

# Output
output_csv       = 'spatial_outliers.csv'
output_plot      = 'spatial_outliers.png'
output_kdist_plot = 'kdist_plot.png'   # k-distance elbow plot to help choose dbscan_eps; '' to skip


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def haversine_matrix(coords):
    """[longitude, latitude] decimal degrees → (N, N) great-circle distances in km."""
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
    dlon = lon[:, np.newaxis] - lon[np.newaxis, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :])
         * np.sin(dlon / 2) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def euclidean_matrix(data):
    from scipy.spatial.distance import cdist
    return cdist(data, data, metric='euclidean')


def load_spatial_data(filepath, x_col, y_col):
    """Load CSV → ((N, 2) coordinate array, full DataFrame)."""
    df = pd.read_csv(filepath)
    x = df.iloc[:, x_col].values.astype(float) if isinstance(x_col, int) else df[x_col].values.astype(float)
    y = df.iloc[:, y_col].values.astype(float) if isinstance(y_col, int) else df[y_col].values.astype(float)
    return np.column_stack([x, y]), df


def _ax_crs(ax, coords, crs_type):
    """Apply correct aspect ratio and axis labels for the given CRS."""
    if crs_type == 'geographic':
        mid_lat = float(np.median(coords[:, 1]))
        aspect  = 1.0 / np.cos(np.radians(mid_lat))
        ax.set_aspect(aspect if np.isfinite(aspect) and aspect > 0 else 'equal')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
    else:
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')


def plot_outliers(coords, is_outlier, votes, crs_type='projected',
                  title='Spatial Outlier Detection', save_path=''):
    """Single map: normals in blue, outliers as red X with vote count annotation."""
    x, y = coords[:, 0], coords[:, 1]
    _, ax = plt.subplots(figsize=(10, 8))

    mask_in  = ~is_outlier
    mask_out =  is_outlier

    ax.scatter(x[mask_in],  y[mask_in],  c='steelblue', s=30, alpha=0.55,
               zorder=2, label=f'Normal  ({mask_in.sum()})')
    ax.scatter(x[mask_out], y[mask_out], c='crimson',   s=100, alpha=0.9,
               marker='X', zorder=3, edgecolors='black', linewidths=0.6,
               label=f'Outlier ({mask_out.sum()})')

    for xi, yi, v in zip(x[mask_out], y[mask_out], votes[mask_out]):
        ax.annotate(str(v), (xi, yi), textcoords='offset points', xytext=(5, 5),
                    fontsize=7, color='darkred')

    _ax_crs(ax, coords, crs_type)
    ax.set_title(f'{title}  ({mask_out.sum()} outliers / {len(is_outlier)} points)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'## Plot saved to: {save_path}')
    plt.show()


def plot_kdist(kd, eps_auto, k, dist_unit, save_path=''):
    """
    k-distance elbow plot — sorted k-NN distances in ascending order.
    The 'elbow' (sharp bend) is a natural choice for DBSCAN eps.
    The auto-chosen value (95th percentile) is marked for reference.
    """
    sorted_kd = np.sort(kd)[::-1]   # descending: elbow visible as a knee
    _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sorted_kd, lw=1.2, color='steelblue')
    ax.axhline(eps_auto, color='crimson', lw=1.2, linestyle='--',
               label=f'auto eps = {eps_auto:.4f} {dist_unit}  (95th pct)')
    ax.set_xlabel('Points sorted by k-NN distance (descending)')
    ax.set_ylabel(f'{k}-NN distance ({dist_unit})')
    ax.set_title(f'k-Distance Plot  (k={k})  — choose eps at the elbow')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'## k-distance plot saved to: {save_path}')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA & BUILD DISTANCE MATRIX
# ═════════════════════════════════════════════════════════════════════════════

print(f'==> Loading spatial data from {data_file} ...')
coords, df_input = load_spatial_data(data_file, x_col, y_col)
N = coords.shape[0]

if crs_type == 'projected':
    crs_label              = f'Projected — UTM Zone {utm_zone}{utm_hemi} (metres)'
    x_label, y_label, unit = 'Easting', 'Northing', 'm'
    coord_fmt              = '.2f'
else:
    crs_label              = 'Geographic — decimal-degree lon/lat (WGS84)'
    x_label, y_label, unit = 'Longitude', 'Latitude', '°'
    coord_fmt              = '.6f'

print(f'    {N} points  |  CRS: {crs_label}')
print(f'    {x_label} [{coords[:,0].min():{coord_fmt}}, {coords[:,0].max():{coord_fmt}}] {unit}  '
      f'{y_label} [{coords[:,1].min():{coord_fmt}}, {coords[:,1].max():{coord_fmt}}] {unit}')

print('\n==> Building distance matrix ...')
if crs_type == 'geographic':
    Dist      = haversine_matrix(coords)
    dist_unit = 'km'
else:
    Dist      = euclidean_matrix(coords)
    dist_unit = 'm'
off_diag = Dist[~np.eye(N, dtype=bool)]
print(f'    Distance range: [{off_diag.min():.3f}, {off_diag.max():.3f}] {dist_unit}')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — OUTLIER DETECTION
# ═════════════════════════════════════════════════════════════════════════════

print('\n==> Running outlier detection ...')

result_cols = {}
flags       = np.zeros(N, dtype=int)   # vote accumulator

# ── k-NN distances (computed once; shared by knn, dbscan auto-eps, kdist plot) ─
print(f'    Pre-computing {knn_k}-NN distances ...')
nbrs     = NearestNeighbors(n_neighbors=knn_k + 1, metric='precomputed')
nbrs.fit(Dist)
knn_dists_all, _ = nbrs.kneighbors(Dist)
kd               = knn_dists_all[:, -1]   # distance to k-th neighbour (col 0 = self = 0)

# ── k-distance elbow plot ─────────────────────────────────────────────────────
eps_auto = float(np.percentile(kd, 95))
if output_kdist_plot:
    plot_kdist(kd, eps_auto, knn_k, dist_unit, save_path=output_kdist_plot)

# ── LOF ───────────────────────────────────────────────────────────────────────
if 'lof' in methods:
    print(f'\n    [LOF]  k={lof_k}, threshold={lof_thresh}')
    lof     = LocalOutlierFactor(n_neighbors=lof_k, metric='precomputed')
    lof.fit(Dist)
    lof_raw = -lof.negative_outlier_factor_   # positive scale: higher = more outlier
    lof_flag = lof_raw > lof_thresh
    result_cols['lof_score'] = np.round(lof_raw, 4)
    flags   += lof_flag.astype(int)
    print(f'      flagged: {lof_flag.sum()}  |  score range: [{lof_raw.min():.3f}, {lof_raw.max():.3f}]')

# ── k-NN distance ─────────────────────────────────────────────────────────────
if 'knn' in methods:
    if knn_thresh is None:
        threshold_knn = float(kd.mean() + 3.0 * kd.std())
        print(f'\n    [k-NN]  k={knn_k}, auto threshold = {threshold_knn:.4f} {dist_unit}  (mean + 3σ)')
    else:
        threshold_knn = knn_thresh
        print(f'\n    [k-NN]  k={knn_k}, threshold = {threshold_knn} {dist_unit}')

    knn_flag = kd > threshold_knn
    result_cols['knn_dist'] = np.round(kd, 4)
    flags   += knn_flag.astype(int)
    print(f'      flagged: {knn_flag.sum()}  |  distance range: [{kd.min():.3f}, {kd.max():.3f}] {dist_unit}')

# ── DBSCAN ────────────────────────────────────────────────────────────────────
if 'dbscan' in methods:
    eps_use = eps_auto if dbscan_eps is None else dbscan_eps
    eps_src = 'auto (95th pct)' if dbscan_eps is None else 'manual'
    print(f'\n    [DBSCAN]  eps={eps_use:.4f} {dist_unit} ({eps_src}), minpts={dbscan_minpts}')

    db        = DBSCAN(eps=eps_use, min_samples=dbscan_minpts, metric='precomputed')
    db_labels = db.fit_predict(Dist)
    db_flag   = db_labels == -1
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    result_cols['dbscan_label'] = db_labels
    flags    += db_flag.astype(int)
    print(f'      flagged: {db_flag.sum()}  |  clusters found: {n_clusters}')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — CONSENSUS
# ═════════════════════════════════════════════════════════════════════════════

is_outlier = flags >= vote_thresh

print(f'\n## Consensus outliers  (≥ {vote_thresh} method(s) agree): '
      f'{is_outlier.sum()} / {N}  ({100*is_outlier.mean():.1f} %)')

outlier_idx = np.where(is_outlier)[0]
if len(outlier_idx) > 0:
    print(f'\n   {"Index":>6}  {"Votes":^5}  Coordinates')
    print('   ' + '-' * 40)
    for idx in outlier_idx:
        cx = f'{coords[idx, 0]:{coord_fmt}}'
        cy = f'{coords[idx, 1]:{coord_fmt}}'
        print(f'   {idx:6d}  {flags[idx]:^5}  ({cx}, {cy})')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

plot_outliers(coords, is_outlier, flags,
              crs_type=crs_type,
              title='Spatial Outlier Detection',
              save_path=output_plot)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — EXPORT CSV
# ═════════════════════════════════════════════════════════════════════════════

if output_csv:
    import os
    df_out = df_input.copy()
    for col, vals in result_cols.items():
        df_out[col] = vals
    df_out['outlier_votes'] = flags
    df_out['is_outlier']    = is_outlier
    df_out.index.name = 'point_index'
    df_out.to_csv(output_csv)
    print(f'\n## Results saved to: {os.path.abspath(output_csv)}')
