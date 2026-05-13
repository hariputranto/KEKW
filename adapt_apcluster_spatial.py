"""
Adaptive Affinity Propagation — Spatial Clustering (Projected Coordinates)
===========================================================================
Copy of adapt_apcluster_script.py, optimised for geographic data in a
projected CRS (e.g. UTM WGS84).  Coordinates are easting/northing in metres.

Changes from script.py and why:
  1. Non-squared Euclidean distance (default) preserves the true linear
     geometry of projected space.  script.py used squared-Euclidean, which
     distorts relative distances and biases cluster boundaries.
     Use sim_type='haversine' only when coordinates are decimal-degree lat/lon.
  2. convits = 50 (script.py: 10).  Longer patience before declaring
     convergence avoids locking onto transient stable states.
  3. Initial preference = pmedian (script.py: pmedian × 0.5).  Starting
     at the median gives a richer initial cluster landscape to prune from.
  4. Kvar escape step = 2·std (script.py: 2·√std).  Bolder step when
     escaping oscillation, better suited to metric distance scales.
  5. cut = 1 (script.py: 3).  Isolated geographic points are valid centres.
  6. Silhouette evaluation uses the same distance matrix as clustering so
     the quality score is consistent with the metric.
  7. Visualisation uses set_aspect('equal') on easting/northing axes so
     cluster shapes are spatially faithful (1 m on x == 1 m on y).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_samples
from sklearn.metrics.cluster import contingency_matrix as _sklearn_contingency


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def haversine_matrix(coords):
    """
    Great-circle distance matrix from (N, 2) array of [longitude, latitude] in decimal degrees.
    coords[:, 0] = longitude (x), coords[:, 1] = latitude (y).
    Returns (N, N) distances in kilometres.
    """
    lon = np.radians(coords[:, 0])   # x column → longitude
    lat = np.radians(coords[:, 1])   # y column → latitude
    dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
    dlon = lon[:, np.newaxis] - lon[np.newaxis, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :])
         * np.sin(dlon / 2) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def similarity_euclid_linear(data):
    """
    Non-squared Euclidean distance matrix. (N, d) → (N, N).
    Uses linear distance to preserve true spatial geometry.
    """
    from scipy.spatial.distance import cdist
    return cdist(data, data, metric='euclidean')


def ind2cluster(labels):
    """Convert flat label vector → list of index arrays (1-based sequential)."""
    labels = np.asarray(labels, dtype=int)
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    new_labels = inv + 1
    clusters   = [np.where(inv == i)[0] for i in range(len(counts))]
    return clusters, new_labels, counts


def _grow(K, N, labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out):
    """Expand output arrays when a new K exceeds current capacity."""
    if K <= labels_out.shape[1]:
        return labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out
    extra       = K - labels_out.shape[1]
    labels_out  = np.hstack([labels_out,  np.zeros((N, extra), dtype=int)])
    labelid_out = np.hstack([labelid_out, np.zeros((N, extra), dtype=int)])
    NC_out      = np.concatenate([NC_out,   np.zeros(extra, dtype=int)])
    NCfix_out   = np.concatenate([NCfix_out, np.zeros(extra, dtype=int)])
    Sp_out      = np.concatenate([Sp_out,   np.zeros(extra)])
    Slam_out    = np.concatenate([Slam_out,  np.zeros(extra)])
    return labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out


def _update_pref(Sprefer, astep, S):
    """Increase preference by astep and update S diagonal in-place."""
    Sprefer = float(np.atleast_1d(Sprefer)[0]) + astep
    np.fill_diagonal(S, Sprefer)
    return Sprefer


def solution_evaluation(dist_matrix, labels, NC, NCfix, cut):
    """
    Score every discovered K with the Silhouette index.
    dist_matrix must be a precomputed (N, N) distance matrix (e.g. Haversine).
    Removes any K whose smallest cluster has fewer than `cut` points.
    """
    NC    = np.array(NC,    dtype=int)
    NCfix = np.array(NCfix, dtype=int)
    Sil       = []
    Silmin    = []
    Sildelete = []

    for i in range(len(NC)):
        Y = labels[:, i]
        n_clusters = len(np.unique(Y))
        N          = len(Y)

        if n_clusters < 2 or n_clusters == N:
            sil_vals = np.zeros(N)
        else:
            sil_vals = silhouette_samples(dist_matrix, Y, metric='precomputed')

        finite_mask = np.isfinite(sil_vals)
        Sil.append(float(np.mean(sil_vals[finite_mask])))

        clusters, _, lengths = ind2cluster(Y)
        Sildelete.append(int(lengths.min()) < cut)
        Q = [
            float(np.mean(sil_vals[idx][np.isfinite(sil_vals[idx])])) if np.isfinite(sil_vals[idx]).any() else np.nan
            for idx in clusters
        ]
        Silmin.append(float(np.nanmin(Q)))

    Sil    = np.array(Sil)
    Silmin = np.array(Silmin)

    del_idx = np.where(Sildelete)[0]
    if len(del_idx) < len(NC):
        Sil    = np.delete(Sil,    del_idx)
        Silmin = np.delete(Silmin, del_idx)
        NC     = np.delete(NC,     del_idx)
        NCfix  = np.delete(NCfix,  del_idx)

    return NC, Sil, Silmin, NCfix


def valid_external(index1, c2):
    """
    External validation vs true labels.
    Returns (4, n) array — rows are [Rand, AdjRand, Jaccard, FM].
    """
    index1 = np.atleast_2d(np.asarray(index1, dtype=float))
    if index1.shape[0] == 1:
        index1 = index1.T
    c2 = np.asarray(c2, dtype=float)

    if not (np.array_equal(index1, index1.astype(int)) and
            np.array_equal(c2, c2.astype(int))):
        return np.array([])

    c2   = c2.astype(int)
    Outs = []

    for i in range(index1.shape[1]):
        c1 = index1[:, i].astype(int)
        n  = len(c1)

        C     = _sklearn_contingency(c1, c2)
        ns    = n * (n - 1) / 2.0
        nis   = float(np.sum(C.sum(axis=1) ** 2))
        njs   = float(np.sum(C.sum(axis=0) ** 2))
        sumC  = float(np.sum(C ** 2))
        sumij = nis + njs

        R_val = ns + sumC - sumij * 0.5
        nc    = (n * (n**2 + 1) - (n + 1)*nis - (n + 1)*njs + 2*(nis*njs)/n) / (2*(n - 1))
        AR    = 0.0 if ns == nc else (R_val - nc) / (ns - nc)
        Rand  = R_val / ns
        Jac   = (sumC - n) / (sumij - sumC - n) if (sumij - sumC - n) != 0 else 0.0

        ni_v  = C.sum(axis=1).astype(float);  ni_v = ni_v * (ni_v - 1) / 2.0
        nj_v  = C.sum(axis=0).astype(float);  nj_v = nj_v * (nj_v - 1) / 2.0
        denom = np.sqrt(ni_v.sum() * nj_v.sum())
        FM    = 0.5 * (sumC - n) / denom if denom > 0 else 0.0

        Outs.append([Rand, AR, Jac, FM])

    return np.array(Outs).T


def valid_errorate(labels, truelabels):
    """Per-cluster and overall misclassification rate vs true labels."""
    nrow = len(truelabels)
    _, truelabels_new, _ = ind2cluster(np.asarray(truelabels, dtype=int))

    sort_idx          = np.argsort(truelabels_new)
    truelabels_sorted = truelabels_new[sort_idx]
    labels_sorted     = np.asarray(labels, dtype=int)[sort_idx]

    clusters, _, _ = ind2cluster(labels_sorted)

    Sm = []
    for clust_idx in clusters:
        Sk = np.sort(clust_idx)
        m  = len(Sk)
        n  = max(round(np.sqrt(np.sqrt(m)) + 0.1), 1)
        Sm.append(float(np.mean(Sk[n - 1: m - n + 1])) if m > 0 else 0.0)

    order = np.argsort(Sm)
    S_arr = np.ones(nrow, dtype=int)

    for new_id, orig_idx in enumerate(order, start=1):
        Sk        = clusters[orig_idx]
        S_arr[Sk] = new_id
        Q         = S_arr[Sk] - truelabels_sorted[Sk]
        print(f'  Error rate of cluster {new_id}: {100.0 * (Q != 0).sum() / len(Sk):.2f}%')

    diff   = (S_arr - truelabels_sorted) != 0
    Rerror = 100.0 * diff.sum() / nrow
    print(f'  Error rate for all data: {Rerror:.2f}%')
    return Rerror


def load_spatial_data(filepath, x_col, y_col):
    """
    Load a CSV and return ((N, 2) coordinate array, full DataFrame).
    x_col / y_col: column name (str) or positional index (int).
    The full DataFrame is kept so the output CSV can preserve all original columns.
    """
    import pandas as pd
    df = pd.read_csv(filepath)

    x = df.iloc[:, x_col].values.astype(float) if isinstance(x_col, int) else df[x_col].values.astype(float)
    y = df.iloc[:, y_col].values.astype(float) if isinstance(y_col, int) else df[y_col].values.astype(float)

    return np.column_stack([x, y]), df


def plot_clusters_geo(coords, labels, labelid, crs_type='projected',
                      title='Spatial AP Clustering', save_path=''):
    """
    Cluster scatter plot for either coordinate type.

    coords   : (N, 2) — [easting, northing] for projected; [longitude, latitude] for geographic
    labels   : (N,)  1-based cluster labels
    labelid  : (N,)  1-based exemplar index for each point
    crs_type : 'projected'  → equal aspect (1 m = 1 m), Easting/Northing labels
               'geographic' → aspect corrected for mid-latitude distortion, Lon/Lat labels
    """
    x = coords[:, 0]   # easting  (projected)  or  longitude  (geographic)
    y = coords[:, 1]   # northing (projected)  or  latitude   (geographic)

    K = int(labels.max())
    if K <= 10:
        palette = [plt.get_cmap('tab10')(i) for i in range(K)]
    elif K <= 20:
        palette = [plt.get_cmap('tab20')(i) for i in range(K)]
    else:
        palette = [plt.get_cmap('hsv')(i / K) for i in range(K)]

    _, ax = plt.subplots(figsize=(10, 8))

    for k in range(1, K + 1):
        color   = palette[k - 1]
        mask    = labels == k
        ctr_idx = int(labelid[mask][0]) - 1   # 0-based

        cx, cy = x[ctr_idx], y[ctr_idx]

        for xi, yi in zip(x[mask], y[mask]):
            ax.plot([xi, cx], [yi, cy], color=color, lw=0.5, alpha=0.35, zorder=1)

        ax.scatter(x[mask], y[mask], c=[color], s=45, zorder=2)
        ax.scatter(cx, cy, c=[color], s=260, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=3)

    if crs_type == 'geographic':
        # Correct for the fact that 1° longitude ≠ 1° latitude in real space
        mid_lat = float(np.median(y))
        aspect  = 1.0 / np.cos(np.radians(mid_lat))
        ax.set_aspect(aspect if np.isfinite(aspect) and aspect > 0 else 'equal')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
    else:
        # Equal aspect: 1 metre on x == 1 metre on y (correct for any projected metric CRS)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

    legend_handles = [
        Line2D([0], [0], marker='o', color='0.4', linestyle='None',
               markersize=7, label='Member'),
        Line2D([0], [0], marker='*', color='0.4', linestyle='None',
               markersize=13, markeredgecolor='black', label='Centre (exemplar)'),
    ]
    if K <= 12:
        for k in range(1, K + 1):
            legend_handles.append(
                Line2D([0], [0], marker='o', color=palette[k - 1],
                       linestyle='None', markersize=7, label=f'Cluster {k}')
            )
    ax.legend(handles=legend_handles, loc='best', fontsize=8, ncol=2 if K > 6 else 1)

    ax.set_title(f'{title}  (K={K})')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\n## Plot saved to: {save_path}')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

data_file   = 'Tourism_coord.csv'   # path to input CSV

# ── Column mapping ────────────────────────────────────────────────────────────
x_col       = 'x'   # column for x  →  easting   (projected)  |  longitude  (geographic)
y_col       = 'y'   # column for y  →  northing  (projected)  |  latitude   (geographic)

# ── Coordinate Reference System ───────────────────────────────────────────────
#   'projected'  → coordinates are in metres (e.g. UTM); uses Euclidean distance
#   'geographic' → coordinates are decimal-degree lon/lat (WGS84); uses Haversine distance
crs_type    = 'projected'

#   UTM-specific: only relevant when crs_type = 'projected'
utm_zone    = 49    # zone number 1–60
utm_hemi    = 'S'   # hemisphere: 'N' or 'S'

maxits      = 2000
convits     = 50                # convergence window — 50 vs script.py's 10
lam         = 0.7               # damping factor [0.5, 1.0)
folds       = 0.01              # preference step factor
cut         = 1                 # min cluster size; 1 = allow singletons
truelabels  = None              # (N,) true labels for validation, or None
output_csv  = 'ap_spatial_results.csv'
output_plot = 'ap_spatial_clusters.png'

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA & BUILD SIMILARITY MATRIX
# ═════════════════════════════════════════════════════════════════════════════

print(f'==> Loading spatial data from {data_file} ...')
coords, df_input = load_spatial_data(data_file, x_col, y_col)
N = coords.shape[0]

if crs_type == 'projected':
    sim_type               = 'euclidean'
    crs_label              = f'Projected — UTM Zone {utm_zone}{utm_hemi} (metres)'
    x_label, y_label, unit = 'Easting', 'Northing', 'm'
    coord_fmt              = '.2f'
else:
    sim_type               = 'haversine'
    crs_label              = 'Geographic — decimal-degree lon/lat (WGS84)'
    x_label, y_label, unit = 'Longitude', 'Latitude', '°'
    coord_fmt              = '.6f'

print(f'    {N} points  |  CRS: {crs_label}')
print(f'    {x_label} [{coords[:,0].min():{coord_fmt}}, {coords[:,0].max():{coord_fmt}}] {unit}  '
      f'{y_label} [{coords[:,1].min():{coord_fmt}}, {coords[:,1].max():{coord_fmt}}] {unit}')

print(f'\n==> Computing {sim_type} distance matrix ...')
if sim_type == 'haversine':
    Dist = haversine_matrix(coords)
    print(f'    Distance range: [{Dist.min():.3f}, {Dist.max():.3f}] km')
else:
    Dist = similarity_euclid_linear(coords)
    print(f'    Distance range: [{Dist.min():.3f}, {Dist.max():.3f}] m')

# Similarity = negative distance (linear, not squared — preserves geometry)
S = -Dist

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — ADAPTIVE AFFINITY PROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

print('\n==> Running Adaptive AP ...')

# Preference: start at pmedian (not pmedian × 0.5 as in script.py)
off_diag = S[~np.eye(N, dtype=bool)]
pmedian  = float(np.median(off_diag))
pstep_base = folds * pmedian
Sprefer    = pmedian          # <-- key change: full median, not half
pfixed     = False

if lam > 0.9:
    print('\n*** Warning: Large damping factor. Consider increasing convits.\n')

if N > 3000:
    print('\n*** Warning: Large memory request. Consider a sparse approach.\n')

# Break degeneracies with a reproducible tiny noise
rng = np.random.get_state()
np.random.seed(0)
S  += (np.finfo(float).eps * S + np.finfo(float).tiny * 100) * np.random.rand(N, N)
np.random.set_state(rng)

np.fill_diagonal(S, Sprefer)

# Message matrices
dS = np.diag(S).copy()
A  = np.zeros((N, N))
R  = np.zeros((N, N))

# State / history buffers
stoptimes = convits if pfixed else max(maxits // 10, 2000)

Hstop     = np.zeros((N, stoptimes))
Hconvits  = np.zeros((N, convits))
nhalf     = max(1, round(0.3 * convits))
Hconvhalf = np.zeros((N, nhalf))

Tdelay    = 10
Hdelay    = Tdelay
Hdelay2   = Tdelay
Hconverg  = False
Hsavehalf = False
Hn1 = 0;  Hn2 = 0

Wstart = max(100, round(convits / 2))
wsize  = 40
Kocil  = np.ones(wsize, dtype=bool)   # 1-D circular oscillation buffer
Noscil = wsize + 10
Svib   = 0
Hvib   = 10;  Tvib = 2.0
Hguid  = 1

pstep = pstep_base
astep = pstep

buf       = maxits + 20
Kset      = np.zeros(buf, dtype=int)
Kmean     = np.zeros(buf)
Kdown     = np.zeros(buf, dtype=bool)
Kunchange = np.zeros(buf)

Kold = 0;  Kfix = 0;  nKfix = 0
Kmax = 0;  nfix = 0
unconverged = False

_cap        = min(N, 512)
labels_out  = np.zeros((N, _cap), dtype=int)
labelid_out = np.zeros((N, _cap), dtype=int)
NC_out      = np.zeros(_cap, dtype=int)
NCfix_out   = np.zeros(_cap, dtype=int)
Sp_out      = np.zeros(_cap)
Slam_out    = np.zeros(_cap)

_FMAX  = np.finfo(float).max
_FEPS  = np.finfo(float).eps
_FTINY = np.finfo(float).tiny

dn = False
it = 0

while not dn:
    it += 1

    # ── Responsibilities  R(i,k) = S(i,k) - max_{j≠k}[A(i,j)+S(i,j)] ────────
    AS    = A + S
    Y     = AS.max(axis=1)
    I_max = AS.argmax(axis=1)
    AS[np.arange(N), I_max] = -_FMAX
    Y2    = AS.max(axis=1)
    AS    = None
    Rold  = R
    R     = S - Y[:, np.newaxis]
    R[np.arange(N), I_max] = S[np.arange(N), I_max] - Y2
    R     = (1 - lam) * R + lam * Rold
    Rold  = None

    # ── Availabilities  A(i,k) = min(0, R(k,k) + Σ_{j≠i,k} max(0,R(j,k))) ──
    Rp = np.maximum(R, 0.0)
    Rp[np.arange(N), np.arange(N)] = R[np.arange(N), np.arange(N)]
    Aold = A
    A    = Rp.sum(axis=0)[np.newaxis, :] - Rp
    Rp   = None
    dA   = np.diag(A).copy()
    np.minimum(A, 0.0, out=A)
    A[np.arange(N), np.arange(N)] = dA
    A    = (1 - lam) * A + lam * Aold
    Aold = None

    # Exemplar decision: point k is an exemplar when A(k,k)+R(k,k) > 0
    E    = (np.einsum('ii->i', A) + np.einsum('ii->i', R)) > 0
    Hconvits[:,  (it - 1) % convits]   = E
    Hstop[:,     (it - 1) % stoptimes] = E
    Hconvhalf[:, (it - 1) % nhalf]     = E

    K        = int(E.sum())
    Kset[it] = K
    newp     = float(Sprefer)
    newlam   = lam

    if it % 100 == 1 or it == maxits:
        print(f'** running at iteration {it}, K = {K}')

    Hsave = Hsave1 = Hsave2 = Hsave3 = False

    # ── Convergence checks ────────────────────────────────────────────────────
    if it >= Wstart or it >= maxits:
        se          = Hconvits.sum(axis=1)
        unconverged = int((se == convits).sum() + (se == 0).sum()) != N
        Hconverg    = not unconverged

        se  = Hstop.sum(axis=1)
        se1 = int((se == stoptimes).sum())
        se2 = int((se == 0).sum())
        if (se1 + se2) == N or it == maxits:
            dn     = True
            Hsave1 = (se1 + se2) == N

        se  = Hconvhalf.sum(axis=1)
        Hsavehalf = (int((se == nhalf).sum()) + int((se == 0).sum()) == N) and (Hguid == 2)

    # ── Adaptive mechanisms ───────────────────────────────────────────────────
    if it > 5:
        Kmean[it]     = Kset[it - 5:it + 1].mean()
        Kdown[it]     = (Kmean[it] - Kmean[it - 1]) < 0
        if Hguid == 2:
            Kdown[it] = Kdown[it] and (K <= Kold)
        Kunchange[it] = int(np.abs(Kset[it] - Kset[it - 5:it]).sum())
        Kocil[(it - 1) % wsize] = Kdown[it] or (Kunchange[it] == 0)
        Noscil = int(Kocil.sum())

    if Hconverg:
        Hdelay += 1
        if Hdelay >= Tdelay:
            Hsave1   = True
            Hdelay   = 0
            Hn1     += 1
            nKfix    = (nKfix + 1) if K == Kfix else 0
            Kfix     = K
            stepfold = np.sqrt(K + 50) / 10.0
            pstep    = folds * pmedian / stepfold
            astep    = nKfix * pstep if nKfix > 1 else pstep
    elif Hsavehalf:
        Hdelay2 += 1
        if Hdelay2 >= Tdelay:
            Hsave2   = True
            Hdelay2  = 0
            Hn2     += 1

    if not Hconverg:
        Hn1 = 0;  Hdelay = 0
    if not Hsavehalf:
        Hn2 = 0;  Hdelay2 = 0

    if (K == 1 or K == 2) and Hsave1:
        dn = True;  unconverged = False

    if Hguid == 1 and Hsave1:
        Hguid    = 2
        Kmax     = K
        stepfold = np.sqrt(Kmax + 50) / 10.0
        pstep    = folds * pmedian / stepfold

    if Hsave1:
        Svib = 0
        if not pfixed:
            Sprefer = _update_pref(Sprefer, astep, S)
    else:
        Svib += 1
        HSvib = ((Svib > wsize and Noscil < 0.66 * wsize) or Svib > 150) and it > Wstart
        if HSvib:
            Hvib += 1
            if Hvib > 10:
                lam = max(0.7, lam)
            else:
                if Tvib >= 3:
                    if lam >= 0.9:
                        lam = min(0.98, 0.025 + lam)
                        if lam >= 0.95 and it % 9 == 2:
                            rng = np.random.get_state()
                            np.random.seed(0)
                            S  += (_FEPS * S + _FTINY * 1000) * np.random.rand(N, N)
                            np.random.set_state(rng)
                            print(' # A small amount of noise is added')
                else:
                    lam = min(0.9, 0.05 + lam)

                if lam >= 0.85:
                    Tvib += 1
                    if not pfixed:
                        if Hguid == 2 and Kold:
                            sf   = 2.0 if Kmax < 1 else max(3.0 / (np.sqrt(Kmax) / 10 + 0.4), 1.0)
                            # 2·std (no sqrt) for bolder escape on spatial scales
                            Kvar = 2.0 * float(np.std(Kset[max(1, it - 49):it + 1]))
                            astep = min(0.8 * Kvar + 0.2 * Tvib, sf) * pstep
                        else:
                            astep = min(Tvib, 2.0) * pstep
                        Sprefer = _update_pref(Sprefer, astep, S)
                        print(' # Escaping oscillation turns on')

            Hvib = 0;  Svib = 0
            print(f' # Damping factor is increased to {lam:.4g}')
        else:
            Tvib = max(Tvib - 0.002, 0.98)
            if lam > 0.9 and Tvib < 1:
                lam = max(lam - 0.0001, 0.5)

    if Hguid >= 2 and it > 1 and ((K < Kold and K > 1) or Kmean[it] == Kmean[it - 1]):
        Hsave3 = True
    Hsave = Hsave1 or Hsave2 or Hsave3
    Kold  = K

    # ── Record solution ───────────────────────────────────────────────────────
    if Hsave or dn:
        if K == 0:
            continue

        I_ex = np.where(E)[0]
        c    = S[:, I_ex].argmax(axis=1)
        c[I_ex] = np.arange(K)

        if Hsave1:
            nfix = Hn1 * Tdelay + convits
        elif Hsave2:
            nfix = Hn2 * Tdelay + nhalf
        elif it > 1 and Kmean[it] == Kmean[it - 1]:
            nfix = 10 if (it > 5 and Kmean[it] == Kmean[max(0, it - 5)]) else 6
        else:
            nfix = 1

        labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out = \
            _grow(K, N, labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out)

        ki = K - 1
        if (K <= Kmax and nfix > NCfix_out[ki]) or (K > Kmax and nfix >= 10):
            NCfix_out[ki]      = nfix
            labels_out[:,  ki] = c + 1
            labelid_out[:, ki] = I_ex[c] + 1
            NC_out[ki]         = K
            Sp_out[ki]         = newp
            Slam_out[ki]       = newlam
        if K > Kmax:
            Kmax = K

# ── Final exemplar refinement ─────────────────────────────────────────────────
print(f' # Programs run over at K= {K}')
I_final = np.where((np.einsum('ii->i', A) + np.einsum('ii->i', R)) > 0)[0]
K_final = len(I_final)

if K_final > 0:
    c_f = S[:, I_final].argmax(axis=1)
    c_f[I_final] = np.arange(K_final)
    for k in range(K_final):
        ii         = np.where(c_f == k)[0]
        j_b        = int(S[np.ix_(ii, ii)].sum(axis=0).argmax())
        I_final[k] = ii[j_b]
    c_f = S[:, I_final].argmax(axis=1)
    c_f[I_final] = np.arange(K_final)
    tmpidx = I_final[c_f]

    labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out = \
        _grow(K_final, N, labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out)
    ki = K_final - 1
    labels_out[:,  ki] = c_f + 1
    labelid_out[:, ki] = tmpidx + 1
    NC_out[ki]    = K_final
    NCfix_out[ki] = nfix
    Sp_out[ki]    = newp
    Slam_out[ki]  = newlam
else:
    tmpidx = np.full(N, -1, dtype=int)

# ── Trim to valid K values ────────────────────────────────────────────────────
if len(NC_out) > 1:
    NC_out[0] = 0

valid = np.where(NC_out)[0]
if len(valid) == 0:
    labels  = np.ones((N, 1), dtype=int)
    labelid = np.ones((N, 1), dtype=int)
    NC      = np.array([0], dtype=int)
    NCfix   = np.array([0], dtype=int)
else:
    labels  = labels_out[:,  valid]
    labelid = labelid_out[:, valid]
    NC      = NC_out[valid]
    NCfix   = NCfix_out[valid]

if unconverged and len(NC) > 0:
    print(f'\n*** Warning: Algorithm did not converge at K = {NC[0]} !')
    print('    Consider increasing maxits and if necessary lam.\n')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — SOLUTION EVALUATION (Silhouette on Haversine distances)
# ═════════════════════════════════════════════════════════════════════════════

NC, Sil, Silmin, NCfix = solution_evaluation(Dist, labels, NC, NCfix, cut)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — FIND OPTIMAL K
# ═════════════════════════════════════════════════════════════════════════════

Sid   = int(np.argmax(Sil))
Smax  = float(Sil[Sid])
NCopt = int(NC[Sid])

print('\n## Spatial AP Clustering result:')
print(f'  Optimal K = {NCopt},  Silhouette = {Smax:.4f}')

if Smax < 0.3 and len(NC) > 1:
    upper_half = np.arange(len(NC) // 2, len(NC))
    Sid2       = upper_half[int(np.argmax(Silmin[upper_half]))]
    NCopt2     = int(NC[Sid2])
    print(f'  Silhouette is low — alternative K = {NCopt2}  (min sil = {Silmin[Sid2]:.4f})')

print('\n## Silhouette by K:')
print('  NC    :', NC)
print('  Sil   :', np.round(Sil,    4))
print('  Silmin:', np.round(Silmin, 4))

optimal_labels  = labels[:,  Sid]
optimal_labelid = labelid[:, Sid]
print(f'\n  Optimal labels (K={NCopt}): {optimal_labels}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALIZATION (geographic lon/lat frame)
# ═════════════════════════════════════════════════════════════════════════════

plot_clusters_geo(coords, optimal_labels, optimal_labelid,
                  crs_type=crs_type,
                  title='Spatial AP Clustering', save_path=output_plot)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — EXPORT CSV
# ═════════════════════════════════════════════════════════════════════════════

if output_csv:
    import os
    import pandas as pd

    is_centre    = (optimal_labelid - 1) == np.arange(N)
    centre_index = optimal_labelid

    df_out = df_input.copy()
    df_out['cluster']      = optimal_labels
    df_out['is_centre']    = is_centre
    df_out['centre_index'] = centre_index
    df_out.index.name = 'point_index'
    df_out.to_csv(output_csv)
    print(f'\n## Results saved to: {os.path.abspath(output_csv)}')
