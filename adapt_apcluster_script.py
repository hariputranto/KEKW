import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.metrics.cluster import contingency_matrix as _sklearn_contingency


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def similarity_euclid(data, squared=False):
    """Pairwise Euclidean (or squared) distance between rows. (N,d) -> (N,N)"""
    R = cdist(data, data, metric='sqeuclidean')
    if squared:
        return R, float(R.max())
    R = np.sqrt(R)
    return R, float(R.max())


def similarity_pearson(data):
    """Pearson correlation between rows. (N,d) -> (N,N)"""
    R = np.corrcoef(data)
    np.fill_diagonal(R, 1.0)
    return R


def ind2cluster(labels):
    """
    Convert flat label vector to list of index arrays, renumbered 1..K.

    Returns
    -------
    clusters   : list of arrays, each holding the point indices for one cluster
    new_labels : relabelled 1..K sequentially
    lengths    : cluster sizes
    """
    labels = np.asarray(labels, dtype=int)
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    new_labels = inv + 1  # 1-based sequential
    clusters   = [np.where(inv == i)[0] for i in range(len(counts))]
    return clusters, new_labels, counts


def _grow(K, N, labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out):
    """Expand output arrays when a newly encountered K exceeds current capacity."""
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
    """Increase preference by astep and update diagonal of S in-place."""
    Sprefer = np.atleast_1d(Sprefer) + astep
    Sprefer = float(Sprefer[0]) if Sprefer.size == 1 else Sprefer
    np.fill_diagonal(S, Sprefer)
    return Sprefer


def solution_evaluation(data, M, labels, NC, NCfix, simatrix, nrow, dtype, cut):
    """
    Score every discovered clustering K using the Silhouette index.
    Removes any K whose smallest cluster has fewer than `cut` points.

    Parameters
    ----------
    data     : (N, d) raw data, or None when simatrix=True
    M        : (M, 3) pre-computed similarity triplets [i, j, s], or None
    labels   : (N, n_valid) 1-based cluster labels, one column per valid K
    NC       : (n_valid,) cluster counts
    NCfix    : (n_valid,) confidence scores
    simatrix : bool — True when input was a pre-computed similarity matrix
    nrow     : int, number of data points
    dtype    : 'euclidean' | 'correlation'
    cut      : int, minimum cluster size threshold

    Returns
    -------
    NC, Sil, Silmin, NCfix  (after removing under-size Ks)
    """
    if simatrix:
        Ms = np.full((nrow, nrow), np.nan)
        np.fill_diagonal(Ms, 0.0)
        ni = M[:, 0].astype(int) - 1
        nj = M[:, 1].astype(int) - 1
        Ms[ni, nj] = -M[:, 2]
        dim_min = float(Ms[ni, nj].min()) if len(ni) > 0 else 0.0
        if dim_min < 0:
            Ms[ni, nj] -= dim_min - 1.0
            np.fill_diagonal(Ms, 0.0)

    # Precompute once — correlation distance matrix does not depend on K
    if not simatrix and dtype == 'correlation':
        D_corr = 1.0 - (1.0 + similarity_pearson(data)) / 2.0

    NC    = np.array(NC,    dtype=int)
    NCfix = np.array(NCfix, dtype=int)
    Sil       = []
    Silmin    = []
    Sildelete = []

    for i in range(len(NC)):
        Y = labels[:, i]

        if simatrix:
            sil_vals = silhouette_samples(Ms, Y, metric='precomputed')
        elif dtype == 'correlation':
            sil_vals = silhouette_samples(D_corr, Y, metric='precomputed')
        else:
            sil_vals = silhouette_samples(data, Y, metric='euclidean')

        finite_mask = np.isfinite(sil_vals)
        Sil.append(float(np.mean(sil_vals[finite_mask])))

        clusters, _, lengths = ind2cluster(Y)
        Sildelete.append(int(lengths.min()) < cut)
        Q = [float(np.mean(sil_vals[idx][np.isfinite(sil_vals[idx])])) if np.isfinite(sil_vals[idx]).any() else np.nan
             for idx in clusters]
        Silmin.append(float(np.nanmin(Q)))

    Sil    = np.array(Sil)
    Silmin = np.array(Silmin)

    del_idx = np.where(Sildelete)[0]
    if len(del_idx) < len(NC):          # keep at least one solution
        Sil    = np.delete(Sil,    del_idx)
        Silmin = np.delete(Silmin, del_idx)
        NC     = np.delete(NC,     del_idx)
        NCfix  = np.delete(NCfix,  del_idx)

    return NC, Sil, Silmin, NCfix


def valid_external(index1, c2):
    """
    External clustering validation vs true labels.

    Parameters
    ----------
    index1 : (N,) or (N, n) predicted labels (1-based int)
    c2     : (N,) true labels (1-based int)

    Returns
    -------
    Outs : (4, n) array — rows are [Rand, AdjRand, Jaccard, FM]
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

    return np.array(Outs).T   # (4, n)


def valid_errorate(labels, truelabels):
    """
    Per-cluster and overall misclassification rate vs true labels.
    Returns overall error rate in percent.
    """
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
    S     = np.ones(nrow, dtype=int)

    for new_id, orig_idx in enumerate(order, start=1):
        Sk    = clusters[orig_idx]
        S[Sk] = new_id
        Q     = S[Sk] - truelabels_sorted[Sk]
        print(f'  Error rate of cluster {new_id}: {100.0 * (Q != 0).sum() / len(Sk):.2f}%')

    diff   = (S - truelabels_sorted) != 0
    Rerror = 100.0 * diff.sum() / nrow
    print(f'  Error rate for all data: {Rerror:.2f}%')
    return Rerror


def load_data(filepath):
    """
    Load a data file and return an (N, d) float ndarray.

    CSV files (.csv)
        Header rows and non-numeric columns (e.g. ID, class label) are
        detected and dropped automatically.  Requires pandas.

    Other files (.txt, etc.)
        Loaded with numpy.loadtxt (whitespace-delimited, no header).

    Parameters
    ----------
    filepath : str — path to the file
    """
    import os
    if os.path.splitext(filepath)[1].lower() == '.csv':
        import pandas as pd
        df = pd.read_csv(filepath, header=None)
        try:
            return df.to_numpy(dtype=float)
        except (ValueError, TypeError):
            # First row is a header, or file has non-numeric columns
            df = pd.read_csv(filepath)
            df = df.select_dtypes(include='number')
            if df.empty:
                raise ValueError(f'No numeric columns found in {filepath!r}')
            return df.to_numpy(dtype=float)
    return np.loadtxt(filepath)


def plot_clusters(data, labels, labelid, title='Adaptive AP Clustering'):
    """
    Scatter plot of a clustering result.

    Points are coloured by cluster.  Each cluster centre (exemplar) is drawn
    as a large star with a black edge.  Thin lines connect every member to its
    centre.  Data with more than 2 features is projected to 2-D via PCA before
    plotting.

    Parameters
    ----------
    data    : (N, d) raw data matrix
    labels  : (N,)  1-based integer cluster labels
    labelid : (N,)  1-based exemplar index for each point
    title   : str, figure title prefix
    """
    N, d = data.shape

    if d > 2:
        coords    = PCA(n_components=2).fit_transform(data)
        ax_labels = ('PC 1', 'PC 2')
        pca_note  = '  [PCA projection]'
    elif d == 2:
        coords    = data.astype(float)
        ax_labels = ('Feature 1', 'Feature 2')
        pca_note  = ''
    else:
        coords    = np.column_stack([data[:, 0].astype(float), np.zeros(N)])
        ax_labels = ('Feature 1', '')
        pca_note  = ''

    K = int(labels.max())
    if K <= 10:
        palette = [plt.get_cmap('tab10')(i) for i in range(K)]
    elif K <= 20:
        palette = [plt.get_cmap('tab20')(i) for i in range(K)]
    else:
        palette = [plt.get_cmap('hsv')(i / K) for i in range(K)]

    _, ax = plt.subplots(figsize=(8, 6))

    for k in range(1, K + 1):
        color   = palette[k - 1]
        mask    = labels == k
        ctr_idx = int(labelid[mask][0]) - 1   # 0-based exemplar index for this cluster

        cx, cy = coords[ctr_idx]

        # thin lines: each member → its centre
        for xi, yi in coords[mask]:
            ax.plot([xi, cx], [yi, cy], color=color, lw=0.5, alpha=0.35, zorder=1)

        # member points
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=45, zorder=2)

        # centre / exemplar marker (star, same colour, black edge)
        ax.scatter(cx, cy, c=[color], s=260, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=3)

    # Legend — marker-type key always shown; per-cluster colour swatches when K is small
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
    ax.legend(handles=legend_handles, loc='best', fontsize=8,
              ncol=2 if K > 6 else 1)

    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_title(f'{title}  (K={K}{pca_note})')
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

data        = load_data('wine.txt')    # supports .csv (auto-detects header) or whitespace .txt
dtype       = 'euclidean'              # 'euclidean' | 'correlation'
pvalues     = None                     # None = median similarity × 0.5; or float/array
folds       = 0.01                     # preference step factor
adapt       = 0                        # 0 = adaptive AP (default);  1 = original/searching AP
maxits      = 5000                     # max iterations
convits     = 50                       # convergence window
lam         = 0.5                      # damping factor [0.5, 1.0)
plot        = False                    # print iteration info
details     = False                    # record per-iteration netsim/dpsim/expref
nonoise     = False                    # skip noise addition
cut         = 3                        # minimum cluster size (drop smaller)
simatrix    = False                    # True if 'data' is a (M,3) similarity matrix
M           = None                     # pre-computed similarity triplets (if simatrix=True)
truelabels  = None                     # (N,) true labels for validation, or None

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — ADAPTIVE AFFINITY PROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

_raw_data = data   # preserved for solution_evaluation; `data` may be set to None below

# 0 → adaptive (adapt<2 computes distances from raw data)
# 1 → original/searching AP (adapt>=2 expects pre-computed similarity as input)
adapt = adapt + 1

# ── Build similarity triplets ─────────────────────────────────────────────────
if adapt < 2:
    if dtype in ('euclidean', 1):
        Dist, _ = similarity_euclid(data, squared=True)
    else:
        Dist = 1.0 - (1.0 + similarity_pearson(data)) / 2.0

    nrow     = Dist.shape[0]
    r, c     = np.where(~np.eye(nrow, dtype=bool))
    s        = np.column_stack([r + 1, c + 1, -Dist[r, c]]).astype(float)
    Dist     = None
else:
    s    = np.asarray(data, dtype=float).copy()
    data = None

# ── Preference setup ──────────────────────────────────────────────────────────
pfixed      = False
valid_mask  = s[:, 2] > -np.finfo(float).max
pmedian     = float(np.median(s[valid_mask, 2]))
pstep_base  = folds * pmedian

pvalues_arr = np.atleast_1d(pvalues if pvalues is not None else pmedian * 0.5).astype(float)
if pvalues is not None:
    pfixed = True

if lam > 0.9:
    print('\n*** Warning: Large damping factor in use. Consider increasing convits.\n')

# ── Build NxN similarity matrix S ─────────────────────────────────────────────
if s.shape[1] == 3:
    tmp = int(max(s[:, 0].max(), s[:, 1].max()))
    N   = tmp if len(pvalues_arr) == 1 else len(pvalues_arr)
    if tmp > N:
        raise ValueError('data point index exceeds number of data points')
    if min(s[:, 0].min(), s[:, 1].min()) <= 0:
        raise ValueError('data point indices must be >= 1')
    S    = np.full((N, N), -np.inf)
    rows = s[:, 0].astype(int) - 1
    cols = s[:, 1].astype(int) - 1
    S[rows, cols] = s[:, 2]
elif s.ndim == 2 and s.shape[0] == s.shape[1]:
    N = s.shape[0]
    if len(pvalues_arr) not in (1, N):
        raise ValueError('pvalues must be scalar or length N')
    S = s.copy()
else:
    raise ValueError('s must have 3 columns or be square')

s    = None
nrow = N

if N > 3000:
    print('\n*** Warning: Large memory request. Consider a sparse approach.\n')

if not nonoise:
    rng = np.random.get_state()
    np.random.seed(0)
    S  += (np.finfo(float).eps * S + np.finfo(float).tiny * 100) * np.random.rand(N, N)
    np.random.set_state(rng)

if len(pvalues_arr) == 1:
    np.fill_diagonal(S, float(pvalues_arr[0]))
else:
    np.fill_diagonal(S, pvalues_arr)

# ── Allocate message matrices and history buffers ─────────────────────────────
dS = np.diag(S).copy()
A  = np.zeros((N, N))
R  = np.zeros((N, N))

netsim_hist = dpsim_hist = expref_hist = idx_hist = None
if plot or details:
    netsim_hist = np.full(maxits + 2, np.nan)
if details:
    dpsim_hist  = np.full(maxits + 2, np.nan)
    expref_hist = np.full(maxits + 2, np.nan)
    idx_hist    = np.full((N, maxits + 2), -1, dtype=int)

# ── State variables ───────────────────────────────────────────────────────────
dn          = False
it          = 0
stoptimes   = convits if pfixed else max(maxits // 10, 2000)

Hstop       = np.zeros((N, stoptimes))
Hconvits    = np.zeros((N, convits))
nhalf       = max(1, round(0.3 * convits))
Hconvhalf   = np.zeros((N, nhalf))

Tdelay      = 10
Hdelay      = Tdelay
Hdelay2     = Tdelay
Hconverg    = False
Hsavehalf   = False
Hn1 = 0;    Hn2 = 0

Wstart      = max(100, round(convits / 2))
wsize       = 40
Kocil       = np.ones(wsize, dtype=bool)
Noscil      = wsize + 10
Svib        = 0
Hvib        = 10;   Tvib = 2.0
Hguid       = 1

Sprefer     = float(pvalues_arr[0]) if len(pvalues_arr) == 1 else pvalues_arr.copy()
pstep       = pstep_base
astep       = pstep

buf         = maxits + 20
Kset        = np.zeros(buf, dtype=int)
Kmean       = np.zeros(buf)
Kdown       = np.zeros(buf, dtype=bool)
Kunchange   = np.zeros(buf)

Kold = 0;   Kfix = 0;  nKfix = 0
Kmax = 0;   nfix = 0
unconverged = False

tmpnetsim = tmpdpsim = tmpexpref = np.nan
tmpidx    = np.full(N, -1, dtype=int)

_cap        = min(N, 512)
labels_out  = np.zeros((N, _cap), dtype=int)
labelid_out = np.zeros((N, _cap), dtype=int)
NC_out      = np.zeros(_cap, dtype=int)
NCfix_out   = np.zeros(_cap, dtype=int)
Sp_out      = np.zeros(_cap)
Slam_out    = np.zeros(_cap)

# Cache float-info constants — avoids repeated object construction inside the hot loop
_FMAX  = np.finfo(float).max
_FEPS  = np.finfo(float).eps
_FTINY = np.finfo(float).tiny

# ── Main message-passing loop ─────────────────────────────────────────────────
while not dn:
    it += 1

    # Responsibilities  R(i,k) = S(i,k) - max_{j≠k}[A(i,j) + S(i,j)]
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

    # Availabilities  A(i,k) = min(0, R(k,k) + Σ_{j≠i,k} max(0,R(j,k)))
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

    # einsum('ii->i', ...) returns a view of the diagonal — no allocation
    E    = (np.einsum('ii->i', A) + np.einsum('ii->i', R)) > 0
    Hconvits[:,  (it - 1) % convits]   = E
    Hstop[:,     (it - 1) % stoptimes] = E
    Hconvhalf[:, (it - 1) % nhalf]     = E

    K        = int(E.sum())
    Kset[it] = K
    newp     = float(np.atleast_1d(Sprefer)[0])
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
        se1 = int((se == nhalf).sum())
        se2 = int((se == 0).sum())
        Hsavehalf = ((se1 + se2) == N) and (Hguid == 2)

    # ── Adaptive mechanisms ───────────────────────────────────────────────────
    # MATLAB's `if adapt` is always True after the +1 increment (adapt ≥ 1).
    if adapt >= 1:
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
                                sf    = 2.0 if Kmax < 1 else max(3.0 / (np.sqrt(Kmax) / 10 + 0.4), 1.0)
                                Kvar  = 2.0 * float(np.sqrt(np.std(Kset[max(1, it - 49):it + 1], ddof=1)))
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

    # ── Record / evaluate solution ────────────────────────────────────────────
    if plot or details or Hsave or dn:
        if K == 0:
            tmpnetsim = tmpdpsim = tmpexpref = np.nan
            tmpidx    = np.full(N, -1, dtype=int)
        else:
            I_ex = np.where(E)[0]
            c    = S[:, I_ex].argmax(axis=1)
            c[I_ex] = np.arange(K)

            if Hsave or dn:
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
            else:
                tmpidx    = I_ex[c]
                tmpnetsim = float(S[np.arange(N), tmpidx].sum())
                tmpexpref = float(dS[I_ex].sum())
                tmpdpsim  = tmpnetsim - tmpexpref

        if details:
            netsim_hist[it]  = tmpnetsim
            dpsim_hist[it]   = tmpdpsim
            expref_hist[it]  = tmpexpref
            if not np.any(tmpidx == -1):
                idx_hist[:, it] = tmpidx
        elif plot:
            netsim_hist[it] = tmpnetsim

# ── Final refinement: re-select best exemplar per cluster ─────────────────────
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
    tmpidx    = I_final[c_f]
    tmpnetsim = float(S[np.arange(N), tmpidx].sum())
    tmpexpref = float(dS[I_final].sum())
    tmpdpsim  = tmpnetsim - tmpexpref

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
    tmpnetsim = tmpdpsim = tmpexpref = np.nan
    tmpidx    = np.full(N, -1, dtype=int)

# ── Package outputs ───────────────────────────────────────────────────────────
if details:
    netsim_hist[it + 1]  = tmpnetsim
    dpsim_hist[it + 1]   = tmpdpsim
    expref_hist[it + 1]  = tmpexpref
    if not np.any(tmpidx == -1):
        idx_hist[:, it + 1] = tmpidx
    netsim_out = netsim_hist[:it + 2]
    dpsim_out  = dpsim_hist[:it + 2]
    expref_out = expref_hist[:it + 2]
    idx_out    = idx_hist[:, :it + 2]
else:
    netsim_out = tmpnetsim
    dpsim_out  = tmpdpsim
    expref_out = tmpexpref
    idx_out    = tmpidx

# ── Trim to valid K values ────────────────────────────────────────────────────
if len(NC_out) > 1:
    NC_out[0] = 0

valid = np.where(NC_out)[0]
if len(valid) == 0:
    labels  = np.ones((N, 1), dtype=int)
    labelid = np.ones((N, 1), dtype=int)
    NC      = np.array([0], dtype=int)
    NCfix   = np.array([0], dtype=int)
    Sp      = np.array([])
    Slam    = np.array([])
else:
    labels  = labels_out[:,  valid]
    labelid = labelid_out[:, valid]
    NC      = NC_out[valid]
    NCfix   = NCfix_out[valid]
    Sp      = Sp_out[valid]
    Slam    = Slam_out[valid]

if unconverged and len(NC) > 0:
    print(f'\n*** Warning: Algorithm did not converge at K = {NC[0]} !')
    print('    Consider increasing maxits and if necessary dampfact.\n')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — SOLUTION EVALUATION (Silhouette index for each valid K)
# ═════════════════════════════════════════════════════════════════════════════

NC, Sil, Silmin, NCfix = solution_evaluation(
    data     = _raw_data,
    M        = M,
    labels   = labels,
    NC       = NC,
    NCfix    = NCfix,
    simatrix = simatrix,
    nrow     = nrow,
    dtype    = dtype,
    cut      = cut,
)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — FIND OPTIMAL K
# ═════════════════════════════════════════════════════════════════════════════

Sid   = int(np.argmax(Sil))
Smax  = float(Sil[Sid])
NCopt = int(NC[Sid])

print('\n## Clustering solution by Adaptive Affinity Propagation:')
print(f'  Optimal number of clusters: {NCopt},  Silhouette = {Smax:.4g}')

if Smax < 0.3:
    upper_half = np.arange(len(NC) // 2, len(NC))
    Tmax_idx   = int(np.argmax(Silmin[upper_half]))
    Sid2       = upper_half[Tmax_idx]
    NCopt2     = int(NC[Sid2])
    print(f'  Silhouette is small — alternative NC = {NCopt2}')
    print(f'  where min Silhouette of single cluster = {Silmin[Sid2]:.4g}')

print('\n## Silhouette values at different NCs:')
print('  NC    :', NC)
print('  Sil   :', np.round(Sil,    4))
print('  Silmin:', np.round(Silmin, 4))

optimal_labels = labels[:, Sid]
print(f'\n  Optimal labels (K={NCopt}): {optimal_labels}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — EXTERNAL VALIDATION (only runs when truelabels are provided)
# ═════════════════════════════════════════════════════════════════════════════

if truelabels is not None:
    truelabels = np.asarray(truelabels, dtype=int)

    print('\n## External validation indices [Rand, AdjRand, Jaccard, FM]:')
    ext = valid_external(labels, truelabels)
    if ext.size > 0:
        for row, name in zip(ext, ['Rand', 'AdjRand', 'Jaccard', 'FM']):
            print(f'  {name:8s}: {"  ".join(f"{v:.4f}" for v in row)}')
        print(f'  (columns correspond to NC = {NC})')

    print(f'\n## Error rate (optimal K={NCopt}):')
    valid_errorate(optimal_labels, truelabels)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
# Skipped when input was a pre-computed similarity matrix (no raw coordinates).

if not simatrix:
    plot_clusters(_raw_data, optimal_labels, labelid[:, Sid])
