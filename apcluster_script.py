import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
    new_labels = inv + 1
    clusters   = [np.where(inv == i)[0] for i in range(len(counts))]
    return clusters, new_labels, counts


def valid_external(index1, c2):
    """
    External clustering validation vs true labels.
    Returns Outs : (4,) array — [Rand, AdjRand, Jaccard, FM].
    """
    c1 = np.asarray(index1, dtype=int)
    c2 = np.asarray(c2,     dtype=int)
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

    return np.array([Rand, AR, Jac, FM])


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

    CSV files have headers and non-numeric columns auto-detected and dropped.
    Other files (.txt) are loaded with numpy.loadtxt (whitespace-delimited).
    """
    import os
    if os.path.splitext(filepath)[1].lower() == '.csv':
        import pandas as pd
        df = pd.read_csv(filepath, header=None)
        try:
            return df.to_numpy(dtype=float)
        except (ValueError, TypeError):
            df = pd.read_csv(filepath)
            df = df.select_dtypes(include='number')
            if df.empty:
                raise ValueError(f'No numeric columns found in {filepath!r}')
            return df.to_numpy(dtype=float)
    return np.loadtxt(filepath)


def plot_clusters(data, labels, labelid, title='Affinity Propagation Clustering',
                  save_path=''):
    """
    Scatter plot of a clustering result. Members coloured by cluster, exemplars
    drawn as black-edged stars, thin lines link members to their exemplar.
    Data with more than 2 features is projected to 2-D via PCA.
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
        ctr_idx = int(labelid[mask][0]) - 1

        cx, cy = coords[ctr_idx]

        for xi, yi in coords[mask]:
            ax.plot([xi, cx], [yi, cy], color=color, lw=0.5, alpha=0.35, zorder=1)

        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=45, zorder=2)
        ax.scatter(cx, cy, c=[color], s=260, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=3)

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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\n## Plot saved to: {save_path}')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

data        = load_data('wine.txt')    # supports .csv (auto-detects header) or whitespace .txt
dtype       = 'euclidean'              # 'euclidean' | 'correlation'
preference  = None                     # None = median similarity; scalar or (N,) array
maxits      = 500                      # max iterations
convits     = 50                       # convergence window
lam         = 0.5                      # damping factor [0.5, 1.0)
nonoise     = False                    # skip noise addition
simatrix    = False                    # True if 'data' is (M,3) triplets or (N,N) similarity
truelabels  = None                     # (N,) true labels for validation, or None
output_csv  = 'ap_results.csv'         # path for result CSV; '' to skip
output_plot = 'ap_clusters.png'        # path for cluster plot; '' to skip

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD SIMILARITY MATRIX
# ═════════════════════════════════════════════════════════════════════════════

_raw_data = data

if not simatrix:
    if dtype in ('euclidean', 1):
        Dist, _ = similarity_euclid(data, squared=True)
    else:
        Dist = 1.0 - (1.0 + similarity_pearson(data)) / 2.0

    nrow = Dist.shape[0]
    r, c = np.where(~np.eye(nrow, dtype=bool))
    s    = np.column_stack([r + 1, c + 1, -Dist[r, c]]).astype(float)
    Dist = None
else:
    s    = np.asarray(data, dtype=float).copy()
    data = None

# ── Pack triplets / square matrix into N×N S ──────────────────────────────────
if s.shape[1] == 3:
    tmp = int(max(s[:, 0].max(), s[:, 1].max()))
    N   = tmp
    if min(s[:, 0].min(), s[:, 1].min()) <= 0:
        raise ValueError('data point indices must be >= 1')
    S    = np.full((N, N), -np.inf)
    rows = s[:, 0].astype(int) - 1
    cols = s[:, 1].astype(int) - 1
    S[rows, cols] = s[:, 2]
elif s.ndim == 2 and s.shape[0] == s.shape[1]:
    N = s.shape[0]
    S = s.copy()
else:
    raise ValueError('s must have 3 columns or be square')

s    = None
nrow = N

if N > 3000:
    print('\n*** Warning: Large memory request. Consider a sparse approach.\n')

if lam > 0.9:
    print('\n*** Warning: Large damping factor in use. Consider increasing convits.\n')

# ── Default preference = median of off-diagonal similarities ──────────────────
if preference is None:
    valid_mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(valid_mask, False)
    pmedian    = float(np.median(S[valid_mask & np.isfinite(S)]))
    preference = pmedian
    print(f'## Preference set to median similarity: {preference:.6g}')

p_arr = np.atleast_1d(preference).astype(float)
if len(p_arr) not in (1, N):
    raise ValueError('preference must be scalar or length N')

# ── Add noise to break degeneracies ───────────────────────────────────────────
if not nonoise:
    rng = np.random.get_state()
    np.random.seed(0)
    S  += (np.finfo(float).eps * S + np.finfo(float).tiny * 100) * np.random.rand(N, N)
    np.random.set_state(rng)

# ── Place preferences on the diagonal ─────────────────────────────────────────
if len(p_arr) == 1:
    np.fill_diagonal(S, float(p_arr[0]))
else:
    np.fill_diagonal(S, p_arr)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — MESSAGE-PASSING LOOP
# ═════════════════════════════════════════════════════════════════════════════

dS = np.diag(S).copy()
A  = np.zeros((N, N))
R  = np.zeros((N, N))
e  = np.zeros((N, convits))

_FMAX = np.finfo(float).max

dn = False
it = 0
unconverged = True

while not dn:
    it += 1

    # Responsibilities  R(i,k) = S(i,k) - max_{j≠k}[A(i,j) + S(i,j)]
    Rold = R
    AS   = A + S
    Y    = AS.max(axis=1)
    I    = AS.argmax(axis=1)
    AS[np.arange(N), I] = -_FMAX
    Y2   = AS.max(axis=1)
    R    = S - Y[:, np.newaxis]
    R[np.arange(N), I] = S[np.arange(N), I] - Y2
    R    = (1 - lam) * R + lam * Rold

    # Availabilities  A(i,k) = min(0, R(k,k) + Σ_{j≠i,k} max(0, R(j,k)))
    Aold = A
    Rp   = np.maximum(R, 0.0)
    Rp[np.arange(N), np.arange(N)] = R[np.arange(N), np.arange(N)]
    A    = Rp.sum(axis=0)[np.newaxis, :] - Rp
    dA   = np.diag(A).copy()
    np.minimum(A, 0.0, out=A)
    A[np.arange(N), np.arange(N)] = dA
    A    = (1 - lam) * A + lam * Aold

    # Convergence check
    E = (np.diag(A) + np.diag(R)) > 0
    e[:, (it - 1) % convits] = E
    K = int(E.sum())

    if it >= convits or it >= maxits:
        se          = e.sum(axis=1)
        unconverged = int(((se == convits) + (se == 0)).sum()) != N
        if (not unconverged and K > 0) or it == maxits:
            dn = True

    if it % 100 == 1 or it == maxits:
        print(f'** running at iteration {it}, K = {K}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — IDENTIFY AND REFINE EXEMPLARS
# ═════════════════════════════════════════════════════════════════════════════

I = np.where((np.diag(A) + np.diag(R)) > 0)[0]
K = len(I)

if K > 0:
    # Initial cluster assignment by maximum similarity to candidate exemplars
    c       = S[:, I].argmax(axis=1)
    c[I]    = np.arange(K)

    # Refine: pick the within-cluster point with the highest summed similarity
    for k in range(K):
        ii   = np.where(c == k)[0]
        j_b  = int(S[np.ix_(ii, ii)].sum(axis=0).argmax())
        I[k] = ii[j_b]

    c       = S[:, I].argmax(axis=1)
    c[I]    = np.arange(K)
    idx     = I[c]                          # 0-based exemplar index for each point
    netsim  = float(S[np.arange(N), idx].sum())
    expref  = float(dS[I].sum())
    dpsim   = netsim - expref

    optimal_labels  = c + 1                 # 1-based cluster id
    optimal_centres = idx + 1               # 1-based exemplar index per point
else:
    idx             = np.full(N, -1, dtype=int)
    netsim = dpsim  = expref = float('nan')
    optimal_labels  = np.ones(N, dtype=int)
    optimal_centres = np.ones(N, dtype=int)

print('\n## Clustering solution by Affinity Propagation:')
print(f'  Number of identified clusters: {K}')
print(f'  Fitness (net similarity)     : {netsim:.6g}')
print(f'    Similarities to exemplars  : {dpsim:.6g}')
print(f'    Preferences of exemplars   : {expref:.6g}')
print(f'  Iterations                   : {it}')

if unconverged:
    print('\n*** Warning: Algorithm did not converge.')
    print('    Consider increasing maxits and/or dampfact.\n')

# ── Silhouette index as a quality indicator ───────────────────────────────────
if K > 1 and not simatrix:
    if dtype == 'correlation':
        D_corr = 1.0 - (1.0 + similarity_pearson(_raw_data)) / 2.0
        sil    = silhouette_score(D_corr, optimal_labels, metric='precomputed')
    else:
        sil    = silhouette_score(_raw_data, optimal_labels, metric='euclidean')
    print(f'  Silhouette index             : {sil:.4g}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — EXTERNAL VALIDATION (only when truelabels are provided)
# ═════════════════════════════════════════════════════════════════════════════

if truelabels is not None:
    truelabels = np.asarray(truelabels, dtype=int)

    print('\n## External validation indices [Rand, AdjRand, Jaccard, FM]:')
    ext = valid_external(optimal_labels, truelabels)
    for name, v in zip(['Rand', 'AdjRand', 'Jaccard', 'FM'], ext):
        print(f'  {name:8s}: {v:.4f}')

    print(f'\n## Error rate (K={K}):')
    valid_errorate(optimal_labels, truelabels)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
# Skipped when input was a pre-computed similarity matrix (no raw coordinates).

if not simatrix and K > 0 and output_plot:
    plot_clusters(_raw_data, optimal_labels, optimal_centres, save_path=output_plot)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — EXPORT RESULTS TO CSV
# ═════════════════════════════════════════════════════════════════════════════

if output_csv:
    import os, pandas as pd

    is_centre = (optimal_centres - 1) == np.arange(nrow)

    if not simatrix:
        n_feat = _raw_data.shape[1]
        df_out = pd.DataFrame(_raw_data,
                              columns=[f'feature_{i + 1}' for i in range(n_feat)])
    else:
        df_out = pd.DataFrame()

    df_out['cluster']      = optimal_labels
    df_out['is_centre']    = is_centre
    df_out['centre_index'] = optimal_centres

    df_out.index.name = 'point_index'
    df_out.to_csv(output_csv)
    print(f'\n## Results saved to: {os.path.abspath(output_csv)}')
