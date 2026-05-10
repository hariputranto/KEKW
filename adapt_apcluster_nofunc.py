"""
Adaptive Affinity Propagation Clustering - Sequential Script
No functions, runs top to bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# CONFIGURATION - CHANGE THESE
# ============================================================================

DATA_FILE = 'table_attr_calc.csv'       # Your CSV data file
HAS_LABELS = False               # True if first column contains class labels
LABEL_COLUMN = 0                # Column index for labels (0 = first column)
DATA_COLUMNS = [1,2]             # Which columns to use as features (None = all except labels)
SIM_TYPE = 1                    # 1 = Euclidean, 2 = Pearson
MAXITS = 2000                   # Maximum iterations
CONVITS = 50                    # Convergence iterations
DAMPFACT = 0.7                  # Damping factor (0.5 to <1)
PSTEP_FOLD = 0.01              # Preference step size
CUT = 1                         # Minimum cluster size
PLOT = True                    # Enable plotting


# ============================================================================
# PART 1: LOAD DATA
# ============================================================================

print(f'==> Loading data from {DATA_FILE} ...')

# Load CSV file using pandas
df = pd.read_csv(DATA_FILE)

if HAS_LABELS:
    # Extract labels from specified column
    true_labels = df.iloc[:, LABEL_COLUMN].values
    
    # Extract features
    if DATA_COLUMNS is not None:
        data = df.iloc[:, DATA_COLUMNS].values
    else:
        # Use all columns except the label column
        cols = [i for i in range(df.shape[1]) if i != LABEL_COLUMN]
        data = df.iloc[:, cols].values
else:
    # No labels
    true_labels = None
    if DATA_COLUMNS is not None:
        data = df.iloc[:, DATA_COLUMNS].values
    else:
        data = df.values

n_samples, n_features = data.shape
print(f'    Data shape: {data.shape}')

if true_labels is not None:
    print(f'    True clusters: {len(np.unique(true_labels))}')


# ============================================================================
# PART 2: COMPUTE SIMILARITY MATRIX
# ============================================================================

print('\n==> Computing similarity matrix ...')

n = data.shape[0]

if SIM_TYPE == 1:
    # Euclidean distance
    dist = np.zeros((n, n))
    dmax = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[i] - data[j]
            d = np.sqrt(np.dot(diff, diff))
            dist[i, j] = d
            dist[j, i] = d
            if d > dmax:
                dmax = d
else:
    # Pearson correlation distance
    from scipy.stats import pearsonr
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corr, _ = pearsonr(data[i], data[j])
            dist[i, j] = 0.5 - 0.5 * corr
            dist[j, i] = dist[i, j]
    dmax = 1.0

# Convert to similarity (negative distance)
S = -dist

print(f'    Similarity matrix: {S.shape}')
print(f'    Distance range: [{dist.min():.3f}, {dist.max():.3f}]')


# ============================================================================
# PART 3: ADAPTIVE AP CLUSTERING
# ============================================================================

print('\n==> Running Adaptive AP ...')

# --- Initialize parameters ---
pmedian = np.median(S[S > -np.inf])
pstep = PSTEP_FOLD * pmedian
Sprefer = pmedian
pfixed = 0

# Add noise to break degeneracies
rng_state = np.random.get_state()
np.random.seed(0)
S = S + (np.finfo(float).eps * S + np.finfo(float).tiny * 100) * np.random.rand(n, n)
np.random.set_state(rng_state)

# Set preferences on diagonal
np.fill_diagonal(S, Sprefer)

# Initialize message matrices
dS = np.diag(S).copy()
A = np.zeros((n, n))
R = np.zeros((n, n))

# Adaptive parameters
lam = DAMPFACT
stoptimes = max(int(MAXITS / 10), 2000)
if pfixed:
    stoptimes = CONVITS

Hstop = np.zeros((n, stoptimes))
Hconvits = np.zeros((n, CONVITS))
Tdelay = 10
Hdelay = Tdelay
Hconverg = 0
nhalf = round(0.3 * CONVITS)
Hdelay2 = Tdelay
Hconvhalf = np.zeros((n, nhalf))
Hsavehalf = 0
Hn1 = 0
Hn2 = 0

Wstart = max(100, round(CONVITS / 2))
wsize = 40
Kunchange = np.zeros(MAXITS + 1)
Kocil = np.ones((n, wsize))
Noscil = wsize + 10
Svib = 0
Hvib = 10
Tvib = 2
Hguid = 1

astep = pstep
Kset = np.zeros(MAXITS + 1)
Kold = 0
Kfix = 0
nKfix = 0
Kmax = 0
nfix = 0

# Storage for solutions
labels_list = []
labelid_list = []
NC_list = []
NCfix_list = []
Sp_list = []
Slam_list = []
stored_K = set()

dn = False
i = 0

# --- Main AP loop ---
while not dn:
    i += 1

    # === Compute Responsibilities ===
    AS = A + S
    Y = np.max(AS, axis=1)
    I = np.argmax(AS, axis=1)
    AS[np.arange(n), I] = -np.finfo(float).max
    Y2 = np.max(AS, axis=1)

    Rold = R.copy()
    R = S - Y[:, np.newaxis]
    R[np.arange(n), I] = S[np.arange(n), I] - Y2
    R = (1 - lam) * R + lam * Rold  # Damping

    # === Compute Availabilities ===
    Rp = np.maximum(R, 0)
    np.fill_diagonal(Rp, np.diag(R))

    Aold = A.copy()
    A = np.sum(Rp, axis=0)[np.newaxis, :] - Rp
    dA = np.diag(A)
    A = np.minimum(A, 0)
    np.fill_diagonal(A, dA)
    A = (1 - lam) * A + lam * Aold  # Damping

    # === Check Convergence ===
    E = (np.diag(A) + np.diag(R)) > 0
    Hconvits[:, (i - 1) % CONVITS] = E
    K = int(np.sum(E))
    Kset[i] = K

    newp = Sprefer
    newlam = lam

    Hstop[:, (i - 1) % stoptimes] = E
    Hconvhalf[:, (i - 1) % nhalf] = E

    if i % 100 == 1 or i == MAXITS:
        print(f'** running at iteration {i}, K = {K}')

    Hsave = Hsave1 = Hsave2 = Hsave3 = 0

    if i >= Wstart or i >= MAXITS:
        se = np.sum(Hconvits, axis=1)
        se1 = np.sum(se == CONVITS)
        se2 = np.sum(se == 0)
        unconverged = (se1 + se2) != n
        Hconverg = int(not unconverged)

        se = np.sum(Hstop, axis=1)
        se1 = np.sum(se == stoptimes)
        se2 = np.sum(se == 0)

        if (se1 + se2) == n or i == MAXITS:
            dn = True
            if (se1 + se2) == n:
                Hsave1 = 1

        se = np.sum(Hconvhalf, axis=1)
        se1 = np.sum(se == nhalf)
        se2 = np.sum(se == 0)
        Hsavehalf = int((se1 + se2) == n) and Hguid == 2

    # === Adaptive Logic ===
    if i > 5:
        Kmean_val = np.mean(Kset[max(0, i - 5):i + 1])
        Kdown_val = Kmean_val - np.mean(Kset[max(0, i - 6):i]) < 0 if i > 6 else False

        if Hguid == 2:
            Kdown_val = Kdown_val and K <= Kold

        Kunchange[i] = np.sum(np.abs(Kset[i] - Kset[max(0, i - 5):i]))
        Kocil[:, (i - 1) % wsize] = np.where(Kdown_val or Kunchange[i] == 0, 1, 0)
        Noscil = np.sum(Kocil, axis=0)

    # Reduce p to yield smaller NC when unchanging
    if Hconverg:
        Hdelay += 1
        if Hdelay >= Tdelay:
            Hsave1 = 1
            Hdelay = 0
            Hn1 += 1

            if K == Kfix:
                nKfix += 1
            else:
                nKfix = 0

            Kfix = K
            stepfold = np.sqrt(K + 50) / 10
            pstep = PSTEP_FOLD * pmedian / stepfold

            if nKfix > 1:
                astep = nKfix * pstep
            else:
                astep = pstep
    elif Hsavehalf:
        Hdelay2 += 1
        if Hdelay2 >= Tdelay:
            Hsave2 = 1
            Hdelay2 = 0
            Hn2 += 1

    if not Hconverg:
        Hn1 = 0
        Hdelay = 0
    if not Hsavehalf:
        Hn2 = 0
        Hdelay2 = 0

    if ((K == 1 or K == 2) and Hsave1):
        dn = True
        Hconverg = 1

    if Hguid == 1 and Hsave1:
        Hguid = 2
        Kmax = K
        stepfold = np.sqrt(Kmax + 50) / 10
        pstep = PSTEP_FOLD * pmedian / stepfold

    if Hsave1:
        Svib = 0
        if not pfixed:
            Sprefer += astep
            np.fill_diagonal(S, Sprefer)
    else:
        Svib += 1
        HSvib = (Svib > wsize and np.mean(Noscil) < 0.66 * wsize) or Svib > 150
        HSvib = HSvib and i > Wstart

        if HSvib:
            Hvib += 1
            if Hvib > 10:
                lam = max(0.7, lam)
            elif Hvib >= 1:
                if Tvib >= 3:
                    if lam >= 0.9:
                        lam = min(0.98, 0.025 + lam)
                        if lam >= 0.95 and i % 9 == 2:
                            rng_state = np.random.get_state()
                            np.random.seed(0)
                            S = S + (np.finfo(float).eps * S + np.finfo(float).tiny * 1000) * np.random.rand(n, n)
                            np.random.set_state(rng_state)
                            print(' # A small amount of noise is added')
                else:
                    lam = min(0.9, 0.05 + lam)

                if lam >= 0.85:
                    Tvib += 1
                    if not pfixed:
                        if Hguid == 2 and Kold:
                            if Kmax < 1:
                                stepfold = 2
                            else:
                                stepfold = max(3 / (np.sqrt(Kmax) / 10 + 0.4), 1)
                            Kvar = 2 * np.std(Kset[max(0, i - 49):i + 1])
                            astep = min(0.8 * Kvar + 0.2 * Tvib, stepfold) * pstep
                        else:
                            astep = min(Tvib, 2) * pstep

                        Sprefer += astep
                        np.fill_diagonal(S, Sprefer)
                        print(' # Escaping oscillation turns on')

            Hvib = 0
            Svib = 0
            print(f' # Damping factor is increased to {lam}')
        else:
            Tvib = max(Tvib - 0.002, 0.98)
            if lam > 0.9 and Tvib < 1:
                lam = max(lam - 0.0001, 0.5)

    if Hguid >= 2 and (K < Kold and K > 1 or (i > 5 and Kset[i] == Kset[i - 1])):
        Hsave3 = 1

    if Hsave1 or Hsave2 or Hsave3:
        Hsave = 1

    Kold = K

    # === Save Solutions ===
    if Hsave or dn:
        if K == 0:
            continue

        I_exemplars = np.where(E)[0]
        if len(I_exemplars) == 0:
            continue

        S_exemplars = S[:, I_exemplars]
        c = np.argmax(S_exemplars, axis=1)

        if Hsave1:
            nfix = Hn1 * Tdelay + CONVITS
        elif Hsave2:
            nfix = Hn2 * Tdelay + nhalf
        elif i > 5 and Kset[i] == Kset[i - 1]:
            nfix = 6
            if i > 10 and Kset[i] == Kset[i - 5]:
                nfix = 10
        else:
            nfix = 1

        # Store solution
        if K not in stored_K:
            if K > len(labels_list):
                for _ in range(K - len(labels_list)):
                    labels_list.append(None)
                    labelid_list.append(None)
                    NC_list.append(0)
                    NCfix_list.append(0)
                    Sp_list.append(0)
                    Slam_list.append(0)

            labels_list[K-1] = c.copy()
            labelid_list[K-1] = I_exemplars[c].copy()
            NC_list[K-1] = K
            NCfix_list[K-1] = nfix
            Sp_list[K-1] = newp
            Slam_list[K-1] = newlam
            stored_K.add(K)

# === Final Solution ===
I = np.where(np.diag(A + R) > 0)[0]
K_final = len(I)

print(f'\n # Programs run over at K= {K_final}')

if K_final > 0:
    S_exemplars = S[:, I]
    c = np.argmax(S_exemplars, axis=1)

    # Refine exemplars
    for k in range(K_final):
        ii = np.where(c == k)[0]
        if len(ii) > 0:
            S_sub = S[np.ix_(ii, ii)]
            y = np.sum(S_sub, axis=0)
            j = np.argmax(y)
            I[k] = ii[j]

    S_exemplars = S[:, I]
    c = np.argmax(S_exemplars, axis=1)
    c[I] = np.arange(K_final)

    # Store final solution
    labels_list.append(c.copy())
    labelid_list.append(I[c].copy())
    NC_list.append(K_final)
    NCfix_list.append(i)
    Sp_list.append(Sprefer)
    Slam_list.append(lam)

# Package results
valid_idx = [i for i, l in enumerate(labels_list) if l is not None]
results_labels = [labels_list[i] for i in valid_idx]
results_labelid = [labelid_list[i] for i in valid_idx]
results_NC = np.array([NC_list[i] for i in valid_idx])
results_NCfix = np.array([NCfix_list[i] for i in valid_idx]) if NCfix_list else np.array([])
results_Sp = np.array([Sp_list[i] for i in valid_idx])
results_Slam = np.array([Slam_list[i] for i in valid_idx])
results_iend = i

print(f'\nNumber of identified clusters: {K_final}')
print(f'Number of iterations: {i}')
print(f'Found {len(results_labels)} solutions\n')


# ============================================================================
# PART 4: EVALUATE SOLUTIONS (Silhouette Index)
# ============================================================================

print('==> Evaluating clustering solutions ...')

# Compute Silhouette for each solution
metric = 'euclidean' if SIM_TYPE == 1 else 'correlation'

Sil = []
Silmin = []
valid_labels_list = []
valid_NC_list = []

for idx_sol in range(len(results_labels)):
    labels = results_labels[idx_sol]

    # Compute Silhouette
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1 or n_clusters == n:
        silhouette = np.zeros(n)
    else:
        # Use dist computed earlier
        silhouette = np.zeros(n)
        for pt in range(n):
            own_cluster = labels[pt]
            own_mask = labels == own_cluster

            if np.sum(own_mask) > 1:
                a_i = np.mean(dist[pt, own_mask & (np.arange(n) != pt)])
            else:
                a_i = 0

            b_i = np.inf
            for label in unique_labels:
                if label != own_cluster:
                    other_mask = labels == label
                    if np.sum(other_mask) > 0:
                        mean_dist = np.mean(dist[pt, other_mask])
                        b_i = min(b_i, mean_dist)

            if b_i == np.inf:
                b_i = 0

            if max(a_i, b_i) > 0:
                silhouette[pt] = (b_i - a_i) / max(a_i, b_i)
            else:
                silhouette[pt] = 0

    valid_mask = np.isfinite(silhouette)
    sil_mean = np.mean(silhouette[valid_mask])

    # Per-cluster Silhouette
    Q = []
    for label in unique_labels:
        cluster_mask = labels == label
        if np.sum(cluster_mask) > 0:
            Q.append(np.mean(silhouette[cluster_mask]))

    sil_min = np.min(Q) if len(Q) > 0 else 0

    # Check cluster sizes
    cluster_sizes = [np.sum(labels == l) for l in unique_labels]
    if all(size >= CUT for size in cluster_sizes):
        Sil.append(sil_mean)
        Silmin.append(sil_min)
        valid_labels_list.append(labels)
        valid_NC_list.append(results_NC[idx_sol])

Sil = np.array(Sil)
Silmin = np.array(Silmin)
valid_NC_list = np.array(valid_NC_list)

print(f'    Valid solutions after filtering: {len(Sil)}')


# ============================================================================
# PART 5: FIND OPTIMAL K
# ============================================================================

print('\n==> Finding optimal number of clusters ...')

if len(Sil) > 0:
    Sid = np.argmax(Sil)
    Smax = Sil[Sid]
    NCopt = valid_NC_list[Sid]
    optimal_labels = valid_labels_list[Sid]

    print(f'\n## Optimal number of clusters: {NCopt}')
    print(f'   Silhouette = {Smax:.4f}')

    if Smax < 0.3 and len(Sil) > 1:
        R_range = range(len(Sil) // 2, len(Sil))
        if len(R_range) > 0:
            Q = np.argmax(Silmin[list(R_range)])
            Sid2 = list(R_range)[Q]
            NCopt2 = valid_NC_list[Sid2]
            print(f'   Alternative (large NCs): K={NCopt2}, min Sil={Silmin[Sid2]:.4f}')

    print(f'\n## Silhouette values at different NCs:')
    for j in range(len(Sil)):
        print(f'   NC={valid_NC_list[j]:2d}: Sil={Sil[j]:.4f}, Silmin={Silmin[j]:.4f}')
else:
    print('   No valid solutions found!')
    optimal_labels = None
    NCopt = 0
    Smax = 0


# ============================================================================
# PART 6: VALIDATION (if true labels available)
# ============================================================================

if true_labels is not None and optimal_labels is not None:
    print('\n==> Validating against true labels ...')

    # Convert to 0-based contiguous labels
    def make_contiguous(labels):
        unique = np.unique(labels)
        new_labels = labels.copy()
        for new_val, old_val in enumerate(unique):
            new_labels[labels == old_val] = new_val
        return new_labels

    pred_contiguous = make_contiguous(optimal_labels)
    true_contiguous = make_contiguous(true_labels)

    # Contingency matrix
    n1 = int(pred_contiguous.max()) + 1
    n2 = int(true_contiguous.max()) + 1
    Cont = np.zeros((n1, n2), dtype=int)
    for idx in range(len(pred_contiguous)):
        Cont[pred_contiguous[idx], true_contiguous[idx]] += 1

    n = len(pred_contiguous)
    nis = np.sum(np.sum(Cont, axis=1) ** 2)
    njs = np.sum(np.sum(Cont, axis=0) ** 2)
    ns = n * (n - 1) / 2
    sumC = np.sum(Cont ** 2)
    sumij = nis + njs
    R_val = ns + sumC - sumij * 0.5
    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))

    if ns == nc:
        AR = 0
    else:
        AR = (R_val - nc) / (ns - nc)

    Rand = R_val / ns

    if sumij - sumC - n > 0:
        Jac = (sumC - n) / (sumij - sumC - n)
    else:
        Jac = 0

    ni = np.sum(Cont, axis=1) * (np.sum(Cont, axis=1) - 1) / 2
    nj = np.sum(Cont, axis=0) * (np.sum(Cont, axis=0) - 1) / 2
    nis2 = np.sum(ni)
    njs2 = np.sum(nj)

    if nis2 * njs2 > 0:
        FM = 0.5 * (sumC - n) / np.sqrt(nis2 * njs2)
    else:
        FM = 0

    print(f'\n## Validation Metrics:')
    print(f'   Rand Index:        {Rand:.4f}')
    print(f'   Adjusted Rand:     {AR:.4f}')
    print(f'   Jaccard:           {Jac:.4f}')
    print(f'   Fowlkes-Mallows:   {FM:.4f}')

    # Error rate
    mismatches = np.sum(pred_contiguous != true_contiguous)
    error_rate = 100 * mismatches / n
    print(f'\n## Error rate: {error_rate:.2f}%')


# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'Optimal K: {NCopt}')
print(f'Best Silhouette: {Smax:.4f}')
print(f'Final preference: {Sprefer:.4f}')
print(f'Initial preference (pmedian): {pmedian:.4f}')
if true_labels is not None:
    print(f'True K: {len(np.unique(true_labels))}')
print(f'Iterations: {results_iend}')
print(f'Data: {DATA_FILE} ({n_samples} samples, {n_features} features)')
print('=' * 60)


# ============================================================================
# PART 7: VISUALIZE CLUSTERING RESULTS
# ============================================================================

if optimal_labels is not None:
    print('\n==> Generating clustering visualization ...')

    # Use raw 2D data directly
    data_2d = data[:, :2]  # Ensure only 2 columns

    # Get exemplars (cluster centers)
    unique_clusters = np.unique(optimal_labels)
    n_clusters = len(unique_clusters)

    # Generate colors
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 20)))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # === Left plot: Clustering result ===
    ax1 = axes[0]

    for cluster_idx, cluster_id in enumerate(unique_clusters):
        mask = optimal_labels == cluster_id
        ax1.scatter(data_2d[mask, 0], data_2d[mask, 1],
                   c=[colors[cluster_idx]], label=f'Cluster {cluster_id}',
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Mark exemplar (cluster center) - find point closest to cluster centroid
        cluster_points = data_2d[mask]
        centroid = np.mean(cluster_points, axis=0)
        distances_to_centroid = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
        exemplar_local_idx = np.argmin(distances_to_centroid)
        center_idx = np.where(mask)[0][exemplar_local_idx]
        ax1.scatter(data_2d[center_idx, 0], data_2d[center_idx, 1],
                   c='red', marker='X', s=200, linewidths=2,
                   edgecolors='black', zorder=5)

        # Draw lines from points to exemplar
        for pt_idx in np.where(mask)[0]:
            ax1.plot([data_2d[pt_idx, 0], data_2d[center_idx, 0]],
                    [data_2d[pt_idx, 1], data_2d[center_idx, 1]],
                    c=colors[cluster_idx], alpha=0.2, linewidth=0.5)

    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title(f'Adaptive AP Clustering (K={n_clusters})', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # === Right plot: True labels (if available) ===
    if true_labels is not None:
        ax2 = axes[1]
        true_unique = np.unique(true_labels)

        for true_idx, true_id in enumerate(true_unique):
            mask = true_labels == true_id
            ax2.scatter(data_2d[mask, 0], data_2d[mask, 1],
                       c=[colors[true_idx]], label=f'Class {true_id}',
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax2.set_xlabel('Feature 1', fontsize=12)
        ax2.set_ylabel('Feature 2', fontsize=12)
        ax2.set_title(f'True Labels (K={len(true_unique)})', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].remove()

    plt.tight_layout()

    # Save figure
    output_file = f'clustering_{DATA_FILE.split(".")[0]}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'    Figure saved: {output_file}')

    plt.show()
else:
    print('\n==> No clustering results to visualize!')


# ============================================================================
# PART 8: EXPORT RESULTS TO CSV/EXCEL
# ============================================================================

if optimal_labels is not None:
    print('\n==> Exporting clustering results ...')

    # Find exemplars (cluster centers) - one per cluster
    unique_clusters = np.unique(optimal_labels)
    exemplar_indices = np.zeros(len(unique_clusters), dtype=int)

    for cluster_idx, cluster_id in enumerate(unique_clusters):
        mask = optimal_labels == cluster_id
        cluster_points = data_2d[mask]
        centroid = np.mean(cluster_points, axis=0)
        distances_to_centroid = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
        exemplar_local_idx = np.argmin(distances_to_centroid)
        exemplar_indices[cluster_idx] = np.where(mask)[0][exemplar_local_idx]

    # Create is_exemplar column
    is_exemplar = np.zeros(n, dtype=int)
    is_exemplar[exemplar_indices] = 1

    # Build output DataFrame
    output_data = {}

    # Add original identifier if available
    if HAS_LABELS and LABEL_COLUMN == 0:
        output_data['ID'] = df.iloc[:, 0].values

    # Add original features
    for feat_idx in range(n_features):
        output_data[f'Feature_{feat_idx+1}'] = data[:, feat_idx]

    # Add cluster assignment and exemplar marker
    output_data['Cluster_Label'] = optimal_labels
    output_data['Is_Cluster_Center'] = is_exemplar

    df_output = pd.DataFrame(output_data)

    # Export to CSV
    csv_file = f'clustering_results_{DATA_FILE.split(".")[0]}.csv'
    df_output.to_csv(csv_file, index=False)
    print(f'    CSV saved: {csv_file}')

    # Export to Excel (if openpyxl is available)
    try:
        excel_file = f'clustering_results_{DATA_FILE.split(".")[0]}.xlsx'
        df_output.to_excel(excel_file, index=False)
        print(f'    Excel saved: {excel_file}')
    except Exception as e:
        print(f'    Excel export skipped (install openpyxl for Excel support): {e}')

    # Print exemplar summary
    print(f'\n    Cluster Centers (Exemplars):')
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        center_idx = exemplar_indices[cluster_idx]
        cluster_size = np.sum(optimal_labels == cluster_id)
        print(f'      Cluster {cluster_id}: Point index {center_idx}, Size = {cluster_size}')
else:
    print('\n==> No clustering results to export!')
