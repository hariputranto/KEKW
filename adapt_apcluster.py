"""Adaptive Affinity Propagation Clustering - Python translation of adapt_apcluster.m"""

import os
import numpy as np
import pandas as pd


# ============================================================================
# Config
# ============================================================================

DATA_FILE = 'wine.txt'  # CSV or TXT
DTYPE = 1              # 1=euclidean, 2=correlation
DAMPFACT = 0.5         # damping factor
MAXITS = 500
CONVITS = 50
PVALUES = None         # None = median
FOLDS = 1
ADAPT = 1              # 1=adaptive, 0=original AP
PLOT = False
DETAILS = False
NONOISE = False
CUT = 3                # drop clusters with < CUT samples
TRUE_LABELS = 0        # 0=no true labels, 1=first column is true labels


# ============================================================================
# Similarity Functions
# ============================================================================

def similarity_euclid(data, squared=False):
    """Pairwise Euclidean distances (or squared distances) between rows of data."""
    data = np.asarray(data)
    nrow = data.shape[0]
    R = np.zeros((nrow, nrow))
    dmax = 0.0

    for i in range(nrow - 1):
        x = data[i]
        for j in range(i + 1, nrow):
            y = x - data[j]
            d = np.sqrt(np.dot(y, y)) if not squared else np.dot(y, y)
            R[i, j] = d
            R[j, i] = d
            if d > dmax:
                dmax = d
    return R, dmax


def similarity_pearson(data):
    """Pearson correlation coefficients between every pair of columns."""
    data = np.asarray(data)
    nrow, ncol = data.shape
    x = np.mean(data, axis=0)
    data = data - np.tile(x, (nrow, 1))
    R = np.ones((ncol, ncol))

    for i in range(ncol - 1):
        xi = data[:, i]
        Xi = np.sqrt(np.dot(xi, xi))
        for j in range(i + 1, ncol):
            y = data[:, j]
            xy = np.dot(xi, y)
            Yi = np.sqrt(np.dot(y, y))
            sim = xy / (Xi * Yi)
            R[i, j] = sim
            R[j, i] = sim
    return R


# ============================================================================
# Data Loading
# ============================================================================

def load_data(filepath, has_true_labels):
    """Load data from CSV or TXT (space/tab/comma delimited)."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(filepath, header=None, dtype=float)
    else:
        df = pd.read_csv(filepath, sep=None, header=None, dtype=float, engine='python')

    df = df.fillna(0)

    if has_true_labels:
        true_labels = df.iloc[:, 0].astype(int).values
        data = df.iloc[:, 1:].values
    else:
        true_labels = None
        data = df.values

    return data, true_labels


# ============================================================================
# Validation
# ============================================================================

def contingency_matrix(idx, true):
    """Build contingency matrix."""
    n_pred = int(np.max(idx))
    n_true = int(np.max(true))
    C = np.zeros((n_pred, n_true), dtype=int)
    for i in range(len(idx)):
        C[int(idx[i]) - 1, int(true[i]) - 1] += 1
    return C


def fowlkes_mallows(idx, true):
    """Fowlkes-Mallows validity index."""
    C = contingency_matrix(idx, true)
    n = len(idx)
    ni = np.sum(C, axis=1)
    nj = np.sum(C, axis=0)
    nis = np.sum(ni * (ni - 1) / 2)
    njs = np.sum(nj * (nj - 1) / 2)
    nij_c = C ** 2
    sumC = np.sum(nij_c) - n
    if nis == 0 or njs == 0:
        return np.nan
    FM = 0.5 * sumC / np.sqrt(nis * njs)
    return FM


def rand_index(idx, true):
    """Rand index."""
    C = contingency_matrix(idx, true)
    n = len(idx)
    ni = np.sum(C, axis=1)
    nj = np.sum(C, axis=0)
    nis = np.sum(ni * (ni - 1) / 2)
    njs = np.sum(nj * (nj - 1) / 2)
    ns = n * (n - 1) / 2
    sumC = np.sum(C ** 2) - n
    nij_sum = nis + njs
    R = ns + sumC - nij_sum * 0.5
    return R / ns


def silhouette_score(data, labels):
    """Silhouette coefficient."""
    # Handle 2D labels array: take the last column (most recent/stable result)
    if labels.ndim > 1:
        labels = labels[:, -1]
    n = len(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    s = np.zeros(n)

    for i in range(n):
        label_i = labels[i]
        if label_i == 0:
            s[i] = 0
            continue

        in_cluster = np.where(labels == label_i)[0]
        if len(in_cluster) > 1:
            distances = np.sqrt(np.sum((data[i] - data[in_cluster]) ** 2, axis=1))
            a_i = np.mean(distances)
        else:
            a_i = 0

        b_i = np.inf
        for lbl in unique_labels:
            if lbl != label_i:
                other_cluster = np.where(labels == lbl)[0]
                if len(other_cluster) > 0:
                    distances = np.sqrt(np.sum((data[i] - data[other_cluster]) ** 2, axis=1))
                    b_i = min(b_i, np.mean(distances))

        if b_i == np.inf:
            b_i = 0

        if max(a_i, b_i) > 0:
            s[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            s[i] = 0

    return np.mean(s)


# ============================================================================
# Affinity Propagation Core
# ============================================================================

def apcluster_core(S, p_arr, N, lam, maxits, convits, nonoise, adapt):
    """Core affinity propagation algorithm."""
    if not nonoise:
        rng_state = np.random.get_state()
        np.random.seed(0)
        S = S + (np.finfo(float).eps * S + np.finfo(float).tiny * 100) * np.random.rand(N, N)
        np.random.set_state(rng_state)

    dS = np.diag(S)
    A = np.zeros((N, N))
    R = np.zeros((N, N))

    stoptimes = max(maxits // 10, 2000)
    if adapt == 0:
        stoptimes = convits

    Hstop = np.zeros((N, stoptimes), dtype=int)
    Hconvits = np.zeros((N, convits), dtype=int)
    Tdelay = 10
    Hdelay = Tdelay
    Hconverg = False
    nhalf = round(0.3 * convits)
    Hdelay2 = Tdelay
    Hconvhalf = np.zeros((N, nhalf), dtype=int)
    Hsavehalf = False
    Hn1 = 0
    Hn2 = 0
    Kmean = 0
    Wstart = max(100, round(convits / 2))
    wsize = 40
    Kocil = np.ones(wsize)
    Noscil = float(wsize + 10)
    Svib = 0
    Hvib = 10
    Tvib = 2
    Hguid = 1
    Sprefer = p_arr.copy()
    astep = FOLDS * np.median(S[S > -np.finfo(float).max])
    Kset = []
    Kold = 0
    Kfix = 0
    nKfix = 0
    Kmax = 0
    nfix = 0

    netsim_all = []
    dpsim_all = []
    expref_all = []
    idx_all = []

    iteration = 0
    dn = False

    while not dn:
        iteration += 1

        # Responsibility pass
        AS = A + S
        Y = np.max(AS, axis=1)
        I = np.argmax(AS, axis=1)
        for k in range(N):
            AS[k, I[k]] = -np.finfo(float).max
        Y2 = np.max(AS, axis=1)
        AS = []

        Rold = R.copy()
        R = S - Y[:, np.newaxis]
        for k in range(N):
            R[k, I[k]] = S[k, I[k]] - Y2[k]
        R = (1 - lam) * R + lam * Rold

        # Availability pass
        Rp = np.maximum(R, 0)
        for k in range(N):
            Rp[k, k] = R[k, k]

        Aold = A.copy()
        A = np.tile(np.sum(Rp, axis=0), (N, 1)) - Rp
        dA = np.diag(A)
        A = np.minimum(A, 0)
        for k in range(N):
            A[k, k] = dA[k]
        A = (1 - lam) * A + lam * Aold

        # Convergence check
        E = (np.diag(A) + np.diag(R)) > 0
        Hconvits[:, (iteration - 1) % convits] = E.astype(int)
        K = int(E.sum())
        Kset.append(K)

        if iteration > 5:
            Kset_arr = np.array(Kset)
            Kmean_arr = np.zeros(iteration)
            for ii in range(6, iteration + 1):
                Kmean_arr[ii - 1] = np.mean(Kset_arr[max(0, ii - 6):ii])
            Kmean = Kmean_arr[iteration - 1]

        Hstop[:, (iteration - 1) % stoptimes] = E.astype(int)
        Hconvhalf[:, (iteration - 1) % nhalf] = E.astype(int)

        if (iteration - 1) % 100 == 0 or iteration == maxits:
            print(f'** running at iteration {iteration}, K = {K}')

        Hsave = Hsave1 = Hsave2 = Hsave3 = False

        if iteration >= Wstart or iteration >= maxits:
            se = np.sum(Hconvits, axis=1)
            se1 = int(np.sum(se == convits))
            se2 = int(np.sum(se == 0))
            unconverged = (se1 + se2) != N
            Hconverg = not unconverged

            se = np.sum(Hstop, axis=1)
            se1 = int(np.sum(se == stoptimes))
            se2 = int(np.sum(se == 0))
            if (se1 + se2) == N or iteration == maxits:
                dn = True
                if (se1 + se2) == N:
                    Hsave1 = True

            se = np.sum(Hconvhalf, axis=1)
            se1 = int(np.sum(se == nhalf))
            se2 = int(np.sum(se == 0))
            Hsavehalf = ((se1 + se2) == N) and (Hguid == 2)

        # Adaptive preference reduction
        if adapt:
            if iteration > 5:
                Kset_arr = np.array(Kset)
                Kdown_arr = np.zeros(iteration)
                for ii in range(7, iteration + 1):
                    k_mean_ii = np.mean(Kset_arr[max(0, ii - 6):ii])
                    k_mean_im1 = np.mean(Kset_arr[max(0, ii - 7):ii - 1])
                    Kdown_arr[ii - 1] = k_mean_ii - k_mean_im1 < 0
                Kdown = Kdown_arr[iteration - 1] if iteration - 1 < len(Kdown_arr) else False
                if Hguid == 2:
                    Kdown = Kdown and (K <= Kold)

                Kunchange = int(np.sum(np.abs(np.array(Kset[max(0, iteration - 6):iteration]) - Kset[iteration - 1])))
                Kocil[(iteration - 1) % wsize] = Kdown or Kunchange == 0
                Noscil = float(np.sum(Kocil))

            if Hconverg:
                Hdelay += 1
                if Hdelay >= Tdelay:
                    Hsave1 = True
                    Hdelay = 0
                    Hn1 += 1
                    if K == Kfix:
                        nKfix += 1
                    else:
                        nKfix = 0
                    Kfix = K
                    stepfold = np.sqrt(K + 50) / 10
                    pstep = FOLDS * np.median(S[S > -np.finfo(float).max]) / stepfold
                    astep = nKfix * pstep if nKfix > 1 else pstep
            elif Hsavehalf:
                Hdelay2 += 1
                if Hdelay2 >= Tdelay:
                    Hsave2 = True
                    Hdelay2 = 0
                    Hn2 += 1

            if not Hconverg:
                Hn1 = 0
                Hdelay = 0
            if not Hsavehalf:
                Hn2 = 0
                Hdelay2 = 0

            if K in (1, 2) and Hsave1:
                dn = True
                unconverged = False

            if Hguid == 1 and Hsave1:
                Hguid = 2
                labels = np.zeros((N, K), dtype=int)
                labelid = np.zeros((N, K), dtype=int)
                NC = np.zeros(K)
                NCfix = np.zeros(K)
                Sp = np.zeros(K)
                Slam = np.zeros(K)
                Kmax = K
                stepfold = np.sqrt(Kmax + 50) / 10
                pstep = FOLDS * np.median(S[S > -np.finfo(float).max]) / stepfold

            if Hsave1:
                Svib = 0
                Sprefer = Sprefer + astep
                if Sprefer.size == 1:
                    np.fill_diagonal(S, Sprefer[0])
                else:
                    for k in range(N):
                        S[k, k] = Sprefer[k]
            else:
                Svib += 1
                HSvib = (Svib > wsize and Noscil < 0.66 * wsize) or Svib > 150
                HSvib = HSvib and iteration > Wstart
                if HSvib:
                    Hvib += 1
                    if Hvib > 10:
                        lam = max(0.7, lam)
                    elif Hvib >= 1:
                        if Tvib >= 3:
                            if lam >= 0.9:
                                lam = min(0.98, 0.025 + lam)
                                if lam >= 0.95 and (iteration - 1) % 9 == 2:
                                    rng_state = np.random.get_state()
                                    np.random.seed(0)
                                    S = S + (np.finfo(float).eps * S + np.finfo(float).tiny * 1000) * np.random.rand(N, N)
                                    np.random.set_state(rng_state)
                                    print(' # A small amount of noise is added')
                        else:
                            lam = min(0.9, 0.05 + lam)
                        if lam >= 0.85:
                            Tvib += 1
                            Sprefer = Sprefer + astep
                            if Sprefer.size == 1:
                                np.fill_diagonal(S, Sprefer[0])
                            else:
                                for k in range(N):
                                    S[k, k] = Sprefer[k]
                            print(' # Escaping oscillation turns on')
                    Hvib = 0
                    Svib = 0
                    print(f' # Damping factor is increased to {lam}')
                else:
                    Tvib = max(Tvib - 0.002, 0.98)
                    if lam > 0.9 and Tvib < 1:
                        lam = max(lam - 0.0001, 0.5)

        newp = Sprefer[0] if Sprefer.size == 1 else Sprefer[-1]
        newlam = lam

        if Hguid >= 2 and (K < Kold and K > 1 or Kmean == np.mean(Kset[max(0, iteration - 2):iteration + 1])):
            Hsave3 = True
        if Hsave1 or Hsave2 or Hsave3:
            Hsave = True
        Kold = K

        # Compute metrics
        if K == 0:
            tmpnetsim = np.nan
            tmpdpsim = np.nan
            tmpexpref = np.nan
            tmpidx = np.full(N, np.nan)
        else:
            idx_E = np.where(E)[0]
            S_col = S[:, idx_E]
            tmp = np.argmax(S_col, axis=0)
            c = np.zeros(len(idx_E), dtype=int)
            c[:] = np.arange(1, K + 1)
            c_map = np.zeros(N, dtype=int)
            c_map[idx_E] = c

            if Hsave or dn:
                if Hsave1:
                    nfix = Hn1 * Tdelay + convits
                elif Hsave2:
                    nfix = Hn2 * Tdelay + nhalf
                elif Kmean == np.mean(Kset[max(0, iteration - 2):iteration + 1]):
                    nfix = 6
                    if iteration >= 6:
                        k_mean_now = np.mean(Kset[max(0, iteration - 6):iteration + 1])
                        if Kmean == k_mean_now:
                            nfix = 10
                else:
                    nfix = 1

                if (K <= Kmax and nfix > NCfix[K - 1]) or (K > Kmax and nfix >= 10):
                    NCfix[K - 1] = nfix
                    labels[:, K - 1] = c_map
                    # Only assign labelid at exemplar positions (idx_E)
                    labelid[idx_E, K - 1] = idx_E
                    NC[K - 1] = K
                    Sp[K - 1] = newp
                    Slam[K - 1] = newlam
                if K > Kmax:
                    Kmax = K
                    if len(NCfix) < K:
                        NCfix = np.append(NCfix, 0)

                # Compute tmpidx_i: for each point i, the exemplar it's assigned to
                # c_map stores cluster position (1-indexed) for each exemplar point
                # For non-exemplars, we need to find which exemplar their cluster uses
                tmpidx_i = np.zeros(N, dtype=int)
                # Exemplars map to themselves (0-indexed: idx_E[k] is the row index)
                tmpidx_i[idx_E] = idx_E
                # For non-exemplars: cluster = c_map[exemplar], need to find exemplar for that cluster
                # c_map[j] = k+1 means exemplar j is at position k in idx_E
                # So exemplar_of_cluster[k] = idx_E[k]
                for i in range(N):
                    if i not in idx_E:
                        # Find which cluster point i belongs to by looking at the max similarity
                        # Actually, for cluster assignment: point i's cluster = label from similarity pattern
                        # Simpler: use argmax of S[i, idx_E] to find closest exemplar
                        # But we need cluster label, not directly executable here
                        # Since c_map only set for exemplars, use the fact that c_map[idx_E[k]] = k+1
                        # For non-exemplar i, we need to find its cluster and then the exemplar for that cluster
                        pass  # Keep existing tmpidx_i[i] = 0 for now - will be overwritten by the code below

                # Correct approach: for each point i, find its closest exemplar in idx_E
                S_to_exemplars = S[:, idx_E]  # N x K matrix of similarities to each exemplar
                closest_exemplar_idx = np.argmax(S_to_exemplars, axis=1)  # Index into idx_E
                tmpidx_i = idx_E[closest_exemplar_idx]  # Actual row indices in S

                # Compute net similarity: sum S[i, tmpidx_i[i]] for each i
                tmpnetsim = 0.0
                for i in range(N):
                    tmpnetsim += S[i, tmpidx_i[i]]

                tmpexpref = np.sum(dS[idx_E])
                tmpdpsim = tmpnetsim - tmpexpref
                tmpidx = tmpidx_i
            else:
                # For each point i, find its closest exemplar
                S_to_exemplars = S[:, idx_E]
                closest_exemplar_idx = np.argmax(S_to_exemplars, axis=1)
                tmpidx_i = idx_E[closest_exemplar_idx]
                tmpnetsim = 0.0
                for i in range(N):
                    tmpnetsim += S[i, tmpidx_i[i]]
                tmpexpref = np.sum(dS[idx_E])
                tmpdpsim = tmpnetsim - tmpexpref
                tmpidx = tmpidx_i

        if DETAILS:
            netsim_all.append(tmpnetsim)
            dpsim_all.append(tmpdpsim)
            expref_all.append(tmpexpref)
            idx_all.append(tmpidx)

        if iteration >= maxits:
            dn = True

    print(f' # Programs run over at K= {K}')
    I_ex = np.where(np.diag(A + R) > 0)[0]
    K = len(I_ex)
    if K > 0:
        # c should be size N (cluster assignment for each data point)
        c = np.argmax(S[:, I_ex], axis=1)  # For each data point, which exemplar is closest
        for k in range(K):
            c[I_ex[k]] = k + 1  # Exemplar points get their cluster number

        # Refine: find actual exemplar for each cluster (point with max sum in cluster)
        for k in range(K):
            ii = np.where(c == k + 1)[0]
            if len(ii) > 0:
                sums = np.sum(S[np.ix_(ii, ii)], axis=1)
                _, jm = np.max(sums), np.argmax(sums)
                I_ex[k] = ii[jm]

        # Recompute cluster assignments using updated I_ex
        c = np.argmax(S[:, I_ex], axis=1)
        for k in range(K):
            c[I_ex[k]] = k + 1

        tmpidx = I_ex[c - 1]
        # Compute net similarity: sum of similarities from each point to its exemplar
        tmpnetsim = 0.0
        for i in range(N):
            tmpnetsim += S[i, tmpidx[i]]
        tmpexpref = np.sum(dS[I_ex])

        labels_final = c
        labelid_final = tmpidx
        NC_final = K
        NCfix_final = nfix
        Sp_final = newp
        Slam_final = newlam
        dpsim_final = tmpnetsim - tmpexpref
    else:
        tmpnetsim = np.nan
        tmpexpref = np.nan
        dpsim_final = np.nan
        tmpidx = np.full(N, np.nan)
        labels_final = np.full(N, 1)
        labelid_final = np.full(N, 1)
        NC_final = 0
        NCfix_final = 0
        Sp_final = np.array([])
        Slam_final = np.array([])

    netsim_out = np.array(netsim_all) if netsim_all else tmpnetsim
    dpsim_out = np.array(dpsim_all) if dpsim_all else dpsim_final
    expref_out = np.array(expref_all) if expref_all else tmpexpref
    idx_out = np.array(idx_all) if idx_all else tmpidx

    # Post-process
    NC_arr = NC if 'NC' in dir() and NC.size > 0 else np.array([NC_final])
    if NC_arr.size > 1:
        NC_arr[0] = 0
    valid_mask = NC_arr > 0
    if NC_arr.size < 1 or (NC_arr.size == 1 and NC_arr[0] == 0):
        if N is not None:
            labels_final = np.ones(N, dtype=int)
            labelid_final = np.ones(N, dtype=int)
        Sp_final = np.array([])
        Slam_final = np.array([])
        NC_final = 0
        NCfix_final = 0
    else:
        S_sel = np.where(valid_mask)[0]
        Sp_final = Sp[S_sel] if 'Sp' in dir() and Sp.size > 0 else Sp_final
        Slam_final = Slam[S_sel] if 'Slam' in dir() and Slam.size > 0 else Slam_final
        labels_final = labels[:, S_sel] if 'labels' in dir() and labels.size > 0 else labels_final
        labelid_final = labelid[:, S_sel] if 'labelid' in dir() and labelid.size > 0 else labelid_final
        NCfix_final = NCfix[S_sel] if 'NCfix' in dir() and NCfix.size > 0 else NCfix_final
        # NC_final is simply K (the number of clusters at the final iteration)
        NC_final = K

    if PLOT or DETAILS:
        print(f'\nNumber of identified clusters: {K}')
        print(f'Fitness (net similarity): {tmpnetsim}')
        print(f'  Similarities of data points to exemplars: {dpsim_final}')
        print(f'  Preferences of selected exemplars: {tmpexpref}')
        print(f'Number of iterations: {iteration}\n')

    if unconverged:
        print(f'\n*** Warning: Algorithm did not converge at K = {NC_final} !')
        print('    The similarities may contain degeneracies - add noise to')
        print('    the similarities to remove degeneracies. To monitor the net')
        print('    similarity, activate plotting. Also, consider increasing')
        print('    maxits and if necessary dampfact.\n')

    return (
        labels_final,
        int(NC_final) if isinstance(NC_final, (int, float, np.number)) else NC_final,
        labelid_final,
        iteration,
        Sp_final,
        Slam_final,
        NCfix_final,
        netsim_out,
        dpsim_out,
        expref_out,
        idx_out,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print(" Adaptive Affinity Propagation Clustering")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading data: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: File not found: {DATA_FILE}")
        print("Please update DATA_FILE in the config section.")
        return

    data, true_labels = load_data(DATA_FILE, TRUE_LABELS == 1)
    print(f"    Data shape: {data.shape[0]} samples x {data.shape[1]} features")
    if TRUE_LABELS == 1:
        print(f"    True labels: {len(np.unique(true_labels))} clusters")
    else:
        print("    True labels: not provided")

    # Compute similarity matrix
    print(f"\n[2] Computing similarity ({'Euclidean' if DTYPE == 1 else 'Pearson'})")
    if DTYPE == 1:
        Dist, _ = similarity_euclid(data, squared=False)
        S = -Dist
    elif DTYPE == 2:
        Dist = 1 - (1 + similarity_pearson(data)) / 2
        S = -Dist
    else:
        print("ERROR: DTYPE must be 1 (euclidean) or 2 (correlation)")
        return

    N = data.shape[0]
    # Parse pvalues
    if PVALUES is None:
        dn_mask = S[S > -np.finfo(float).max]
        p_arr = np.array([np.median(dn_mask) * 0.5])
    else:
        p_arr = np.atleast_1d(PVALUES)

    print(f"    Similarity matrix: {N}x{N}")

    # Run clustering
    print(f"\n[3] Running Adaptive Affinity Propagation Clustering")
    print(f"    Damping: {DAMPFACT}, Max iterations: {MAXITS}, Convergence: {CONVITS}")
    print(f"    Adaptive: {'Yes' if ADAPT else 'No'}, Cut: {CUT}")

    labels, NC, labelid, iend, Sp, Slam, NCfix, netsim, dpsim, expref, idx_out = apcluster_core(
        S, p_arr, N, DAMPFACT, MAXITS, CONVITS, NONOISE, ADAPT
    )

    # Extract final cluster assignments from labelid
    # labelid stores exemplar index for each point in the last column
    # We need to map exemplar indices to cluster labels (1, 2, 3, ...)
    if labelid.ndim > 1:
        final_exemplars = labelid[:, -1]  # exemplar index for each point
        # Find unique exemplars and map to cluster labels
        unique_exemplars = np.unique(final_exemplars)
        unique_exemplars = unique_exemplars[unique_exemplars > 0]  # exclude 0
        K_actual = len(unique_exemplars)
        # Create mapping: exemplar -> cluster label (1, 2, 3, ...)
        exemplar_to_cluster = {ex: i+1 for i, ex in enumerate(unique_exemplars)}
        # Map each point to its cluster
        labels = np.array([exemplar_to_cluster.get(ex, 0) for ex in final_exemplars])
        NC = K_actual
    else:
        labels = labelid.copy() if np.ndim(labelid) == 1 else labelid[:, -1]

    print(f"\n    Iterations: {iend}")
    print(f"    Clusters found (before cut): {NC}")
    print(f"    DEBUG: unique labels = {np.unique(labels)}, counts = {[np.sum(labels == l) for l in np.unique(labels)]}")

    # Drop small clusters
    if CUT > 1:
        unique_labels = np.unique(labels)
        valid_clusters = []
        for lbl in unique_labels:
            if lbl == 0:
                continue
            count = np.sum(labels == lbl)
            if count >= CUT:
                valid_clusters.append(lbl)
        valid_clusters = np.array(valid_clusters)
        print(f"    DEBUG: valid_clusters = {valid_clusters}")

        if len(valid_clusters) > 0:
            new_labels = np.zeros_like(labels)
            for new_idx, old_lbl in enumerate(valid_clusters, start=1):
                new_labels[labels == old_lbl] = new_idx
            labels = new_labels
            NC = len(valid_clusters)
        else:
            labels = np.zeros_like(labels)
            NC = 0
        print(f"    Clusters found (after cut={CUT}): {NC}")

    # Validation
    print(f"\n[4] Validation")
    if TRUE_LABELS == 1 and true_labels is not None:
        fm = fowlkes_mallows(labels, true_labels)
        ri = rand_index(labels, true_labels)
        print(f"    Fowlkes-Mallows Index: {fm:.4f}")
        print(f"    Rand Index: {ri:.4f}")

    sil = silhouette_score(data, labels)
    print(f"    Silhouette Score: {sil:.4f}")

    # Results
    print(f"\n[5] Results Summary")
    print(f"    Number of clusters: {NC}")
    print(f"    Net similarity: {netsim:.4f}" if isinstance(netsim, (int, float, np.number)) and not np.isnan(netsim) else f"    Net similarity: {netsim}")

    # Save labels
    output_file = os.path.splitext(DATA_FILE)[0] + '_labels.csv'
    labels_df = pd.DataFrame({
        'sample_id': np.arange(1, N + 1),
        'cluster_label': labels
    })
    labels_df.to_csv(output_file, index=False)
    print(f"\n    Labels saved to: {output_file}")

    print("\n" + "=" * 60)
    print(" Done")
    print("=" * 60)


if __name__ == '__main__':
    main()