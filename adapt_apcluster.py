import numpy as np
from scipy.spatial.distance import cdist


# ── Similarity helpers ─────────────────────────────────────────────────────────

def similarity_euclid(data, squared=False):
    """
    Pairwise Euclidean (or squared Euclidean) distance between rows of data.

    Parameters
    ----------
    data    : (N, d) ndarray
    squared : bool — if True return squared distances

    Returns
    -------
    R    : (N, N) distance matrix
    dmax : float, maximum distance value
    """
    R = cdist(data, data, metric='sqeuclidean')
    if squared:
        return R, float(R.max())
    R = np.sqrt(R)
    return R, float(R.max())


def similarity_pearson(data):
    """
    Pearson correlation between every pair of rows in data.

    Parameters
    ----------
    data : (N, d) ndarray — rows are observations

    Returns
    -------
    R : (N, N) correlation matrix in [-1, 1]
    """
    R = np.corrcoef(data)
    np.fill_diagonal(R, 1.0)
    return R


# ── Main algorithm ─────────────────────────────────────────────────────────────

def adapt_apcluster(data, dtype='euclidean', pvalues=None, folds=0.01, adapt=1,
                    maxits=500, convits=50, lam=0.5,
                    plot=False, details=False, nonoise=False):
    """
    Adaptive Affinity Propagation clustering.

    Parameters
    ----------
    data    : ndarray
              (N, d) raw data matrix,
              OR (M, 3) pre-computed similarities as [i, j, s] with 1-based indices.
    dtype   : 'euclidean' | 'correlation' | 1 | 2
              Ignored when data is already in (M, 3) similarity format.
    pvalues : float or (N,) ndarray, preference(s).
              None → use median similarity × 0.5.
    folds   : float, preference step factor (default 0.01).
    adapt   : int, >0 for adaptive AP; 0 for original AP.
    maxits  : int, maximum iterations.
    convits : int, convergence window (iterations exemplars must stay fixed).
    lam     : float, damping factor in [0.5, 1.0).
    plot    : bool, print iteration info (matplotlib plotting not implemented).
    details : bool, record per-iteration netsim/dpsim/expref/idx.
    nonoise : bool, skip the small noise addition step.

    Returns
    -------
    labels  : (N, n_valid) int ndarray
              Cluster assignment (1-based) for each valid K found.
              Column j corresponds to NC[j] clusters.
    NC      : (n_valid,) int ndarray, valid cluster counts discovered.
    labelid : (N, n_valid) int ndarray
              Exemplar index (1-based) for each point, per valid K.
    it      : int, total iterations run.
    Sp      : (n_valid,) float, preference value at each valid K.
    Slam    : (n_valid,) float, damping factor at each valid K.
    NCfix   : (n_valid,) int, confidence score per valid K (higher = more stable).
    netsim  : float or ndarray, net similarity.
    dpsim   : float or ndarray, discriminating similarity.
    expref  : float or ndarray, sum of selected exemplar preferences.
    idx     : (N,) or (N, T) int ndarray, 0-based exemplar assignment per point.
    """
    adapt = adapt + 1  # adapt=0 → original AP (adapt < 2); adapt>=1 → adaptive (adapt >= 2)

    # ── Build sparse similarity triplets [i, j, s] (1-based i, j) ────────────
    if adapt < 2:
        if dtype in ('euclidean', 1):
            Dist, _ = similarity_euclid(data, squared=True)
        else:
            Dist = 1.0 - (1.0 + similarity_pearson(data)) / 2.0

        nrow   = Dist.shape[0]
        r, c   = np.where(~np.eye(nrow, dtype=bool))
        s      = np.column_stack([r + 1, c + 1, -Dist[r, c]]).astype(float)
        Dist   = None
    else:
        s = np.asarray(data, dtype=float).copy()
        data = None

    # ── Preference setup ──────────────────────────────────────────────────────
    pfixed = False
    valid_mask = s[:, 2] > -np.finfo(float).max
    pmedian = float(np.median(s[valid_mask, 2]))
    pstep_base = folds * pmedian

    pvalues_arr = np.atleast_1d(pvalues if pvalues is not None else pmedian * 0.5).astype(float)
    if pvalues is not None:
        pfixed = True

    if lam > 0.9:
        print('\n*** Warning: Large damping factor in use. Consider increasing convits.\n')

    # ── Validate input and build NxN similarity matrix S ─────────────────────
    if s.shape[1] == 3:
        tmp = int(max(s[:, 0].max(), s[:, 1].max()))
        N = tmp if len(pvalues_arr) == 1 else len(pvalues_arr)
        if tmp > N:
            raise ValueError('data point index exceeds number of data points')
        if min(s[:, 0].min(), s[:, 1].min()) <= 0:
            raise ValueError('data point indices must be >= 1')
        S = np.full((N, N), -np.inf)
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

    s = None

    if N > 3000:
        print('\n*** Warning: Large memory request. Consider a sparse approach.\n')

    # Small noise to break degeneracies
    if not nonoise:
        rng = np.random.get_state()
        np.random.seed(0)
        S += (np.finfo(float).eps * S + np.finfo(float).tiny * 100) * np.random.rand(N, N)
        np.random.set_state(rng)

    # Set preferences on diagonal
    if len(pvalues_arr) == 1:
        np.fill_diagonal(S, float(pvalues_arr[0]))
    else:
        np.fill_diagonal(S, pvalues_arr)

    # ── Allocate message matrices and history buffers ─────────────────────────
    dS = np.diag(S).copy()
    A = np.zeros((N, N))
    R = np.zeros((N, N))

    netsim_hist = dpsim_hist = expref_hist = idx_hist = None
    if plot or details:
        netsim_hist = np.full(maxits + 2, np.nan)
    if details:
        dpsim_hist  = np.full(maxits + 2, np.nan)
        expref_hist = np.full(maxits + 2, np.nan)
        idx_hist    = np.full((N, maxits + 2), -1, dtype=int)

    # ── State variables ───────────────────────────────────────────────────────
    dn = False
    it = 0
    stoptimes = convits if pfixed else max(maxits // 10, 2000)

    Hstop     = np.zeros((N, stoptimes))
    Hconvits  = np.zeros((N, convits))
    nhalf     = max(1, round(0.3 * convits))
    Hconvhalf = np.zeros((N, nhalf))

    Tdelay   = 10
    Hdelay   = Tdelay
    Hdelay2  = Tdelay
    Hconverg = False
    Hsavehalf = False
    Hn1 = 0;  Hn2 = 0

    Wstart    = max(100, round(convits / 2))
    wsize     = 40
    Kocil     = np.ones(wsize, dtype=bool)
    Noscil    = wsize + 10
    Svib      = 0
    Hvib      = 10;  Tvib = 2.0
    Hguid     = 1

    Sprefer   = float(pvalues_arr[0]) if len(pvalues_arr) == 1 else pvalues_arr.copy()
    pstep     = pstep_base
    astep     = pstep

    buf = maxits + 20
    Kset      = np.zeros(buf, dtype=int)
    Kmean     = np.zeros(buf)
    Kdown     = np.zeros(buf, dtype=bool)
    Kunchange = np.zeros(buf)

    Kold  = 0;  Kfix = 0;  nKfix = 0
    Kmax  = 0;  nfix = 0
    unconverged = False

    tmpnetsim = tmpdpsim = tmpexpref = np.nan
    tmpidx = np.full(N, -1, dtype=int)

    # Output storage — columns indexed by (K - 1)
    _cap = min(N, 512)
    labels_out  = np.zeros((N, _cap), dtype=int)
    labelid_out = np.zeros((N, _cap), dtype=int)
    NC_out      = np.zeros(_cap, dtype=int)
    NCfix_out   = np.zeros(_cap, dtype=int)
    Sp_out      = np.zeros(_cap)
    Slam_out    = np.zeros(_cap)

    def _grow(k):
        nonlocal labels_out, labelid_out, NC_out, NCfix_out, Sp_out, Slam_out
        if k <= labels_out.shape[1]:
            return
        extra = k - labels_out.shape[1]
        labels_out  = np.hstack([labels_out,  np.zeros((N, extra), dtype=int)])
        labelid_out = np.hstack([labelid_out, np.zeros((N, extra), dtype=int)])
        NC_out      = np.concatenate([NC_out,   np.zeros(extra, dtype=int)])
        NCfix_out   = np.concatenate([NCfix_out, np.zeros(extra, dtype=int)])
        Sp_out      = np.concatenate([Sp_out,   np.zeros(extra)])
        Slam_out    = np.concatenate([Slam_out,  np.zeros(extra)])

    # Cache float-info constants — avoids repeated object construction inside the hot loop
    _FMAX  = np.finfo(float).max
    _FEPS  = np.finfo(float).eps
    _FTINY = np.finfo(float).tiny

    # ── Main message-passing loop ─────────────────────────────────────────────
    while not dn:
        it += 1

        # Responsibilities  R(i,k) = S(i,k) - max_{j≠k}[A(i,j) + S(i,j)]
        AS = A + S
        Y      = AS.max(axis=1)
        I_max  = AS.argmax(axis=1)
        AS[np.arange(N), I_max] = -_FMAX
        Y2     = AS.max(axis=1)
        AS     = None
        Rold   = R
        R      = S - Y[:, np.newaxis]
        R[np.arange(N), I_max] = S[np.arange(N), I_max] - Y2
        R      = (1 - lam) * R + lam * Rold
        Rold   = None

        # Availabilities  A(i,k) = min(0, R(k,k) + Σ_{j≠i,k} max(0,R(j,k)))
        Rp = np.maximum(R, 0.0)
        Rp[np.arange(N), np.arange(N)] = R[np.arange(N), np.arange(N)]
        Aold = A
        A    = Rp.sum(axis=0)[np.newaxis, :] - Rp  # col sums broadcast minus Rp
        Rp   = None
        dA   = np.diag(A).copy()
        np.minimum(A, 0.0, out=A)
        A[np.arange(N), np.arange(N)] = dA
        A    = (1 - lam) * A + lam * Aold
        Aold = None

        E = (np.einsum('ii->i', A) + np.einsum('ii->i', R)) > 0
        Hconvits[:, (it - 1) % convits]   = E
        Hstop[:,    (it - 1) % stoptimes]  = E
        Hconvhalf[:,(it - 1) % nhalf]      = E

        K      = int(E.sum())
        Kset[it] = K
        newp   = float(np.atleast_1d(Sprefer)[0])
        newlam = lam

        if it % 100 == 1 or it == maxits:
            print(f'** running at iteration {it}, K = {K}')

        Hsave = Hsave1 = Hsave2 = Hsave3 = False

        # ── Convergence checks ────────────────────────────────────────────
        if it >= Wstart or it >= maxits:
            se = Hconvits.sum(axis=1)
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

        # ── Adaptive mechanisms ───────────────────────────────────────────
        # MATLAB's `if adapt` is always True after the +1 increment, so these
        # mechanisms run for both adapt=0 and adapt=1 user inputs.
        if adapt >= 1:
            if it > 5:
                Kmean[it]     = Kset[it - 5:it + 1].mean()
                Kdown[it]     = (Kmean[it] - Kmean[it - 1]) < 0
                if Hguid == 2:
                    Kdown[it] = Kdown[it] and (K <= Kold)
                Kunchange[it] = int(np.abs(Kset[it] - Kset[it - 5:it]).sum())
                Kocil[(it - 1) % wsize] = Kdown[it] or (Kunchange[it] == 0)
                Noscil = int(Kocil.sum())

            # Reduce preference when K has converged
            if Hconverg:
                Hdelay += 1
                if Hdelay >= Tdelay:
                    Hsave1  = True
                    Hdelay  = 0
                    Hn1    += 1
                    nKfix   = (nKfix + 1) if K == Kfix else 0
                    Kfix    = K
                    stepfold = np.sqrt(K + 50) / 10.0
                    pstep   = folds * pmedian / stepfold
                    astep   = nKfix * pstep if nKfix > 1 else pstep
            elif Hsavehalf:
                Hdelay2 += 1
                if Hdelay2 >= Tdelay:
                    Hsave2  = True
                    Hdelay2 = 0
                    Hn2    += 1

            if not Hconverg:
                Hn1 = 0;  Hdelay = 0
            if not Hsavehalf:
                Hn2 = 0;  Hdelay2 = 0

            if (K == 1 or K == 2) and Hsave1:
                dn = True;  unconverged = False

            # Transition to guidance phase
            if Hguid == 1 and Hsave1:
                Hguid    = 2
                Kmax     = K
                stepfold = np.sqrt(Kmax + 50) / 10.0
                pstep    = folds * pmedian / stepfold

            if Hsave1:
                Svib = 0
                if not pfixed:
                    Sprefer = np.atleast_1d(Sprefer) + astep
                    Sprefer = float(Sprefer[0]) if Sprefer.size == 1 else Sprefer
                    np.fill_diagonal(S, Sprefer)
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
                                    S += (_FEPS * S + _FTINY * 1000) * np.random.rand(N, N)
                                    np.random.set_state(rng)
                                    print(' # A small amount of noise is added')
                        else:
                            lam = min(0.9, 0.05 + lam)

                        if lam >= 0.85:
                            Tvib += 1
                            if not pfixed:
                                if Hguid == 2 and Kold:
                                    sf = 2.0 if Kmax < 1 else max(3.0 / (np.sqrt(Kmax) / 10 + 0.4), 1.0)
                                    Kvar  = 2.0 * float(np.sqrt(np.std(Kset[max(1, it - 49):it + 1], ddof=1)))
                                    astep = min(0.8 * Kvar + 0.2 * Tvib, sf) * pstep
                                else:
                                    astep = min(Tvib, 2.0) * pstep
                                Sprefer = np.atleast_1d(Sprefer) + astep
                                Sprefer = float(Sprefer[0]) if Sprefer.size == 1 else Sprefer
                                np.fill_diagonal(S, Sprefer)
                                print(' # Escaping oscillation turns on')

                    Hvib = 0;  Svib = 0
                    print(f' # Damping factor is increased to {lam:.4g}')
                else:
                    Tvib = max(Tvib - 0.002, 0.98)
                    if lam > 0.9 and Tvib < 1:
                        lam = max(lam - 0.0001, 0.5)

        # Catch decreasing K
        if Hguid >= 2 and it > 1 and ((K < Kold and K > 1) or Kmean[it] == Kmean[it - 1]):
            Hsave3 = True
        Hsave = Hsave1 or Hsave2 or Hsave3
        Kold  = K

        # ── Record / evaluate solution ────────────────────────────────────
        if plot or details or Hsave or dn:
            if K == 0:
                tmpnetsim = tmpdpsim = tmpexpref = np.nan
                tmpidx = np.full(N, -1, dtype=int)
            else:
                I_ex = np.where(E)[0]               # exemplar indices (0-based)
                c    = S[:, I_ex].argmax(axis=1)    # each point → index into I_ex
                c[I_ex] = np.arange(K)              # exemplars map to themselves

                if Hsave or dn:
                    # Compute confidence score for this K
                    if Hsave1:
                        nfix = Hn1 * Tdelay + convits
                    elif Hsave2:
                        nfix = Hn2 * Tdelay + nhalf
                    elif it > 1 and Kmean[it] == Kmean[it - 1]:
                        nfix = 10 if (it > 5 and Kmean[it] == Kmean[max(0, it - 5)]) else 6
                    else:
                        nfix = 1

                    _grow(K)
                    ki = K - 1  # 0-based column
                    if (K <= Kmax and nfix > NCfix_out[ki]) or (K > Kmax and nfix >= 10):
                        NCfix_out[ki]   = nfix
                        labels_out[:,  ki] = c + 1          # 1-based cluster number
                        labelid_out[:, ki] = I_ex[c] + 1    # 1-based exemplar index
                        NC_out[ki]      = K
                        Sp_out[ki]      = newp
                        Slam_out[ki]    = newlam
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
                if tmpidx is not None and not np.any(np.isnan(tmpidx.astype(float))):
                    idx_hist[:, it] = tmpidx
            elif plot:
                netsim_hist[it] = tmpnetsim

    # ── Final refinement: re-select best exemplar per cluster ─────────────────
    print(f' # Programs run over at K= {K}')
    I_final = np.where((np.einsum('ii->i', A) + np.einsum('ii->i', R)) > 0)[0]
    K_final = len(I_final)

    if K_final > 0:
        c_f = S[:, I_final].argmax(axis=1)
        c_f[I_final] = np.arange(K_final)
        for k in range(K_final):
            ii    = np.where(c_f == k)[0]
            j_b   = int(S[np.ix_(ii, ii)].sum(axis=0).argmax())
            I_final[k] = ii[j_b]
        c_f = S[:, I_final].argmax(axis=1)
        c_f[I_final] = np.arange(K_final)
        tmpidx    = I_final[c_f]
        tmpnetsim = float(S[np.arange(N), tmpidx].sum())
        tmpexpref = float(dS[I_final].sum())
        tmpdpsim  = tmpnetsim - tmpexpref

        _grow(K_final)
        ki = K_final - 1
        labels_out[:,  ki] = c_f + 1
        labelid_out[:, ki] = tmpidx + 1
        NC_out[ki]   = K_final
        NCfix_out[ki] = nfix
        Sp_out[ki]   = newp
        Slam_out[ki] = newlam
    else:
        tmpnetsim = tmpdpsim = tmpexpref = np.nan
        tmpidx = np.full(N, -1, dtype=int)

    # ── Package detail outputs ────────────────────────────────────────────────
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

    # ── Trim to valid K values (MATLAB convention: zero out K=1 slot first) ───
    if len(NC_out) > 1:
        NC_out[0] = 0

    valid = np.where(NC_out)[0]
    if len(valid) == 0:
        labels_ret  = np.ones((N, 1), dtype=int)
        labelid_ret = np.ones((N, 1), dtype=int)
        NC_ret      = np.array([0], dtype=int)
        NCfix_ret   = np.array([0], dtype=int)
        Sp_ret      = np.array([])
        Slam_ret    = np.array([])
    else:
        labels_ret  = labels_out[:,  valid]
        labelid_ret = labelid_out[:, valid]
        NC_ret      = NC_out[valid]
        NCfix_ret   = NCfix_out[valid]
        Sp_ret      = Sp_out[valid]
        Slam_ret    = Slam_out[valid]

    if plot or details:
        print(f'\nNumber of identified clusters: {K_final}')
        if not np.isnan(tmpnetsim):
            print(f'Fitness (net similarity): {tmpnetsim:.6f}')
        print(f'Number of iterations: {it}\n')

    if unconverged and len(NC_ret) > 0:
        print(f'\n*** Warning: Algorithm did not converge at K = {NC_ret[0]} !')
        print('    Consider increasing maxits and if necessary dampfact.\n')

    return (labels_ret, NC_ret, labelid_ret, it,
            Sp_ret, Slam_ret, NCfix_ret,
            netsim_out, dpsim_out, expref_out, idx_out)
