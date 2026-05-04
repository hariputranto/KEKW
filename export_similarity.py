import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

input_file   = 'wine.txt'              # same file used in adapt_apcluster_script.py
sim_type     = 'euclidean'             # 'euclidean' | 'correlation'
output_file  = 'similarity_matrix.csv' # output path; '' to skip export
print_limit  = 100                      # max rows/cols to print to console (0 = full)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_data(filepath):
    if os.path.splitext(filepath)[1].lower() == '.csv':
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


def compute_similarity(data, sim_type):
    """
    Return (N, N) similarity matrix.

    Values are negative Euclidean distances (euclidean) or
    transformed Pearson correlations (correlation).
    Off-diagonal preference values; diagonal is left as 0.0 here
    (adapt_apcluster_script sets it to the preference before running).
    """
    if sim_type in ('euclidean', 1):
        D = cdist(data, data, metric='sqeuclidean')
        D = np.sqrt(D)
        S = -D
    else:
        R = np.corrcoef(data)
        np.fill_diagonal(R, 1.0)
        S = -(0.5 - 0.5 * R)   # transform to [−1, 0]; 0 = identical

    np.fill_diagonal(S, 0.0)
    return S


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

data = load_data(input_file)
N    = data.shape[0]
print(f'Loaded {input_file!r}: {N} points, {data.shape[1]} features')

S = compute_similarity(data, sim_type)

# ── Summary stats ────────────────────────────────────────────────────────────
off_diag = S[~np.eye(N, dtype=bool)]
print(f'\nSimilarity matrix  ({N} x {N})  [{sim_type}]')
print(f'  min  : {off_diag.min():.6f}')
print(f'  max  : {off_diag.max():.6f}')
print(f'  mean : {off_diag.mean():.6f}')
print(f'  median: {np.median(off_diag):.6f}')

# ── Print to console ─────────────────────────────────────────────────────────
labels = [f'p{i + 1}' for i in range(N)]
df_S   = pd.DataFrame(S, index=labels, columns=labels)

if print_limit and N > print_limit:
    print(f'\nShowing first {print_limit} rows/cols (set print_limit=0 for full matrix):')
    print(df_S.iloc[:print_limit, :print_limit].to_string())
    if N > print_limit:
        print(f'  ... [{N - print_limit} more rows / cols]')
else:
    print('\nFull similarity matrix:')
    print(df_S.to_string())

# ── Export ───────────────────────────────────────────────────────────────────
if output_file:
    df_S.to_csv(output_file)
    print(f'\nSimilarity matrix saved to: {os.path.abspath(output_file)}')
