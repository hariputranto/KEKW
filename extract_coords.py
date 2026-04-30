import os
import re
import time
import random
import pandas as pd
import requests


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

input_file  = 'places.csv'          # .csv or .xlsx input file
url_column  = 'google_maps_link'    # column name that holds the Google Maps URLs
output_file = 'places_coords.csv'   # output path (.csv or .xlsx); '' = overwrite input
delay       = (5, 30)              # random delay range in seconds between HTTP requests

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def save_file(df, path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


# Ordered from most-specific to least-specific to reduce false matches
_PATTERNS = [
    # /@lat,lng,zoom  — standard browser share URL
    (r'/@(-?\d+\.\d+),(-?\d+\.\d+)', 1, 2),
    # !3d{lat}!4d{lng} — embedded in the data parameter of place URLs
    (r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)', 1, 2),
    # ?q=lat,lng or &q=lat,lng — direct coordinate search
    (r'[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)', 1, 2),
    # ?ll=lat,lng — older Google Maps format
    (r'[?&]ll=(-?\d+\.\d+),(-?\d+\.\d+)', 1, 2),
    # ?center=lat,lng
    (r'[?&]center=(-?\d+\.\d+),(-?\d+\.\d+)', 1, 2),
    # @lat,lng fallback (no leading slash)
    (r'@(-?\d+\.\d+),(-?\d+\.\d+)', 1, 2),
]


_GMAPS_HINTS = re.compile(
    r'(maps\.google\.com|google\.com/maps|maps\.app\.goo\.gl|goo\.gl/maps)',
    re.IGNORECASE,
)


def _find_url_column(df, hint):
    """
    Resolve which column holds Google Maps URLs.

    Resolution order:
      1. Exact match on `hint`.
      2. Case-insensitive name match.
      3. Content scan — return the string column whose first 20 non-null
         values contain the most Google Maps URLs.

    Raises ValueError if nothing is found.
    """
    # 1. Exact match
    if hint in df.columns:
        return hint

    # 2. Case-insensitive name match
    lower_hint = hint.lower()
    for col in df.columns:
        if col.lower() == lower_hint:
            print(f'Column {hint!r} not found; using {col!r} (case-insensitive match).')
            return col

    # 3. Content scan
    best_col, best_score = None, 0
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        score  = sample.apply(lambda v: bool(_GMAPS_HINTS.search(v))).sum()
        if score > best_score:
            best_score, best_col = score, col

    if best_col and best_score > 0:
        print(f'Column {hint!r} not found; auto-detected {best_col!r} '
              f'({best_score} Google Maps URL(s) found in sample).')
        return best_col

    raise ValueError(
        f'Could not find a Google Maps URL column in {df.columns.tolist()}.\n'
        f'Set url_column to the correct column name.'
    )


def _parse(url):
    """Try every regex pattern and return (lat, lng) on first match."""
    for pattern, gi, gj in _PATTERNS:
        m = re.search(pattern, url)
        if m:
            return float(m.group(gi)), float(m.group(gj))
    return None, None


def extract_coords(url, session, delay_s=(15, 30)):
    """
    Extract (lat, lng) from a Google Maps URL.

    First attempts regex parsing on the raw URL (works for full URLs that
    already contain coordinates).  If that fails, follows HTTP redirects
    (needed for short URLs like maps.app.goo.gl or goo.gl/maps) and retries.

    Returns (None, None) if coordinates cannot be found.
    """
    if not isinstance(url, str) or not url.strip():
        return None, None

    url = url.strip()

    # Fast path: parse without any HTTP request
    lat, lng = _parse(url)
    if lat is not None:
        return lat, lng

    # Slow path: follow redirect then parse
    try:
        resp = session.head(url, allow_redirects=True, timeout=10)
        resolved = resp.url
    except requests.exceptions.RequestException:
        # HEAD blocked by some servers — fall back to GET
        try:
            resp = session.get(url, allow_redirects=True, timeout=10)
            resolved = resp.url
        except requests.exceptions.RequestException:
            return None, None

    wait = random.uniform(*delay_s) if isinstance(delay_s, tuple) else delay_s
    time.sleep(wait)
    return _parse(resolved)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

df = load_file(input_file)

url_column = _find_url_column(df, url_column)

_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
]

session = requests.Session()
session.headers.update({
    'User-Agent'     : random.choice(_USER_AGENTS),
    'Accept'         : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer'        : 'https://www.google.com/',
})

lats, lngs = [], []
n = len(df)

for i, url in enumerate(df[url_column], start=1):
    is_empty = not isinstance(url, str) or not str(url).strip()
    if is_empty:
        lats.append(None)
        lngs.append(None)
        print(f'[{i}/{n}] SKIP  (no link)')
        continue

    lat, lng = extract_coords(url, session, delay)
    lats.append(lat)
    lngs.append(lng)

    if lat is not None:
        print(f'[{i}/{n}] OK   lat={lat:.6f}  long={lng:.6f}')
    else:
        print(f'[{i}/{n}] FAIL  could not parse: {str(url)[:80]}')

df['lat']     = lats
df['long']    = lngs
df['longlat'] = [
    f'{lng:.6f},{lat:.6f}' if lat is not None else ''
    for lat, lng in zip(lats, lngs)
]

out = output_file if output_file else input_file
save_file(df, out)

skipped = sum(1 for v, url in zip(lats, df[url_column])
              if v is None and (not isinstance(url, str) or not str(url).strip()))
failed  = sum(1 for v in lats if v is None) - skipped
print(f'\n{n - skipped - failed}/{n} extracted  |  {skipped} skipped (no link)  |  {failed} failed.')
print(f'Results saved to: {os.path.abspath(out)}')
