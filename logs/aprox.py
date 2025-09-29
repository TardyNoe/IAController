import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= Parameters you can tweak =========
in_path = Path("logs/drive_points_20250928_221916.csv")
out_path = Path("track_equal_spaced_closed.csv")

apply_smoothing = True
smooth_coeff = 0.01        # 5% of track length -> heavier number = more smoothing
target_delta = None        # set a spacing (same units as your x/y). If None -> median original step
include_start_point = True # keep the original first point in the resampled set

# ========= Helpers =========
def pick_xy_columns(df):
    lower_cols = {c.lower(): c for c in df.columns}
    for a, b in [("x","y"),("lon","lat"),("long","lat"),
                 ("longitude","latitude"),("easting","northing")]:
        if a in lower_cols and b in lower_cols:
            return lower_cols[a], lower_cols[b]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        raise ValueError("Couldn't find two numeric columns to use as coordinates.")
    return num_cols[0], num_cols[1]

def moving_average(arr, win):
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(padded, kernel, mode="valid")

def close_loop(x, y):
    # Ensure last equals first
    if len(x) >= 1:
        x[-1] = x[0]
        y[-1] = y[0]
    return x, y

def cumulative_arclength(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, seg

def resample_equal_arc(x, y, delta, include_start=True):
    """
    Return (xr, yr) resampled along arc length with constant delta.
    The path is treated as closed (last == first) already.
    """
    s, _ = cumulative_arclength(x, y)
    total = s[-1]
    if total == 0:
        return np.array([x[0]]), np.array([y[0]])

    # Number of intervals (round to ensure near-exact coverage)
    n_intervals = max(3, int(round(total / delta)))
    uniform_s = np.linspace(0, total, n_intervals + 1)

    # Interpolate x(s), y(s)
    xr = np.interp(uniform_s, s, x)
    yr = np.interp(uniform_s, s, y)

    # For a closed loop, drop the duplicate last point if you want a simple polygon
    # whose first == last after plotting. We'll return without the duplicate and
    # the caller can decide how to plot.
    xr = xr[:-1] if include_start else xr[1:-1]
    yr = yr[:-1] if include_start else yr[1:-1]
    return xr, yr

# ========= Load & prep =========
df = pd.read_csv(in_path)
x_col, y_col = pick_xy_columns(df)
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
df = df.dropna(subset=[x_col, y_col]).reset_index(drop=True)

x = df[x_col].to_numpy()
y = df[y_col].to_numpy()

# Optional smoothing
if apply_smoothing:
    n = len(df)
    win = int(max(7, min(301, round(n * smooth_coeff))))
    if win % 2 == 0:
        win += 1
    x = moving_average(x, win)
    y = moving_average(y, win)

# Force perfect closure before resampling
x, y = close_loop(x.copy(), y.copy())

# Pick spacing
if target_delta is None:
    # Use median original step as a reasonable, robust spacing
    _, seg = cumulative_arclength(x, y)
    # avoid zeros if there are duplicates
    med = np.median(seg[seg > 0]) if np.any(seg > 0) else 1.0
    target_delta = float(med) if med > 0 else 1.0

# Resample at constant arc-length spacing
xr, yr = resample_equal_arc(x, y, target_delta, include_start=include_start_point)

# Make a DataFrame and also add a closing copy of the first point for clarity if you want a closed CSV
res_df = pd.DataFrame({x_col: xr, y_col: yr})
# Optional: write with an explicit closing point (uncomment next two lines if you prefer last==first in the CSV)
# res_df = pd.concat([res_df, res_df.iloc[[0]]], ignore_index=True)

res_df.to_csv(out_path, index=False)
print(f"Equal-spaced, closed track saved to: {out_path} (spacing ≈ {target_delta:.3f} units)")

# ========= Plot: raw vs. equal-spaced =========
plt.figure(figsize=(8,8))
plt.plot(-df[y_col], -df[x_col],'o', label="Raw", alpha=0.5, linewidth=0.8)
plt.plot( -np.r_[yr, yr[0]],-np.r_[xr, xr[0]],'o', label="Equal-Spaced (Closed)", linewidth=1.2)
plt.axis("equal")
plt.title("Track: Raw vs. Equal-Spaced Closed Resample")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend()
plt.tight_layout()
plt.show()
