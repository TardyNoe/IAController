import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input and output paths
left_csv = "/home/noe/Simulator/autoverse-linux/Linux/Autoverse/Saved/Environments/YasMarina/Data/yas_tbs_left.csv"
right_csv = "/home/noe/Simulator/autoverse-linux/Linux/Autoverse/Saved/Environments/YasMarina/Data/yas_tbs_right_2.csv"
out = "centerline_xy.csv"

# --- Utility functions ---
def ensure_same_direction(left, right):
    """Ensure polylines are oriented consistently."""
    ls, le = left[0], left[-1]
    rs, re = right[0], right[-1]
    same = np.linalg.norm(ls - rs) + np.linalg.norm(le - re)
    cross = np.linalg.norm(ls - re) + np.linalg.norm(le - rs)
    if cross < same:
        right = right[::-1].copy()
    return right

def poly_params(poly):
    """Arclength and normalized parameter along polyline."""
    d = np.diff(poly, axis=0)
    seglens = np.sqrt((d**2).sum(axis=1))
    s = np.concatenate(([0.0], np.cumsum(seglens)))
    total = s[-1] if s[-1] > 0 else 1.0
    return s, s / total

def vectorized_project_points_onto_polyline(points, poly):
    """Closest projections of points onto polyline segments (vectorized)."""
    A = poly[:-1]   # (S,2)
    B = poly[1:]    # (S,2)
    AB = B - A
    AB_len2 = (AB**2).sum(axis=1)
    N = points.shape[0]
    S = A.shape[0]

    best_d2 = np.full(N, np.inf)
    best_proj = np.zeros_like(points)

    for i in range(S):
        a = A[i]
        ab = AB[i]
        denom = AB_len2[i]
        if denom == 0:
            proj = np.tile(a, (N, 1))
        else:
            ap = points - a
            t = (ap @ ab) / denom
            t = np.clip(t, 0.0, 1.0)
            proj = a + np.outer(t, ab)
        d = points - proj
        d2 = (d**2).sum(axis=1)
        mask = d2 < best_d2
        best_d2[mask] = d2[mask]
        best_proj[mask] = proj[mask]
    return best_proj

# --- Load data ---
left_df = pd.read_csv(left_csv).iloc[:, :2]
left_df.columns = ["x", "y"]
right_df = pd.read_csv(right_csv).iloc[:, :2]
right_df.columns = ["x", "y"]

left = left_df.to_numpy(float)
right = right_df.to_numpy(float)

# Align direction
right = ensure_same_direction(left, right)

# Parameters along each boundary
_, left_t = poly_params(left)
_, right_t = poly_params(right)

# Midpoints: left points to right polyline
projL_on_R = vectorized_project_points_onto_polyline(left, right)
mids_from_left = (left + projL_on_R) / 2.0

# Midpoints: right points to left polyline
projR_on_L = vectorized_project_points_onto_polyline(right, left)
mids_from_right = (right + projR_on_L) / 2.0

# Merge and sort by normalized arclength parameters
params = np.concatenate([left_t, right_t])
center_pts = np.vstack([mids_from_left, mids_from_right])
order = np.argsort(params)
centerline = center_pts[order]

# Deduplicate
dedup = [centerline[0]]
for p in centerline[1:]:
    if np.linalg.norm(p - dedup[-1]) > 1e-8:
        dedup.append([(p[0]/100)-2.840,(p[1]/100)-2.580])
centerline = np.array(dedup[1:])

# --- Save and plot ---
pd.DataFrame(centerline, columns=["x", "y"]).to_csv(out, index=False)
print(f"Centerline saved to {out} with {len(centerline)} points")

plt.figure(figsize=(8, 6))
plt.plot(left[:,0], left[:,1], "o", label="Left limit")
plt.plot(right[:,0], right[:,1], "o", label="Right limit")
plt.plot(centerline[:,0], centerline[:,1], ".", label="Centerline")
plt.axis("equal")
plt.legend()
plt.title("Centerline via Closest-Segment Midpoints")
plt.show()
