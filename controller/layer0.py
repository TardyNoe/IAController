import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
def normalize_observation(obs, bounds):
    normalized_obs = (obs - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    np.clip(normalized_obs, 0.0, 1.0, out=normalized_obs)
    return normalized_obs.astype(np.float32)
def load_track(center_csv_path: str = "center.csv", limits_csv_path: str = "limits.csv"):
    center_df = pd.read_csv(center_csv_path)
    if {"x", "y"}.issubset(center_df.columns):
        center = center_df[["x", "y"]].to_numpy(dtype=float)
    else:
        center = center_df.iloc[:, :2].to_numpy(dtype=float)

    left_bounds = None
    right_bounds = None
    try:
        lim_df = pd.read_csv(limits_csv_path)
        cols = set(lim_df.columns.str.lower())
        if {"xl", "yl", "xr", "yr"}.issubset(cols):
            def col(name):
                for c in lim_df.columns:
                    if c.lower() == name:
                        return c
                return name
            left_bounds = lim_df[[col("xl"), col("yl")]].to_numpy(dtype=float)
            right_bounds = lim_df[[col("xr"), col("yr")]].to_numpy(dtype=float)
    except Exception:
        pass

    if left_bounds is None or right_bounds is None:
        width = 8.0
        n = len(center)
        left_bounds = np.zeros_like(center)
        right_bounds = np.zeros_like(center)
        for i in range(n):
            p_prev = center[(i - 1) % n]
            p_next = center[(i + 1) % n]
            tangent = p_next - p_prev
            if np.allclose(tangent, 0):
                tangent = np.array([1.0, 0.0])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
            normal = np.array([-tangent[1], tangent[0]])
            left_bounds[i]  = center[i] + normal * (width * 0.5)
            right_bounds[i] = center[i] - normal * (width * 0.5)

    return center, left_bounds, right_bounds

class layer0class:
    def __init__(self,center_csv_path):
        self.center_csv_path = center_csv_path#"track_data/Yas_center.csv"
        self.limits_csv_path = "track_data/Yas_limits.csv"
        self.center, self.left_bounds, self.right_bounds = load_track(self.center_csv_path, self.limits_csv_path)
        self._prepare_track_geometry()
        self.bounds = np.array([
            [-20.0, 20.0],    # steer
            [-1.0, 1.0],    # pedals
            [0.0, 100.0],   # speed
            [0.0, 100.0],   # completion %
            [-5.0, 5.0],    # offset
            [-180.0, 180.0],# track orientation
            [-180.0, 180.0],# next track orientation 1
            [-180.0, 180.0],# next track orientation 2
            [-180.0, 180.0],# next track orientation 3
            [-180.0, 180.0],# next track orientation 4
            [-180.0, 180.0],# next track orientation 5
            [-180.0, 180.0],# next track orientation 6
            [-180.0, 180.0],# next track orientation 7
            [-180.0, 180.0],# next track orientation 8
            [-180.0, 180.0],# next track orientation 9
            [-180.0, 180.0], # next track orientation 10
            [1, 5],
            [0.5, 1.5],
            [0.0,5.0]
        ], dtype=np.float32)

    def load_track(self):        
        center, left_bounds, right_bounds = load_track(self.center_csv_path, self.limits_csv_path)
        return center, left_bounds, right_bounds
    
    def _prepare_track_geometry(self):
        """
        Prepare centerline polyline geometry: ensure closed loop, segments, lengths,
        cumulative distances, segment orientations, and KD-tree over segment midpoints.
        """
        pts = np.asarray(self.center, dtype=np.float32)
        # Ensure closed loop (append first point if not closed)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        self.track_points = pts                               # shape (N,2)
        self.num_points = len(pts)
        self.num_segments = self.num_points - 1               # segments between i and i+1

        # Segment vectors and lengths
        seg_vecs = self.track_points[1:] - self.track_points[:-1]
        seg_len = np.linalg.norm(seg_vecs, axis=1)
        # Guard against degenerate segments
        seg_len = np.where(seg_len < 1e-9, 1e-9, seg_len)

        self.segment_vectors = seg_vecs                        # (S,2)
        self.segment_lengths = seg_len                         # (S,)
        self.segment_lengths_squared = seg_len**2

        # Cumulative distances at vertices (length N)
        self.cumulative_distances = np.concatenate([[0.0], np.cumsum(self.segment_lengths, dtype=np.float64)])
        self.total_length = float(self.cumulative_distances[-1])

        # Segment midpoints (for KD-tree queries)
        self.segment_midpoints = self.track_points[:-1] + 0.5 * self.segment_vectors

        # Segment orientations in degrees (heading of each segment)
        self.segment_orientations = np.degrees(np.arctan2(self.segment_vectors[:, 1], self.segment_vectors[:, 0])).astype(np.float32)

        # KD-tree over segment midpoints
        self.kdtree = cKDTree(self.segment_midpoints)

        # Small cache (optional)
        self.last_min_index = 0

    def calculate_completion_and_offset_with_future_orientations(
        self, point, future_steps, search_radius: float = 10.0
        ):
        """
        Calculate completion percentage, signed lateral offset (right=+), current
        segment orientation (deg), and future orientations (deg) at the closest
        projected point on the centerline to `point`.
        """
        
        point = np.asarray(point, dtype=np.float32)

        # Query KD-Tree for nearby segment midpoints
        indices = self.kdtree.query_ball_point(point, r=search_radius)
        if not indices:
            # Fallback: consider all segments (safe but slower)
            indices = range(self.num_segments)

        # Vector from segment start to the point, per candidate segment
        starts = self.track_points[:-1][indices]                          # (K,2)
        seg_vecs = self.segment_vectors[indices]                          # (K,2)
        seg_len2 = self.segment_lengths_squared[indices]                  # (K,)
        p_vecs = point - starts                                           # (K,2)

        # Projection scalar t in [0,1] along each segment
        t = np.clip(np.einsum('ij,ij->i', p_vecs, seg_vecs) / seg_len2, 0.0, 1.0)  # (K,)

        # Projected points
        proj_pts = starts + t[:, None] * seg_vecs                         # (K,2)

        # Distances from the query point to projected points
        dists = np.linalg.norm(point - proj_pts, axis=1)                  # (K,)

        # Closest segment among candidates
        kmin = int(np.argmin(dists))
        min_index = int(indices[kmin])
        min_dist = float(dists[kmin])
        closest_proj_point = proj_pts[kmin]

        # Cache for potential reuse
        self.last_min_index = min_index

        # Completion distance and percentage
        # distance up to the start of the segment + along-segment distance to projection
        completion_distance = (
            float(self.cumulative_distances[min_index]) +
            float(np.linalg.norm(closest_proj_point - self.track_points[min_index]))
        )
        completion_percentage = (completion_distance / max(1e-9, self.total_length)) * 100.0

        # Signed lateral offset: right = positive, left = negative (following your rule)
        seg_vec = self.segment_vectors[min_index]
        p_vec = point - self.track_points[min_index]
        # 2D cross product z-component
        cross_z = seg_vec[0]*p_vec[1] - seg_vec[1]*p_vec[0]
        side_right = (cross_z < 0.0)
        offset = +min_dist if side_right else -min_dist

        # Current track orientation (deg)
        track_orientation_degrees = float(self.segment_orientations[min_index])

        # Future orientations
        future_indices = np.clip(min_index + np.asarray(future_steps, dtype=int), 0, self.num_segments - 1)
        future_orientations = self.segment_orientations[future_indices].tolist()

        return float(completion_percentage), float(offset), track_orientation_degrees, future_orientations
    
    def compute_obs_vector(self, input_steer, input_pedals,car_speed , car_position_x, car_position_y, car_orientation,
                           mu,break_force_factor,center_proxi):
        car_orientation = np.degrees(car_orientation)
        completion, offset, track_orientation, next_track_orientation = self.calculate_completion_and_offset_with_future_orientations(
        point=np.array([car_position_x, car_position_y], dtype=np.float32),
        future_steps=[5, 10, 15, 30, 60, 80, 130, 160, 180, 200],  
        search_radius=12.0             
        )
        for i in range(len(next_track_orientation)):
            next_track_orientation[i] = ((car_orientation-next_track_orientation[i])+180)%360 - 180
        next_track_orientation = np.array(next_track_orientation)
        track_orientation = ((car_orientation-track_orientation)+180)%360 - 180
        
        observation = np.array([
            input_steer,
            input_pedals,
            car_speed,
            completion,
            offset,
            track_orientation,
            next_track_orientation[0],
            next_track_orientation[1],
            next_track_orientation[2],
            next_track_orientation[3],
            next_track_orientation[4],
            next_track_orientation[5],
            next_track_orientation[6],
            next_track_orientation[7],
            next_track_orientation[8],
            next_track_orientation[9],
            mu,
            break_force_factor,
            center_proxi
            ], 
            dtype=np.float32)


        norm_obs = normalize_observation(observation, self.bounds)

        return norm_obs, completion, offset, track_orientation