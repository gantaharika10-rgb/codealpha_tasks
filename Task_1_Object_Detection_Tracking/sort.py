
"""
sort.py - Simple Online and Realtime Tracking (SORT)
 
A self-contained implementation of the SORT algorithm
(Bewley et al., 2016) using only numpy and scipy.
 
Each tracked object has a Kalman filter that predicts its position.
New detections are matched to existing tracks using the Hungarian
algorithm on an IoU (Intersection over Union) cost matrix.
"""
 
import numpy as np
from scipy.optimize import linear_sum_assignment
 
 
def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes in [x1, y1, x2, y2] format.
 
    bb_test: (N, 4), bb_gt: (M, 4)  ->  returns (N, M)
    """
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float64)
 
    bb_gt = np.expand_dims(bb_gt, 0)      # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)
 
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
 
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
 
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
 
    union = area_test + area_gt - inter
    return inter / np.maximum(union, 1e-9)
 
 
def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, area, aspect_ratio] as a (4, 1) column."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-9)
    return np.array([cx, cy, s, r], dtype=np.float64).reshape((4, 1))
 
 
def x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Kalman state -> [x1, y1, x2, y2] as a (4,) array."""
    s = max(float(x[2, 0]), 1e-6)
    r = max(float(x[3, 0]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    cx, cy = float(x[0, 0]), float(x[1, 0])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)
 
 
class KalmanBoxTracker:
    """Tracks a single object using a constant-velocity Kalman filter.
 
    State vector: [cx, cy, s, r, vcx, vcy, vs]
    """
 
    count = 0  # global ID counter
 
    def __init__(self, bbox: np.ndarray, cls_id: int = -1, conf: float = 0.0):
        # State transition matrix (constant velocity model)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
 
        # Measurement matrix (we observe cx, cy, s, r)
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
 
        # Measurement noise
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
 
        # State covariance (high uncertainty for unobserved velocities)
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
 
        # Process noise
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
 
        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)
 
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
 
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
 
        self.cls_id = int(cls_id)
        self.conf = float(conf)
 
    def predict(self) -> np.ndarray:
        """Advance the state one frame and return the predicted box."""
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
 
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
 
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_bbox(self.x)
 
    def update(self, bbox: np.ndarray, cls_id: int = -1, conf: float = 0.0) -> None:
        """Correct the state with a matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls_id = int(cls_id)
        self.conf = float(conf)
 
        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
 
    def get_state(self) -> np.ndarray:
        return x_to_bbox(self.x)
 
 
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """Match detections to predicted tracker boxes with the Hungarian algorithm.
 
    Returns (matches [K,2], unmatched_detections, unmatched_trackers).
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))
    if len(detections) == 0:
        return (np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=int),
                np.arange(len(trackers)))
 
    iou_matrix = iou_batch(detections, trackers)
 
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)
 
    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]
 
    matches = []
    for d, t in matched_indices:
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detections.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append([d, t])
 
    matches = np.array(matches, dtype=int).reshape(-1, 2)
    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)
 
 
class Sort:
    """SORT multi-object tracker."""
 
    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
 
    def reset(self) -> None:
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
 
    def update(self, dets: np.ndarray = None) -> np.ndarray:
        """Update with detections of shape (N, 6): [x1, y1, x2, y2, conf, cls].
 
        Call once per frame (even with no detections - pass an empty array).
        Returns an array (M, 7): [x1, y1, x2, y2, track_id, cls, conf].
        """
        self.frame_count += 1
 
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float64)
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
 
        # 1) Predict new locations of existing tracks
        predicted = np.zeros((len(self.trackers), 4))
        to_delete = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            predicted[i, :] = pos
            if np.any(np.isnan(pos)):
                to_delete.append(i)
        for i in reversed(to_delete):
            self.trackers.pop(i)
        predicted = np.delete(predicted, to_delete, axis=0)
 
        # 2) Associate detections with predictions
        matched, unmatched_dets, _ = associate_detections_to_trackers(
            dets[:, :4], predicted, self.iou_threshold
        )
 
        # 3) Update matched trackers
        for d, t in matched:
            self.trackers[t].update(dets[d, :4], dets[d, 5], dets[d, 4])
 
        # 4) Create new trackers for unmatched detections
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4], dets[d, 5], dets[d, 4]))
 
        # 5) Collect output and drop dead tracks
        results = []
        alive = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                box = trk.get_state()
                results.append([*box, trk.id, trk.cls_id, trk.conf])
            if trk.time_since_update <= self.max_age:
                alive.append(trk)
        self.trackers = alive
 
        if results:
            return np.array(results, dtype=np.float64)
        return np.empty((0, 7), dtype=np.float64)