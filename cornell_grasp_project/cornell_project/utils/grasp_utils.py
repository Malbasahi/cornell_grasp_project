"""
utils/grasp_utils.py
---------------------
Core grasp geometry utilities for the Cornell dataset.

Cornell grasp rectangles are defined by 4 corner points (in pixel coords).
We convert them into dense pixel maps for training:
    - quality map  : 1 inside valid grasp rectangles, 0 elsewhere
    - angle map    : grasp orientation in radians at each pixel
    - width map    : gripper opening width in pixels at each pixel

Evaluation uses the standard Cornell criterion:
    A predicted grasp is CORRECT if:
        1. angle error < 30 degrees
        2. Jaccard index (IoU) with GT rectangle > 0.25
"""

import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max
import cv2


# ─────────────────────────────────────────────────────────────
# GraspRectangle class
# ─────────────────────────────────────────────────────────────

class GraspRectangle:
    """
    A single grasp rectangle defined by 4 corner points.

    The rectangle encodes a parallel-jaw gripper pose:
        - center: midpoint of the rectangle
        - angle:  orientation of the gripper jaw axis (radians)
        - width:  distance between the two jaw midpoints (pixels)
        - height: jaw depth (pixels) — often called 'opening'

    Args:
        points (np.ndarray): 4×2 array of (row, col) corner coordinates
    """

    def __init__(self, points):
        self.points = np.array(points, dtype=np.float32)  # shape (4, 2)

    @property
    def center(self):
        return self.points.mean(axis=0)   # (row, col)

    @property
    def angle(self):
        """
        Grasp angle in radians, measured from the horizontal axis.
        Computed from the vector connecting midpoints of opposite edges.
        """
        # Edge 1: midpoint of points[0]-points[1]
        # Edge 2: midpoint of points[2]-points[3]
        p1 = (self.points[0] + self.points[1]) / 2.0
        p2 = (self.points[2] + self.points[3]) / 2.0
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]
        return np.arctan2(-dy, dx)   # negative because row axis points down

    @property
    def width(self):
        """Distance between the two jaw midpoints (pixels)."""
        p1 = (self.points[0] + self.points[1]) / 2.0
        p2 = (self.points[2] + self.points[3]) / 2.0
        return float(np.linalg.norm(p2 - p1))

    @property
    def height(self):
        """Jaw depth — distance between opposite corners."""
        return float(np.linalg.norm(self.points[1] - self.points[0]))

    def polygon_mask(self, shape):
        """
        Rasterise the rectangle into a boolean pixel mask.

        Args:
            shape (tuple): (H, W) image shape

        Returns:
            mask (np.ndarray): H×W bool array, True inside rectangle
        """
        rr, cc = polygon(self.points[:, 0], self.points[:, 1], shape)
        mask = np.zeros(shape, dtype=bool)
        mask[rr, cc] = True
        return mask

    def iou(self, other):
        """
        Jaccard index (IoU) between this rectangle and another.

        Args:
            other (GraspRectangle): another rectangle

        Returns:
            iou (float): intersection-over-union score
        """
        shape = (480, 640)
        m1 = self.polygon_mask(shape)
        m2 = other.polygon_mask(shape)
        intersection = np.logical_and(m1, m2).sum()
        union        = np.logical_or(m1, m2).sum()
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    def to_array(self):
        """Return [center_r, center_c, angle, width, height]."""
        c = self.center
        return np.array([c[0], c[1], self.angle, self.width, self.height],
                        dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# File parsing
# ─────────────────────────────────────────────────────────────

def load_grasp_rectangles(filepath):
    """
    Load all grasp rectangles from a Cornell label file (cpos.txt / cneg.txt).

    File format: each rectangle is 4 lines, each line = "x y" (col row).
    Rectangles separated by blank lines.

    Args:
        filepath (str): path to *cpos.txt or *cneg.txt

    Returns:
        list of GraspRectangle
    """
    rectangles = []
    points = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if len(points) == 4:
                    # Convert from (col, row) → (row, col)
                    pts = np.array([[p[1], p[0]] for p in points],
                                   dtype=np.float32)
                    rectangles.append(GraspRectangle(pts))
                points = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    points.append((x, y))
                except ValueError:
                    continue

    # Handle last rectangle (no trailing blank line)
    if len(points) == 4:
        pts = np.array([[p[1], p[0]] for p in points], dtype=np.float32)
        rectangles.append(GraspRectangle(pts))

    return rectangles


# ─────────────────────────────────────────────────────────────
# Pixel map generation
# ─────────────────────────────────────────────────────────────

def build_pixel_maps(rectangles, shape=(480, 640)):
    """
    Convert a list of GraspRectangles into dense pixel maps.

    Args:
        rectangles (list): list of GraspRectangle
        shape      (tuple): (H, W) output map shape

    Returns:
        quality (np.ndarray): H×W float32, 1 inside any grasp rectangle
        angle   (np.ndarray): H×W float32, grasp angle (radians) at each pixel
        width   (np.ndarray): H×W float32, gripper width (pixels) at each pixel
    """
    quality = np.zeros(shape, dtype=np.float32)
    angle   = np.zeros(shape, dtype=np.float32)
    width   = np.zeros(shape, dtype=np.float32)

    for rect in rectangles:
        mask          = rect.polygon_mask(shape)
        quality[mask] = 1.0
        angle[mask]   = rect.angle
        width[mask]   = rect.width / shape[1]   # normalise to [0,1]

    return quality, angle, width


# ─────────────────────────────────────────────────────────────
# Prediction → GraspRectangle
# ─────────────────────────────────────────────────────────────

def prediction_to_grasp(quality_map, angle_map, width_map,
                         threshold=0.5, num_peaks=1):
    """
    Convert model output maps into GraspRectangle predictions.

    Strategy:
        1. Threshold quality map to find grasp candidates
        2. Find local maxima (peak_local_max) in thresholded quality
        3. At each peak, read angle and width
        4. Reconstruct the 4 corner points

    Args:
        quality_map (np.ndarray): H×W predicted quality in [0,1]
        angle_map   (np.ndarray): H×W predicted angle (radians)
        width_map   (np.ndarray): H×W predicted width (normalised)
        threshold   (float):      quality threshold
        num_peaks   (int):        number of top predictions to return

    Returns:
        list of GraspRectangle (length = num_peaks)
    """
    H, W = quality_map.shape

    # Find local maxima in the quality map
    q = quality_map.copy()
    q[q < threshold] = 0

    peaks = peak_local_max(q, min_distance=20, num_peaks=num_peaks)

    grasps = []
    for peak in peaks:
        r, c     = peak
        ang      = float(angle_map[r, c])
        w_norm   = float(width_map[r, c])
        w_px     = w_norm * W           # denormalise width
        h_px     = 20.0                 # fixed jaw depth (pixels)

        # Build 4 corners from center, angle, width, height
        cos_a, sin_a = np.cos(ang), np.sin(ang)

        # Half-vectors along jaw axis (width direction) and jaw depth direction
        jaw_vec   = np.array([ cos_a,  sin_a]) * (w_px / 2.0)   # (dc, dr) format
        depth_vec = np.array([-sin_a,  cos_a]) * (h_px / 2.0)

        # 4 corners in (row, col)
        center = np.array([r, c], dtype=np.float32)

        # Convert jaw_vec / depth_vec from (col,row) to (row,col)
        jaw_rc   = np.array([ jaw_vec[1],   jaw_vec[0]])
        depth_rc = np.array([depth_vec[1],  depth_vec[0]])

        p0 = center - jaw_rc + depth_rc
        p1 = center + jaw_rc + depth_rc
        p2 = center + jaw_rc - depth_rc
        p3 = center - jaw_rc - depth_rc

        grasps.append(GraspRectangle(np.stack([p0, p1, p2, p3])))

    return grasps


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def is_grasp_correct(pred, gt_list,
                     iou_threshold=0.25,
                     angle_threshold_deg=30.0):
    """
    Check if a predicted GraspRectangle is correct given a list of
    ground-truth rectangles.

    Correct = exists at least one GT rectangle such that:
        IoU > iou_threshold AND |angle_error| < angle_threshold_deg

    Args:
        pred              (GraspRectangle): predicted grasp
        gt_list           (list): ground-truth GraspRectangle list
        iou_threshold     (float): minimum IoU
        angle_threshold_deg (float): maximum angle error in degrees

    Returns:
        bool
    """
    for gt in gt_list:
        iou = pred.iou(gt)
        angle_err = abs(pred.angle - gt.angle)
        # Handle angle wrap-around (angles are in (-pi/2, pi/2))
        angle_err = min(angle_err, np.pi - angle_err)
        angle_err_deg = np.degrees(angle_err)

        if iou > iou_threshold and angle_err_deg < angle_threshold_deg:
            return True
    return False


def evaluate_predictions(pred_grasps, gt_grasps,
                          iou_threshold=0.25,
                          angle_threshold_deg=30.0):
    """
    Evaluate a batch of predictions against ground truth.

    Args:
        pred_grasps (list of list): outer = samples, inner = predicted GraspRectangles
        gt_grasps   (list of list): outer = samples, inner = GT GraspRectangles
        iou_threshold (float)
        angle_threshold_deg (float)

    Returns:
        accuracy (float): fraction of samples with at least one correct grasp
        results  (list of bool): per-sample correctness
    """
    results = []
    for preds, gts in zip(pred_grasps, gt_grasps):
        correct = False
        for pred in preds:
            if is_grasp_correct(pred, gts, iou_threshold, angle_threshold_deg):
                correct = True
                break
        results.append(correct)

    accuracy = float(np.mean(results)) if results else 0.0
    return accuracy, results
