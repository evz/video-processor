"""
Simple IoU-based multi-object tracker for filtering static false positives.

Inspired by SORT/OC-SORT but simplified for security camera use case.
Tracks detections across frames and calculates displacement to identify
static objects (likely false positives from tree stumps, rocks, etc).
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackedObject:
    """Represents a single tracked object across frames."""
    track_id: int
    category: str
    detection_ids: List[int] = field(default_factory=list)
    positions: List[Tuple[float, float, float, float]] = field(default_factory=list)  # (x, y, w, h)
    frame_numbers: List[int] = field(default_factory=list)
    last_seen: int = 0
    hits: int = 0
    age: int = 0

    @property
    def start_frame(self) -> int:
        return min(self.frame_numbers) if self.frame_numbers else 0

    @property
    def end_frame(self) -> int:
        return max(self.frame_numbers) if self.frame_numbers else 0

    @property
    def centroid(self) -> Tuple[float, float]:
        """Get centroid of most recent bounding box."""
        if not self.positions:
            return (0.0, 0.0)
        x, y, w, h = self.positions[-1]
        return (x + w / 2, y + h / 2)

    @property
    def first_centroid(self) -> Tuple[float, float]:
        """Get centroid of first bounding box."""
        if not self.positions:
            return (0.0, 0.0)
        x, y, w, h = self.positions[0]
        return (x + w / 2, y + h / 2)

    @property
    def last_centroid(self) -> Tuple[float, float]:
        """Get centroid of last bounding box."""
        return self.centroid

    def displacement(self) -> float:
        """
        Calculate total displacement as fraction of frame diagonal.
        Returns Euclidean distance from first to last centroid.
        Coordinates are normalized (0-1), so distance is relative to frame size.
        """
        if len(self.positions) < 2:
            return 0.0

        x1, y1 = self.first_centroid
        x2, y2 = self.last_centroid

        # Distance in normalized coordinates
        # Frame diagonal in normalized space is sqrt(2) ~= 1.414
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)

        # Return as fraction of frame diagonal
        return distance / math.sqrt(2)

    def predict_position(self) -> Tuple[float, float, float, float]:
        """
        Predict next position using simple linear extrapolation.
        Returns the predicted bounding box (x, y, w, h).
        """
        if len(self.positions) < 2:
            return self.positions[-1] if self.positions else (0, 0, 0, 0)

        # Use last two positions for velocity estimation
        x1, y1, w1, h1 = self.positions[-2]
        x2, y2, w2, h2 = self.positions[-1]

        # Simple linear extrapolation
        vx = x2 - x1
        vy = y2 - y1

        return (x2 + vx, y2 + vy, w2, h2)


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: (x, y, width, height) - top-left corner format
        box2: (x, y, width, height) - top-left corner format

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corner format
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class SimpleTracker:
    """
    Simple IoU-based multi-object tracker.

    For each frame:
    1. Predict new positions for existing tracks
    2. Calculate IoU between predictions and new detections
    3. Use Hungarian algorithm for optimal assignment
    4. Update matched tracks, create new tracks for unmatched detections
    5. Mark tracks as lost if not seen for max_age frames
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3):
        """
        Initialize tracker.

        Args:
            iou_threshold: Minimum IoU for detection-track association
            max_age: Maximum frames a track can be lost before deletion
            min_hits: Minimum detections before track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[TrackedObject] = []
        self.completed_tracks: List[TrackedObject] = []
        self.next_id = 1

    def update(self, detections: List[Dict], frame_num: int) -> None:
        """
        Update tracker with new detections for a frame.

        Args:
            detections: List of detection dicts with keys:
                - id: Detection database ID
                - category: Detection category ('1', '2', '3')
                - bbox: (x, y, width, height) tuple
            frame_num: Current frame number
        """
        # Age all tracks
        for track in self.tracks:
            track.age += 1

        if not detections:
            # No detections - mark lost tracks
            self._handle_lost_tracks(frame_num)
            return

        if not self.tracks:
            # No existing tracks - create new ones for all detections
            for det in detections:
                self._create_track(det, frame_num)
            return

        # Build cost matrix using IoU (inverted since we want to maximize IoU)
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets))

        for t_idx, track in enumerate(self.tracks):
            predicted_box = track.predict_position()
            for d_idx, det in enumerate(detections):
                iou = calculate_iou(predicted_box, det['bbox'])
                # Cost = 1 - IoU (lower is better)
                cost_matrix[t_idx, d_idx] = 1.0 - iou

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Track which detections and tracks are matched
        matched_tracks = set()
        matched_dets = set()

        for t_idx, d_idx in zip(row_indices, col_indices):
            # Check if IoU meets threshold
            iou = 1.0 - cost_matrix[t_idx, d_idx]
            if iou >= self.iou_threshold:
                # Also check category matches
                track = self.tracks[t_idx]
                det = detections[d_idx]
                if track.category == det['category']:
                    self._update_track(track, det, frame_num)
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self._create_track(det, frame_num)

        # Handle lost tracks
        self._handle_lost_tracks(frame_num, matched_tracks)

    def _create_track(self, detection: Dict, frame_num: int) -> None:
        """Create a new track from a detection."""
        track = TrackedObject(
            track_id=self.next_id,
            category=detection['category'],
            detection_ids=[detection['id']],
            positions=[detection['bbox']],
            frame_numbers=[frame_num],
            last_seen=frame_num,
            hits=1,
            age=0
        )
        self.tracks.append(track)
        self.next_id += 1

    def _update_track(self, track: TrackedObject, detection: Dict, frame_num: int) -> None:
        """Update an existing track with a new detection."""
        track.detection_ids.append(detection['id'])
        track.positions.append(detection['bbox'])
        track.frame_numbers.append(frame_num)
        track.last_seen = frame_num
        track.hits += 1
        track.age = 0

    def _handle_lost_tracks(self, frame_num: int, matched_tracks: set = None) -> None:
        """Handle tracks that weren't matched in this frame."""
        if matched_tracks is None:
            matched_tracks = set()

        tracks_to_keep = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx in matched_tracks:
                tracks_to_keep.append(track)
            elif track.age <= self.max_age:
                tracks_to_keep.append(track)
            else:
                # Track is lost - move to completed if it had enough hits
                if track.hits >= self.min_hits:
                    self.completed_tracks.append(track)

        self.tracks = tracks_to_keep

    def finalize(self) -> None:
        """
        Finalize tracking - move all remaining tracks to completed.
        Call this after processing all frames.
        """
        for track in self.tracks:
            if track.hits >= self.min_hits:
                self.completed_tracks.append(track)
        self.tracks = []

    def get_completed_tracks(self) -> List[TrackedObject]:
        """Return all completed tracks."""
        return self.completed_tracks


def group_detections_by_frame(detections) -> Dict[int, List[Dict]]:
    """
    Group detection objects by frame number.

    Args:
        detections: QuerySet of Detection objects with frame relation

    Returns:
        Dict mapping frame_number to list of detection dicts
    """
    frames_dict = defaultdict(list)
    for det in detections:
        det_dict = {
            'id': det.id,
            'category': det.category,
            'bbox': (det.x_coord, det.y_coord, det.box_width, det.box_height)
        }
        frames_dict[det.frame.frame_number].append(det_dict)
    return frames_dict
