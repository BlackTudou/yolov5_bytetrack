"""
ByteTrack implementation for multi-object tracking
Based on ByteTrack: Multi-Object Tracking by Associating Every Detection Box
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import torch


class STrack:
    """
    Single object tracking
    """
    def __init__(self, tlwh, score, cls_id):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.cls_id = cls_id

        # Initialize Kalman filter (simplified)
        self.mean = np.zeros((4,), dtype=np.float32)
        tlwh_array = np.asarray(tlwh, dtype=np.float32)
        self.mean[:2] = tlwh_array[:2] + tlwh_array[2:] / 2  # cx, cy
        self.mean[2:] = tlwh_array[2:]  # w, h
        self.covariance = np.eye(4, dtype=np.float32) * 2

        # Track state
        self.track_id = 0
        self.frame_id = 0
        self.track_len = 0
        self.state = 'New'
        self.is_activated = False
        self.history = deque(maxlen=30)

    def update(self, new_track, frame_id):
        """Update track with new detection"""
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean[:2] = new_tlwh[:2] + new_tlwh[2:] / 2
        self.mean[2:] = new_tlwh[2:]

        self.tlwh = new_tlwh
        self.score = new_track.score

        # Update history
        self.history.append(new_tlwh)

    def activate(self, frame_id, track_id):
        """Activate track"""
        self.frame_id = frame_id
        self.track_id = track_id
        self.is_activated = True
        self.state = 'Tracked'

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate lost track"""
        self.frame_id = frame_id
        self.track_len = 0
        self.track_id = new_id if new_id else self.track_id

        new_tlwh = new_track.tlwh
        self.mean[:2] = new_tlwh[:2] + new_tlwh[2:] / 2
        self.mean[2:] = new_tlwh[2:]

        self.tlwh = new_tlwh
        self.score = new_track.score
        self.is_activated = True
        self.state = 'Tracked'

    def mark_lost(self):
        """Mark track as lost"""
        self.state = 'Lost'

    def mark_removed(self):
        """Mark track as removed"""
        self.state = 'Removed'

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr"""
        if not isinstance(tlwh, np.ndarray):
            tlwh = np.array(tlwh)
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh"""
        if not isinstance(tlbr, np.ndarray):
            tlbr = np.array(tlbr)
        ret = tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    def to_tlbr(self):
        """Get tlbr coordinates"""
        return self.tlwh_to_tlbr(self.tlwh)

    def to_xyah(self):
        """Get xyah coordinates (x, y, aspect ratio, height)"""
        ret = self.tlwh.copy()
        ret[2] /= ret[3]  # aspect ratio
        ret[:2] += ret[3] / 2  # center
        return ret


class ByteTracker:
    """
    ByteTrack Tracker
    """
    def __init__(self, frame_rate=30, track_thresh=0.5, high_thresh=0.6,
                 match_thresh=0.8, track_buffer=30, min_box_area=10):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.max_time_lost = frame_rate / 30.0 * self.track_buffer

    def update(self, detections, frame_id):
        """
        Update tracker with new detections

        Args:
            detections: List of detections, each as [tlwh, score, class_id]
            frame_id: Current frame number

        Returns:
            List of tracked objects
        """
        self.frame_id = frame_id

        # Convert detections to STracks
        detections = [STrack(tlwh, score, cls_id)
                      for tlwh, score, cls_id in detections]

        # Separate detections into high and low score
        detections_high = [d for d in detections if d.score >= self.high_thresh]
        detections_low = [d for d in detections if d.score < self.high_thresh and
                          d.score >= self.track_thresh]

        # Update tracked stracks
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]
        lost_stracks = [t for t in self.lost_stracks if t.is_activated]

        # Associate high score detections with tracked stracks
        matched, unmatched_track_ids = self.associate_detections_to_trackers(
            tracked_stracks, detections_high)

        # Update matched tracks
        for track_idx, det_idx in matched:
            track = tracked_stracks[track_idx]
            det = detections_high[det_idx]
            track.update(det, self.frame_id)

        # Reactivate lost tracks with high score detections
        reactivated, unmatched_lost_ids = self.associate_detections_to_trackers(
            lost_stracks, detections_high)

        for track_idx, det_idx in reactivated:
            track = lost_stracks[track_idx]
            det = detections_high[det_idx]
            track.re_activate(det, self.frame_id, new_id=False)

        # Get unmatched high score detections
        unmatched_det_ids = [i for i in range(len(detections_high))
                             if i not in [m[1] for m in matched] and
                             i not in [r[1] for r in reactivated]]

        # Associate unmatched tracks with low score detections
        lost_tracks_to_match = []
        for i in unmatched_track_ids:
            if 0 <= i < len(tracked_stracks):
                lost_tracks_to_match.append(tracked_stracks[i])

        matched_a = []
        if len(lost_tracks_to_match) > 0 and len(detections_low) > 0:
            iou_dist = self.compute_iou_distance(lost_tracks_to_match, detections_low)
            matched_a, unmatched_tracks_a, unmatched_dets_a = \
                self.linear_assignment(iou_dist, thresh=self.match_thresh)

            for itracked, idet in matched_a:
                t = lost_tracks_to_match[itracked]
                t.re_activate(detections_low[idet], self.frame_id, new_id=False)

        # Create new tracks for unmatched high score detections
        unmatched_dets = [detections_high[i] for i in unmatched_det_ids]
        for d in unmatched_dets:
            if d.score >= self.high_thresh:
                d.activate(self.frame_id, len(self.tracked_stracks))
                self.tracked_stracks.append(d)

        # Update lost and removed tracks
        matched_track_indices_in_lost = set()
        for itracked, idet in matched_a:
            matched_track_indices_in_lost.add(itracked)

        for i, t in enumerate(lost_tracks_to_match):
            if i not in matched_track_indices_in_lost:
                if self.frame_id - t.frame_id > self.max_time_lost:
                    t.mark_removed()
                    self.removed_stracks.append(t)
                else:
                    t.mark_lost()
                    self.lost_stracks.append(t)

        # Update tracking state
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'Tracked']
        self.lost_stracks = [t for t in self.lost_stracks if t.state == 'Lost']

        # Return tracked objects
        return [t for t in self.tracked_stracks if t.is_activated]

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        tlbr1 = STrack.tlwh_to_tlbr(box1)
        tlbr2 = STrack.tlwh_to_tlbr(box2)

        inter_tl = np.maximum(tlbr1[:2], tlbr2[:2])
        inter_br = np.minimum(tlbr1[2:], tlbr2[2:])
        inter_area = np.prod(np.clip(inter_br - inter_tl, 0, None))

        area1 = np.prod(tlbr1[2:] - tlbr1[:2])
        area2 = np.prod(tlbr2[2:] - tlbr2[:2])

        iou = inter_area / (area1 + area2 - inter_area + 1e-16)
        return iou

    def compute_iou_distance(self, tracks, detections):
        """Compute IoU distance matrix"""
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - self.compute_iou(track.tlwh, det.tlwh)

        return cost_matrix

    def associate_detections_to_trackers(self, tracks, detections):
        """Associate detections to trackers"""
        if len(tracks) == 0:
            return [], list(range(len(detections)))

        if len(detections) == 0:
            return [], []

        # Compute IoU distance
        iou_dist = self.compute_iou_distance(tracks, detections)

        # Linear assignment
        matched, unmatched_tracks, unmatched_dets = \
            self.linear_assignment(iou_dist, thresh=self.match_thresh)

        return matched, unmatched_tracks

    def linear_assignment(self, cost_matrix, thresh):
        """
        Solve linear assignment problem
        Returns: matched_indices, unmatched_a, unmatched_b
        """
        if cost_matrix.size == 0:
            return [], [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        cost_matrix[cost_matrix > thresh] = thresh + 1e-5

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matched.append((r, c))

        unmatched_a = [i for i in range(cost_matrix.shape[0])
                       if i not in [m[0] for m in matched]]
        unmatched_b = [i for i in range(cost_matrix.shape[1])
                       if i not in [m[1] for m in matched]]

        return matched, unmatched_a, unmatched_b


if __name__ == '__main__':
    # Simple test
    tracker = ByteTracker()
    detections = [
        [[10, 10, 50, 100], 0.9, 0],  # person
        [[100, 20, 50, 100], 0.8, 0],  # person
    ]
    tracked = tracker.update(detections, 0)
    print(f"Tracked {len(tracked)} objects")

