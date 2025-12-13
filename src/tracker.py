"""Tracker basado en flujo óptico + matching IoU."""

import numpy as np
import cv2
import logging
from scipy.optimize import linear_sum_assignment
from src.utils import iou_bb

logging.basicConfig(level=logging.INFO, format='[tracker] %(message)s')
logger = logging.getLogger('tracker')

class Track:
    def __init__(self, track_id, bbox, frame_gray, max_corners=50):
        self.id = track_id
        self.bbox = np.array(bbox, dtype=float)  # x1,y1,x2,y2
        self.last_frame = frame_gray
        self.age = 0
        self.missed = 0
        self.kps = None
        self._init_kps(max_corners)

    def _init_kps(self, max_corners=50):
        x1,y1,x2,y2 = map(int, np.round(self.bbox))
        h = max(3, y2 - y1)
        w = max(3, x2 - x1)
        # sanity clamp
        H, W = self.last_frame.shape[:2]
        x1c = max(0, min(x1, W-1))
        y1c = max(0, min(y1, H-1))
        x2c = max(0, min(x1c + w, W-1))
        y2c = max(0, min(y1c + h, H-1))
        if y2c <= y1c or x2c <= x1c:
            self.kps = np.empty((0,1,2), dtype=np.float32)
            return
        roi = self.last_frame[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            self.kps = np.empty((0,1,2), dtype=np.float32)
            return
        kps = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners, qualityLevel=0.01, minDistance=4)
        if kps is None:
            self.kps = np.empty((0,1,2), dtype=np.float32)
            return
        kps[:,:,0] += x1c
        kps[:,:,1] += y1c
        self.kps = kps

    def predict_from_flow(self, new_gray):
        if self.kps is None or len(self.kps) == 0:
            return self.bbox
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame, new_gray, self.kps, None)
        if p1 is None or st is None:
            return self.bbox
        good1 = p1[st.flatten()==1].reshape(-1,2)
        good0 = self.kps[st.flatten()==1].reshape(-1,2)
        if len(good1) < 3:
            return self.bbox
        disp = np.median(good1 - good0, axis=0)
        x1,y1,x2,y2 = self.bbox
        x1 += disp[0]; x2 += disp[0]; y1 += disp[1]; y2 += disp[1]
        return np.array([x1,y1,x2,y2])

    def update(self, bbox, frame_gray):
        self.bbox = np.array(bbox, dtype=float)
        self.last_frame = frame_gray
        self._init_kps()
        self.age += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

class OpticalFlowTracker:
    def __init__(self, iou_thresh=0.3, max_missed=5):
        self.tracks = []
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        logger.info(f'Inicializado OpticalFlowTracker iou_thresh={iou_thresh} max_missed={max_missed}')

    def step(self, frame, detections):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preds = []
        for tr in self.tracks:
            preds.append(tr.predict_from_flow(frame_gray))
        preds = np.array(preds) if preds else np.empty((0,4))
        det_boxes = np.array([d['bbox'] for d in detections]) if len(detections)>0 else np.empty((0,4))

        assigned_tr = set(); assigned_det = set()
        if len(preds)>0 and len(det_boxes)>0:
            cost = np.zeros((len(preds), len(det_boxes)), dtype=float)
            for i,p in enumerate(preds):
                for j,d in enumerate(det_boxes):
                    cost[i,j] = 1.0 - iou_bb(p, d)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r,c in zip(row_ind, col_ind):
                if cost[r,c] <= 1.0 - self.iou_thresh:
                    self.tracks[r].update(det_boxes[c], frame_gray)
                    assigned_tr.add(r); assigned_det.add(c)

        for j in range(len(det_boxes)):
            if j in assigned_det:
                continue
            new_tr = Track(self.next_id, det_boxes[j], frame_gray)
            self.next_id += 1
            self.tracks.append(new_tr)

        for i,tr in list(enumerate(self.tracks)):
            if i not in assigned_tr:
                tr.mark_missed()
            if tr.missed > self.max_missed:
                logger.debug(f'Eliminando track {tr.id} por inactividad (missed={tr.missed})')
                self.tracks.remove(tr)

        out = []
        for tr in self.tracks:
            out.append({'id': tr.id, 'bbox': tr.bbox.tolist()})
        return out
