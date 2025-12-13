"""Módulo de evaluación: métricas simples para detección y scaffolding para métricas MOT."""

import logging
from src.utils import iou_bb

logging.basicConfig(level=logging.INFO, format='[eval] %(message)s')
logger = logging.getLogger('eval')

def detection_metrics(dets, gts, iou_thresh=0.5):
    """
    dets: dict key -> list of bboxes (each bbox [x1,y1,x2,y2])
    gts: dict same keys -> list of bboxes (groundtruth)
    """
    TP = 0; FP = 0; FN = 0
    for k, gt_boxes in gts.items():
        det_boxes = dets.get(k, [])
        matched_gt = set()
        for db in det_boxes:
            best_iou = 0; best_g = None
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = iou_bb(db, gb)
                if iou > best_iou:
                    best_iou = iou; best_g = i
            if best_iou >= iou_thresh:
                TP += 1
                matched_gt.add(best_g)
            else:
                FP += 1
        FN += len(gt_boxes) - len(matched_gt)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    logger.info(f'Detección: TP={TP} FP={FP} FN={FN} P={precision:.3f} R={recall:.3f} F1={f1:.3f}')
    return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': precision, 'recall': recall, 'f1': f1}

def tracking_metrics_from_tracks(gt_mot, hyp_mot):
    logger.info('Para evaluación de tracking use motmetrics: construir DataFrames en formato MOT y luego usar el Accumulator.')
    return {}
