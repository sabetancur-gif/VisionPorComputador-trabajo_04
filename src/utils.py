"""Funciones utilitarias: IoU, conversiones y helpers de IO simples."""

import numpy as np

def iou_bb(a, b):
    """Calcula IoU entre dos cajas [x1,y1,x2,y2]."""
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
    area_a = max(0, ax2-ax1)*max(0, ay2-ay1)
    area_b = max(0, bx2-bx1)*max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-9)

def to_int_bbox(bbox):
    """Convierte bbox float a int con formato seguro y clamp >=0."""
    return [int(round(max(0, x))) for x in bbox]

def coco_to_xyxy(bbox):
    """Convierte bbox COCO [x,y,w,h] a [x1,y1,x2,y2]."""
    x,y,w,h = bbox
    return [x, y, x+w, y+h]

def clamp_bbox_xyxy(bbox, w=None, h=None):
    """Clamp de una bbox xyxy a los límites (opcional)."""
    x1,y1,x2,y2 = bbox
    if w is not None:
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
    if h is not None:
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
    return [x1,y1,x2,y2]
