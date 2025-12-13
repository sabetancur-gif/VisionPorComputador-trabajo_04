"""Detector YOLOv8 simple (ultralytics) adaptado a dataset COCO-like.

Devuelve detecciones en formato {'bbox':[x1,y1,x2,y2], 'score':float, 'class':int}.
"""

import logging
from ultralytics import YOLO
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='[detector] %(message)s')
logger = logging.getLogger('detector')

class YOLODetector:
    """Wrapper ligero para modelos YOLO (ultralytics).

    Args:
        weights (str): ruta al archivo .pt de pesos (o nombre predefinido).
        conf (float): umbral de confianza.
        device (str): 'cpu' o 'cuda'.
        cls_map (dict): opcional, map id->name para clases del dataset.
    """
    def __init__(self, weights='yolov8n.pt', conf=0.25, device='cpu', cls_map=None):
        self.weights = weights
        self.conf = conf
        self.device = device
        self.cls_map = cls_map or {}
        logger.info(f'Inicializando YOLO con {weights}, conf={conf}, device={device}')
        # cargar modelo (ultralytics)
        self.model = YOLO(self.weights)

    def detect(self, image):
        """Detecta objetos en una imagen BGR (numpy array).

        Args:
            image (np.ndarray): imagen BGR.

        Returns:
            List[dict]: lista de detecciones: {'bbox': [x1,y1,x2,y2], 'score':float, 'class':int}
        """
        # ultralytics acepta np.ndarray directamente
        results = self.model(image, device=self.device, conf=self.conf)
        outs = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                # b.xyxy es tensor Nx4, b.conf and b.cls are tensors
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                score = float(b.conf[0].cpu())
                cls = int(b.cls[0].cpu())
                outs.append({'bbox': xyxy, 'score': score, 'class': cls})
        logger.debug(f'Detecciones: {len(outs)}')
        return outs

    def draw(self, img, detections):
        """Dibuja cajas y scores sobre la imagen y la devuelve."""
        out = img.copy()
        for d in detections:
            x1,y1,x2,y2 = map(int,d['bbox'])
            cv2.rectangle(out, (x1,y1),(x2,y2),(0,255,0),2)
            lbl = f"{d['class']}:{d['score']:.2f}"
            if d.get('class') in self.cls_map:
                lbl = f"{self.cls_map[d['class']]}:{d['score']:.2f}"
            cv2.putText(out, lbl, (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        return out
