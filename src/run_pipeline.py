"""
Script principal que conecta detección, tracking y generación de resultados,
adaptado a JSON COCO-like en data/sample.

Uso típico:
python run_pipeline.py --images data/sample/images --annotations data/sample/cis_val_annotations.json --out results/sample --detector yolov8n.pt --device cpu --n 200
"""

import os
import argparse
import glob
import cv2
import logging
import pandas as pd
from tqdm import tqdm
from src.detector import YOLODetector
from src.tracker import OpticalFlowTracker
from src.eval import detection_metrics
from src.utils import to_int_bbox
from src.data_loader import get_image_list_from_json, load_annotations

logging.basicConfig(level=logging.INFO, format='[run_pipeline] %(message)s')
logger = logging.getLogger('run_pipeline')

def write_csv_detections(out_csv, detections_by_frame):
    rows = []
    for i, dets in detections_by_frame.items():
        for d in dets:
            x1,y1,x2,y2 = to_int_bbox(d['bbox'])
            rows.append({'frame': i, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'score': d.get('score', 0.0), 'class': d.get('class', -1)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

def write_csv_tracks(out_csv, tracks_by_frame):
    rows = []
    for i, trks in tracks_by_frame.items():
        for t in trks:
            x1,y1,x2,y2 = to_int_bbox(t['bbox'])
            rows.append({'frame': i, 'id': t['id'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

def find_local_sample_dir():
    candidates = [
        'data/sample',
        'sample_images',
        'examples/images',
        'assets/images',
        'data/images',
    ]
    for c in candidates:
        if os.path.isdir(c):
            files = [f for f in glob.glob(os.path.join(c, '*')) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            if len(files) > 0:
                return c
    return None

def run_image_sequence(image_tuples, out_dir, detector_weights, device, fps=2, anns_by_image=None):
    os.makedirs(out_dir, exist_ok=True)
    logger.info('Inicializando detector y tracker...')
    det = YOLODetector(weights=detector_weights, device=device)
    tracker = OpticalFlowTracker()

    detections_by_frame = {}
    tracks_by_frame = {}
    vis_frames = []
    gt_mapping = {}

    for idx, (image_id, imgf) in enumerate(tqdm(image_tuples, desc='Frames')):
        frame = cv2.imread(imgf)
        if frame is None:
            logger.warning(f'No se pudo leer {imgf}, saltando')
            continue

        dets = det.detect(frame)
        tracks = tracker.step(frame, dets)

        detections_by_frame[idx] = [{'bbox': d['bbox'], 'score': d['score'], 'class': d['class']} for d in dets]
        tracks_by_frame[idx] = [{'id': t['id'], 'bbox': t['bbox']} for t in tracks]

        # groundtruth para evaluación (si existe)
        if anns_by_image is not None and image_id in anns_by_image:
            gts = [a['bbox'] for a in anns_by_image[image_id]]
        else:
            gts = []
        gt_mapping[idx] = gts

        vis = frame.copy()
        for d in dets:
            x1,y1,x2,y2 = to_int_bbox(d['bbox'])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(vis, f"{d['score']:.2f}", (x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
        for t in tracks:
            x1,y1,x2,y2 = to_int_bbox(t['bbox'])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(vis, str(t['id']), (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
        vis_frames.append(vis)

    logger.info('Escribiendo CSVs de detecciones y tracks...')
    write_csv_detections(os.path.join(out_dir, 'detections.csv'), detections_by_frame)
    write_csv_tracks(os.path.join(out_dir, 'tracks.csv'), tracks_by_frame)

    if len(vis_frames) > 0:
        logger.info('Generando video de salida...')
        h,w = vis_frames[0].shape[:2]
        outv = os.path.join(out_dir, 'tracking.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outv, fourcc, fps, (w,h))
        for f in vis_frames:
            writer.write(f)
        writer.release()
        logger.info(f'Video guardado en {outv}')

    total_dets = sum(len(v) for v in detections_by_frame.values())
    logger.info(f'Total detecciones: {total_dets} en {len(detections_by_frame)} frames')

    metrics = {'total_images': len(detections_by_frame), 'total_detections': total_dets}
    import json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # si tenemos GT, calcular métricas simples
    if anns_by_image is not None:
        dets_for_eval = {i: [d['bbox'] for d in v] for i,v in detections_by_frame.items()}
        gt_for_eval = gt_mapping
        simple = detection_metrics(dets_for_eval, gt_for_eval, iou_thresh=0.5)
        with open(os.path.join(out_dir, 'detection_metrics.json'), 'w') as f:
            json.dump(simple, f, indent=2)

    logger.info('Pipeline completado.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=False, help='Directorio con imágenes.')
    parser.add_argument('--annotations', required=False, help='JSON COCO-like con images + annotations')
    parser.add_argument('--out', '-o', default='results/sample', help='Directorio de salida')
    parser.add_argument('--detector', default='yolov8n.pt', help='Pesos del detector YOLO')
    parser.add_argument('--device', default='cpu', help='cpu o cuda')
    parser.add_argument('--download', action='store_true', help='(FALLBACK) usa carpeta local de ejemplo si existe.')
    parser.add_argument('--n', type=int, default=200, help='Número máximo de imágenes a usar (solo si aplica)')
    args = parser.parse_args()

    images_dir = args.images

    if args.download and images_dir is None:
        logger.info('--download solicitado: buscando carpeta local de muestra (data/sample, sample_images, examples/images, etc.)...')
        sample_dir = find_local_sample_dir()
        if sample_dir is not None:
            logger.info(f'Carpeta de muestra encontrada: {sample_dir}')
            images_dir = sample_dir
        else:
            raise RuntimeError('No se encontró carpeta local de ejemplo. Coloca imágenes en data/sample/images o pase --images.')

    if images_dir is None:
        raise RuntimeError('Debe indicar --images o usar --download (si tienes carpeta local).')

    # si se pasa annotations JSON, obtenemos la lista ordenada desde el JSON
    if args.annotations:
        image_tuples = get_image_list_from_json(args.annotations, images_dir, n=args.n)
        _, anns_by_image, categories = load_annotations(args.annotations, images_dir)
        logger.info(f'Usando anotaciones {args.annotations} con {len(image_tuples)} imágenes encontradas en disco.')
    else:
        files = sorted([f for f in glob.glob(os.path.join(images_dir, '*')) if f.lower().endswith(('.jpg','.png','.jpeg'))])[:args.n]
        image_tuples = [(os.path.basename(f), f) for f in files]
        anns_by_image = None
        categories = {}

    run_image_sequence(image_tuples, args.out, detector_weights=args.detector, device=args.device, fps=2, anns_by_image=anns_by_image)
