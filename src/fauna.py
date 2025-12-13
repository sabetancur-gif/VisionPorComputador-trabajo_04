#!/usr/bin/env python3
"""
fauna_recognizer.py

Lee un video (por defecto data/fauna.*) y usa los CSVs generados en results/
para anotar el video con bounding boxes, IDs y etiquetas (si existen).
Salida:
 - results/fauna_recognized.mp4  (video anotado)
 - results/fauna_frame_summary.csv (resumen por frame: detecciones, ids, labels)
"""

import os
import cv2
import argparse
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def deterministic_color_for_id(id_val: str) -> Tuple[int,int,int]:
    """Genera un color BGR determinista a partir del id (para boxes)."""
    h = abs(hash(str(id_val))) % (256**3)
    b = (h & 0xFF)
    g = (h >> 8) & 0xFF
    r = (h >> 16) & 0xFF
    return (int(b), int(g), int(r))

def try_load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        logging.warning(f"No existe el archivo CSV: {path}")
        return None
    try:
        df = pd.read_csv(path)
        logging.info(f"Cargado CSV {path} con {len(df)} filas.")
        return df
    except Exception as e:
        logging.error(f"Error leyendo {path}: {e}")
        return None

def normalize_detection_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta normalizar/renombrar columnas comunes en detections/tracks CSVs.
    Buscamos: frame, id, x1,x2,y1,y2 (o x,y,w,h), score, label
    """
    df = df.copy()
    cols = [c.lower().strip() for c in df.columns]
    mapping = {}
    # frame
    for name in ('frame','frame_id','frame_nr','frame_no','frame_number'):
        if name in cols:
            mapping[df.columns[cols.index(name)]] = 'frame'
            break
    # id
    for name in ('id','track_id','track','object_id'):
        if name in cols:
            mapping[df.columns[cols.index(name)]] = 'id'
            break
    # bbox: x1,y1,x2,y2
    if any(x in cols for x in ('x1','x2','y1','y2')):
        for name in ('x1','y1','x2','y2'):
            if name in cols:
                mapping[df.columns[cols.index(name)]] = name
    # bbox: x,y,w,h
    elif all(x in cols for x in ('x','y','w','h')):
        mapping[df.columns[cols.index('x')]] = 'x'
        mapping[df.columns[cols.index('y')]] = 'y'
        mapping[df.columns[cols.index('w')]] = 'w'
        mapping[df.columns[cols.index('h')]] = 'h'
    # score
    for name in ('score','conf','confidence'):
        if name in cols:
            mapping[df.columns[cols.index(name)]] = 'score'
            break
    # label
    for name in ('label','class','species','cat'):
        if name in cols:
            mapping[df.columns[cols.index(name)]] = 'label'
            break

    if mapping:
        df = df.rename(columns=mapping)

    # if we have x,y,w,h convert to x1,y1,x2,y2
    if set(['x','y','w','h']).issubset(df.columns):
        df['x1'] = df['x'].astype(int)
        df['y1'] = df['y'].astype(int)
        df['x2'] = (df['x'] + df['w']).astype(int)
        df['y2'] = (df['y'] + df['h']).astype(int)
    return df

def build_frame_index(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Devuelve un dict frame -> sub-DataFrame (detecciones en ese frame)."""
    index = {}
    if df is None or df.empty:
        return index
    if 'frame' not in df.columns:
        logging.warning("El CSV no contiene columna 'frame'. Trataré todo como frame=0.")
        df['frame'] = 0
    for frame, g in df.groupby('frame'):
        index[int(frame)] = g
    return index

def annotate_frame(frame_img, detections_df: pd.DataFrame):
    """Dibuja boxes, label y id sobre la imagen (OpenCV BGR)."""
    if detections_df is None or detections_df.empty:
        return frame_img
    img = frame_img
    for _, row in detections_df.iterrows():
        # obtener bbox
        if all(k in row for k in ('x1','y1','x2','y2')):
            x1,y1,x2,y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        else:
            continue
        idv = row['id'] if 'id' in row and not pd.isna(row['id']) else None
        label = str(row['label']) if 'label' in row and not pd.isna(row['label']) else None
        score = row['score'] if 'score' in row and not pd.isna(row['score']) else None

        color = deterministic_color_for_id(idv if idv is not None else label or x1)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

        # texto: "label id (score)" compact
        pieces = []
        if label:
            pieces.append(str(label))
        if idv:
            pieces.append(f"ID:{idv}")
        if score is not None:
            try:
                pieces.append(f"{float(score):.2f}")
            except:
                pieces.append(str(score))
        text = " ".join(pieces) if pieces else ""
        # fondo de texto
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(y1-18,0)), (x1+tw+6, y1), color, -1)
            cv2.putText(img, text, (x1+3, max(y1-5,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img

def process_video(video_path: str,
                  detections_csv: Optional[str],
                  tracks_csv: Optional[str],
                  out_video: str,
                  out_summary_csv: str,
                  show_preview: bool = False):
    # Carga CSVs
    det_df = try_load_csv(detections_csv) if detections_csv else None
    trk_df = try_load_csv(tracks_csv) if tracks_csv else None
    if det_df is not None:
        det_df = normalize_detection_df(det_df)
    if trk_df is not None:
        trk_df = normalize_detection_df(trk_df)

    det_index = build_frame_index(det_df)
    trk_index = build_frame_index(trk_df)

    # abrir video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logging.info(f"Video abierto {video_path}  —  {w}x{h} @ {fps}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w,h))

    # resumen rows
    summary_rows = []
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # usando frame index (si el CSV usa 0-based o 1-based — manejamos ambos)
        # primero intentar frame_no (0-based), luego frame_no+1
        dets = det_index.get(frame_no, pd.DataFrame())
        if dets.empty:
            dets = det_index.get(frame_no+1, pd.DataFrame())
        # tracks preferentemente
        trks = trk_index.get(frame_no, pd.DataFrame())
        if trks.empty:
            trks = trk_index.get(frame_no+1, pd.DataFrame())

        # combinar detecciones y tracks (tracks puede complementar IDs)
        # si dets tiene bbox sin id pero trks tiene el mismo bbox -> intentar asignar id
        # simplificación: priorizar dets; añadir columnas id/label si existen en trks
        annotated_df = dets.copy() if not dets.empty else trks.copy()
        # si dets no tiene id y trks tiene -> left-join by bbox (approx)
        if (not dets.empty) and (not trks.empty) and ('id' not in dets.columns or dets['id'].isnull().all()):
            # hacer emparejamiento simple por IoU máximo
            def iou(a,b):
                xA = max(a[0], b[0]); yA = max(a[1], b[1])
                xB = min(a[2], b[2]); yB = min(a[3], b[3])
                interW = max(0, xB-xA); interH = max(0, yB-yA)
                inter = interW*interH
                aarea = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
                barea = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
                denom = aarea + barea - inter
                return inter/denom if denom>0 else 0
            dets2 = dets.copy()
            trk_rows = trks.to_dict('records')
            assigned_ids = []
            for i, r in dets2.iterrows():
                if all(c in r for c in ('x1','y1','x2','y2')):
                    a = (int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2']))
                    best_iou = 0; best_id = None
                    for tr in trk_rows:
                        if all(k in tr for k in ('x1','y1','x2','y2','id')):
                            b = (int(tr['x1']), int(tr['y1']), int(tr['x2']), int(tr['y2']))
                            val = iou(a,b)
                            if val > best_iou:
                                best_iou = val
                                best_id = tr.get('id')
                    if best_iou > 0.3 and best_id is not None:
                        dets2.at[i,'id'] = best_id
            annotated_df = dets2

        # dibujar
        out_frame = annotate_frame(frame.copy(), annotated_df)
        # pie de frame con número y counts
        text = f"Frame: {frame_no}  |  dets: {len(annotated_df)}"
        cv2.putText(out_frame, text, (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        writer.write(out_frame)
        if show_preview:
            cv2.imshow("fauna_recognizer", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # resumen fila
        ids = list(annotated_df['id'].dropna().unique()) if 'id' in annotated_df.columns else []
        labels = list(annotated_df['label'].dropna().unique()) if 'label' in annotated_df.columns else []
        summary_rows.append({
            'frame': frame_no,
            'detections': len(annotated_df),
            'ids': ";".join(map(str, ids)),
            'labels': ";".join(map(str, labels))
        })

        frame_no += 1

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    # guardar resumen CSV
    os.makedirs(os.path.dirname(out_summary_csv) or ".", exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index=False)
    logging.info(f"Procesado completo. Video anotado guardado en: {out_video}")
    logging.info(f"Resumen guardado en: {out_summary_csv}")

def main():
    parser = argparse.ArgumentParser(description="Anotar video con detecciones y tracks existentes")
    parser.add_argument("--video", type=str, default="data/fauna", help="Ruta al video fuente (ej: data/fauna.mp4)")
    parser.add_argument("--detections", type=str, default="results/cis_val_run/detections.csv", help="CSV de detecciones")
    parser.add_argument("--tracks", type=str, default="results/cis_val_run/tracks.csv", help="CSV de tracks (opcional)")
    parser.add_argument("--out", type=str, default="results/fauna_recognized.mp4", help="Ruta de salida video anotado")
    parser.add_argument("--summary", type=str, default="results/fauna_frame_summary.csv", help="CSV resumen por frame")
    parser.add_argument("--preview", action="store_true", help="Mostrar previsualización en pantalla mientras procesa")
    args = parser.parse_args()

    # si video no tiene extensión, intentar añadir .mp4
    video_path = args.video
    if not os.path.splitext(video_path)[1]:
        if os.path.exists(video_path + ".mp4"):
            video_path = video_path + ".mp4"
        elif os.path.exists(video_path + ".avi"):
            video_path = video_path + ".avi"

    process_video(video_path, args.detections, args.tracks, args.out, args.summary, show_preview=args.preview)

if __name__ == "__main__":
    main()
