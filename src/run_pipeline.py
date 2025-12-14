"""
Pipeline principal de detección + tracking para MOT17 (COCO-like).

Uso (Windows):
python src\run_pipeline.py --images data\mot17\images --annotations data\mot17\annotations\train.json --out results\cis_val_run --detector yolov8n.pt --device cpu --n 300
"""

# Imports y setup de paths
import os
import sys
import argparse
import cv2
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Root del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from detector import YOLODetector
from tracker import OpticalFlowTracker
from eval import detection_metrics
from utils import to_int_bbox
from data_loader import get_image_list_from_json, load_annotations

logging.basicConfig(level=logging.INFO, format='[run_pipeline] %(message)s')
logger = logging.getLogger("run_pipeline")


# Helpers CSV

def write_csv_detections(out_csv, detections_by_frame):
    rows = []
    for frame_id, dets in detections_by_frame.items():
        for d in dets:
            x1, y1, x2, y2 = to_int_bbox(d["bbox"])
            rows.append({
                "frame": frame_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": float(d.get("score", 0.0)),
                "class": int(d.get("class", -1))
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def write_csv_tracks(out_csv, tracks_by_frame):
    rows = []
    for frame_id, tracks in tracks_by_frame.items():
        for t in tracks:
            x1, y1, x2, y2 = to_int_bbox(t["bbox"])
            rows.append({
                "frame": frame_id,
                "id": t["id"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)



# Pipeline principal

def run_image_sequence(image_tuples, out_dir, detector_weights, device,
                       fps=15, anns_by_image=None):

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Inicializando detector y tracker...")
    detector = YOLODetector(weights=detector_weights, device=device)
    tracker = OpticalFlowTracker()

    detections_by_frame = {}
    tracks_by_frame = {}
    gt_mapping = {}
    vis_frames = []

    for idx, (image_id, img_path) in enumerate(tqdm(image_tuples, desc="Frames")):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # 1. Detección
        dets = detector.detect(frame)

        # 2. Tracking
        tracks = tracker.step(frame, dets)

        detections_by_frame[idx] = dets
        tracks_by_frame[idx] = tracks

        # 3. Ground Truth
        if anns_by_image and image_id in anns_by_image:
            gt_mapping[idx] = [a["bbox"] for a in anns_by_image[image_id]]
        else:
            gt_mapping[idx] = []


        # Visualización

        vis = frame.copy()

        # Detecciones (VERDE)
        for d in dets:
            x1, y1, x2, y2 = to_int_bbox(d["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Tracks (AZUL + ID GRANDE)
        for t in tracks:
            x1, y1, x2, y2 = to_int_bbox(t["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = f"ID {t['id']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Fondo del texto
            cv2.rectangle(
                vis,
                (x1, max(y1 - th - 10, 0)),
                (x1 + tw + 6, y1),
                (255, 0, 0),
                -1
            )

            # Texto blanco
            cv2.putText(
                vis,
                label,
                (x1 + 3, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        # Ground Truth (ROJO)
        for gt in gt_mapping[idx]:
            x1, y1, x2, y2 = to_int_bbox(gt)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            vis,
            f"Frame {idx} | Dets {len(dets)} | Tracks {len(tracks)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        vis_frames.append(vis)

        if idx < 5:
            cv2.imwrite(str(fig_dir / f"sample_frame_{idx}.png"), vis)


    # Guardar resultados

    write_csv_detections(out_dir / "detections.csv", detections_by_frame)
    write_csv_tracks(out_dir / "tracks.csv", tracks_by_frame)

    # Video
    if vis_frames:
        h, w = vis_frames[0].shape[:2]
        out_video = out_dir / "tracking_result.mp4"
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )
        for f in vis_frames:
            writer.write(f)
        writer.release()


    # Métricas detección

    if anns_by_image:
        dets_eval = {i: [d["bbox"] for d in v] for i, v in detections_by_frame.items()}
        metrics = detection_metrics(dets_eval, gt_mapping)

        with open(out_dir / "detection_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        plt.figure(figsize=(6, 4))
        plt.bar(["Precision", "Recall", "F1"],
                [metrics["precision"], metrics["recall"], metrics["f1"]])
        plt.ylim(0, 1)
        plt.title("Detection metrics (IoU = 0.5)")
        plt.tight_layout()
        plt.savefig(fig_dir / "detection_metrics.png")
        plt.close()
        
        
    # Figura: detecciones por frame

    frames = list(detections_by_frame.keys())
    det_counts = [len(detections_by_frame[f]) for f in frames]

    plt.figure(figsize=(8, 4))
    plt.plot(frames, det_counts)
    plt.xlabel("Frame")
    plt.ylabel("Detections")
    plt.title("Detections per frame")
    plt.tight_layout()
    plt.savefig(fig_dir / "detections_per_frame.png")
    plt.close()

    logger.info("Pipeline completado correctamente.")


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", default="results/cis_val_run")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    image_tuples = get_image_list_from_json(
        args.annotations, args.images, n=args.n
    )
    _, anns_by_image, _ = load_annotations(args.annotations, args.images)

    run_image_sequence(
        image_tuples,
        args.out,
        detector_weights=args.detector,
        device=args.device,
        fps=15,
        anns_by_image=anns_by_image
    )
