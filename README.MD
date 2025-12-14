# Visión por Computador: QuantumViz

## 🧠 Trabajo 04 — Sistema de Detección y Seguimiento de Peatones (MOT17)

---

### 📌 Resumen

Este repositorio presenta el desarrollo, implementación y evaluación de un pipeline completo de visión por computador para detección y seguimiento de objetos en secuencias de video, utilizando:

* YOLOv8 para detección de objetos (personas)

* Seguimiento basado en Flujo Óptico + IoU para asignación de identidades (tracking)

El sistema se aplica sobre el dataset MOT17, ampliamente utilizado en investigación para Multiple Object Tracking (MOT), y permite generar resultados visuales, métricas cuantitativas y salidas reproducibles para análisis académico.

---

### 📁 Estructura del repositorio

```
vision-por-computador-trabajo_04/
├── data/
│   └── mot17/
│       ├── images/
│       │   ├── MOT17-02-FRCNN/
│       │   ├── MOT17-04-FRCNN/
│       │   └── ...
│       └── annotations/
│           ├── train.json
│           ├── test.json
│           ├── train_half.json
│           └── val_half.json
│
├── notebooks/
│   └── exploratorio.ipynb
│
├── results/
│   ├── cis_val_run/
│   │   ├── figures/
│   │   │   ├── detection_metrics.png
│   │   │   ├── detections_per_frame.png
│   │   │   ├── sample_frame_0.png
│   │   │   └── ...
│   │   ├── detections.csv
│   │   ├── tracks.csv
│   │   ├── metrics.json
│   │   ├── detection_metrics.json
│   │   └── tracking_result.mp4
│   └── exploratorio/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── detector.py
│   ├── tracker.py
│   ├── eval.py
│   ├── utils.py
│   └── run_pipeline.py
│
├── requirements.txt
├── yolo8n.pt
└── README.md
```

---

### 🎯 Objetivos específicos

* Implementar un detector de peatones usando YOLOv8.
* Aplicar seguimiento multi-objeto mediante flujo óptico (Lucas–Kanade) y matching por IoU.
* Integrar detección y tracking en un pipeline reproducible.
* Evaluar el desempeño mediante métricas de detección (Precision, Recall, F1).
* Generar visualizaciones y resultados claros para análisis técnico y académico.

---

### 📦 Datos

El proyecto utiliza el dataset MOT17 (Multiple Object Tracking) en formato COCO-like, disponible públicamente.

* Clases: pedestrian
* Resolución típica: 1920×1080
* Anotaciones por frame con bounding boxes
* Secuencias temporales reales (tracking)

El dataset base: `https://www.kaggle.com/datasets/nhnhunhch/mot17-coco`

---

### 🛠 Instalación y entorno

Se recomienda usar un entorno virtual (conda o venv). A continuación instrucciones con `venv` (Windows / Linux / macOS).

```bash
python -m venv venv
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` incluye (ejemplos): `numpy, pandas, scikit-learn, matplotlib, opencv-python, scikit-image, torch, torchvision, tqdm`.

---

### 🚀 Ejecución — Notebooks (recomendado)

Abrir `run_pipeline.py`o desde la raíz del proyecto y ejecutar desde la terminar el siguiente comando:
``` 
 python src\run_pipeline.py --images data\mot17\images --annotations data\mot17\annotations\train.json --out results\cis_val_run --detector yolov8n.pt --device cpu --n 300

```

**Nota:** El archivo `notebooks/exploratorio.ipynb` es el único .ipynb ejecutable que nos permite realizar un primer acercamiento realizando resultados vistos a lo largo del curso sobre la data.

---

### 🧭 Uso desde Python — API mínima

Ejemplos de import desde `src`:

```python
from src.detector import YOLODetector
from src.tracker import OpticalFlowTracker
from src.eval import detection_metrics
from src.utils import to_int_bbox
from src.data_loader import get_image_list_from_json, load_annotations
```
---

