# Visión por Computador: QuantumViz

## 🧠 Trabajo 04 — Sistema de Detección y Seguimiento de Objetos en Video

---

### 📌 Resumen

Este repositorio documenta, implementa y evalúa un pipeline completo para la detección de objetos mediante YOLO y seguimiento de objetos mediante técnicas de flujo óptico, aplicada a un problema práctico.

El objetivo es desarrollar una aplicación de visión por computador que integre detección de objetos mediante YOLO y seguimiento de objetos mediante técnicas de flujo óptico, aplicada a un problema práctico utilizando datos de video disponibles públicamente.

---

### 📁 Estructura del repositorio

```
proyecto-clasificacion-imagenes-medicas/
├── data/
│   ├── sample/
│   │   ├── images/
│   │   │   ├── 5a0affad-23d2-11e8-a6a3-ec086b02610b
│   │   │   └── 5a0affad-23d2-11e8-a6a3-ec086b02610b
│   │   ├── cis_test_annotations.json
│   │   ├── cis_val_annotations.json
│   │   ├── train_annotations.json
│   │   ├── trans_val_annotations.json
│   │   └── fauna.mp4
├── notebooks/
│   └── exploratorio.ipynb
├── resultados/
│   └── cis_val_run/
│   │   ├── detection_metrics.json
│   │   ├── detection.csv
│   │   ├── metrics.json
│   │   ├── tracks.csv
│   ├── fauna_frame_summary.csv
│   └── fauna_recognized.mp4
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── detector.py
│   ├── eval.py
│   ├── fauna.py
│   ├── run_pipeline.py
│   ├── tracker.py
│   └── utils.py
├── .gitignore
├── README.md   # <- este archivo
├── requirements.txt
├── run_pipeline.sh
└── yolo8n.pt
```

---

### 🎯 Objetivos específicos

* Implementar y configurar un modelo de detección de objetos YOLO (v5, v8 u otra versión).
* Aplicar técnicas de flujo óptico (Lucas-Kanade o flujo denso) para el seguimiento de objetos entre fotogramas.
* Integrar ambas técnicas en un pipeline coherente de análisis de video.
* Evaluar cuantitativamente el desempeño del sistema.
* Comunicar resultados técnicos de forma clara y profesional.

---

### 📦 Datos

* El dataset base tanto las imagenes como los metadatos proviene de un conjunto público de `https://lila.science/datasets/caltech-camera-traps`, más precisamente de `Benchmark images(6GB)` y `Metadata files for train/val/cis/trans splits (3MB)`


**Nota:** mantener `5a0affad-23d2-11e8-a6a3-ec086b02610b` (u otros ejemplos) en `data/` para pruebas rápidas.

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
 python -m src.run_pipeline --images data/sample/images --annotations data/sample/cis_val_annotations.json --out results/cis_val_run --detector yolov8n.pt --device cpu --n 200
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
