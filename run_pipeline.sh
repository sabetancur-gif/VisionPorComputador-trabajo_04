#!/usr/bin/env bash
# Script auxiliar para crear entorno e instalar dependencias (Linux / Mac)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Entorno creado e instalado. Usa: python src/run_pipeline.py --help"
