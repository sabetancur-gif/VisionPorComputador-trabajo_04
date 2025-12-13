"""Helpers para leer los JSON COCO-like (las anotaciones que subiste)
y mapear image_id/file_name -> bboxes.

Funcionalidad principal:
- load_annotations(json_path, images_dir) -> (imageid_to_filepath, anns_by_image, categories)
- get_image_list_from_json(json_path, images_dir, n=None)
"""

import json
import os

def coco_bbox_to_xyxy(bbox):
    # bbox: [x,y,w,h] -> [x1,y1,x2,y2]
    x,y,w,h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]

def load_annotations(json_path, images_dir=None):
    """
    Carga el JSON COCO-like y devuelve:
      - imageid_to_filepath: dict image_id -> full path (if images_dir provided and file exists) or file_name
      - anns_by_image: dict image_id -> list of {'bbox':[x1,y1,x2,y2], 'category_id':int, 'id':anno_id}
      - categories: dict category_id -> name (if present)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories_list = data.get('categories', [])

    categories = {c['id']: c.get('name', str(c['id'])) for c in categories_list}

    imageid_to_name = {}
    imageid_to_filepath = {}
    for im in images:
        iid = im.get('id') or im.get('file_name')
        fname = im.get('file_name') if 'file_name' in im else im.get('id')
        imageid_to_name[iid] = fname
        if images_dir:
            candidate = os.path.join(images_dir, fname)
            if os.path.exists(candidate):
                imageid_to_filepath[iid] = candidate
            else:
                imageid_to_filepath[iid] = fname  # fallback: relative name
        else:
            imageid_to_filepath[iid] = fname

    anns_by_image = {}
    for a in annotations:
        iid = a.get('image_id')
        bbox = a.get('bbox')
        if bbox is None:
            continue
        xyxy = coco_bbox_to_xyxy(bbox)
        rec = {'bbox': xyxy, 'category_id': a.get('category_id'), 'id': a.get('id')}
        anns_by_image.setdefault(iid, []).append(rec)

    return imageid_to_filepath, anns_by_image, categories

def get_image_list_from_json(json_path, images_dir, n=None, require_file_exists=True):
    """Devuelve lista ordenada de rutas de imagen (solo las que existan si require_file_exists)."""
    imageid_to_filepath, anns, cats = load_annotations(json_path, images_dir=images_dir)
    ordered = []
    # preserve order in JSON images array
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for im in data.get('images', []):
        iid = im.get('id') or im.get('file_name')
        path = imageid_to_filepath.get(iid)
        if require_file_exists:
            if path and os.path.exists(path):
                ordered.append((iid, path))
        else:
            ordered.append((iid, path))
        if n and len(ordered) >= n:
            break
    return ordered  # list of tuples (image_id, path)
