import os
import sys
import cv2
import json
import pickle
import base64
import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_file

# ── Configurar PYTHONPATH para importar de vision_computacional ──────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_vc_dir = os.path.join(_current_dir, 'vision_computacional')
if _vc_dir not in sys.path:
    sys.path.insert(0, _vc_dir)

# Importación diferida para que sys.path ya esté configurado
from utils.inference import TBInference

tb_detector_bp = Blueprint('tb_detector', __name__, template_folder='templates')

# ── Cargar modelo de inferencia de Tuberculosis (clasificador EfficientNet) ───
_checkpoint_path = os.path.join(_vc_dir, 'runs', 'checkpoints', 'best.pt')
_infer = TBInference(checkpoint_path=_checkpoint_path)

# ── Cargar modelo del Árbol de Decisión ───────────────────────────────────────
_ad_dir       = os.path.join(_current_dir, 'arbol_decision')
_pkl_path     = os.path.join(_ad_dir, 'miarbol_tb.pkl')
_metrics_path = os.path.join(_ad_dir, 'metrics_tb.json')
_tree_png     = os.path.join(_ad_dir, 'tb_tree.png')
_cm_png       = os.path.join(_ad_dir, 'confusion_matrix_tb.png')
_depth_json   = os.path.join(_ad_dir, 'best_depth.json')

_arbol_model     = None
_arbol_metrics   = None
_arbol_depth_cfg = None


def _load_arbol():
    global _arbol_model, _arbol_metrics, _arbol_depth_cfg
    if os.path.exists(_pkl_path):
        with open(_pkl_path, 'rb') as f:
            _arbol_model = pickle.load(f)
    if os.path.exists(_metrics_path):
        with open(_metrics_path, 'r', encoding='utf-8') as f:
            _arbol_metrics = json.load(f)
    if os.path.exists(_depth_json):
        with open(_depth_json, 'r', encoding='utf-8') as f:
            _arbol_depth_cfg = json.load(f)


_load_arbol()


def encode_img_base64(img_np):
    """Convierte imagen BGR (numpy) a data-URI base64 para el frontend."""
    if img_np is None:
        return None
    _, buffer = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')


# ── Preprocesamiento para el árbol (mismo que en train_arbol_tb.py) ──────────
IMG_SIZE_ARBOL = 64


def _discretizar_pixel(valor):
    if valor <= 80:
        return 0   # Oscuro
    elif valor <= 170:
        return 1   # Medio
    else:
        return 2   # Claro


def _preprocesar_imagen(img_bytes):
    """Convierte bytes de imagen a vector discretizado 4096-D."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")
    img_resized = cv2.resize(img, (IMG_SIZE_ARBOL, IMG_SIZE_ARBOL))
    vector = img_resized.flatten()
    vector_discreto = np.array([_discretizar_pixel(int(p)) for p in vector])
    return vector_discreto.reshape(1, -1)


# ════════════════════════════════════════════════════════════════════════════
# RUTAS — VISIÓN COMPUTACIONAL (Clasificador EfficientNet-B3 + Grad-CAM)
# ════════════════════════════════════════════════════════════════════════════

@tb_detector_bp.route('/')
def index():
    return render_template('tb_detector.html')


@tb_detector_bp.route('/predict', methods=['POST'])
def predict():
    """
    Recibe una radiografía y retorna:
      - verdict:      'health' | 'sick' | 'tb'
      - class_report: probabilidades por clase (%)
      - image_boxes:  imagen con bounding boxes generados por Grad-CAM
      - image_gradcam: imagen con mapa de calor Grad-CAM superpuesto
      - inference_ms: tiempo de inferencia
      - boxes:        lista de cajas [{x1,y1,x2,y2,confidence}, ...]
    """
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No se subió ninguna imagen"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Archivo no seleccionado"}), 400

        img_bytes    = file.read()
        score_thresh = float(request.form.get('threshold', 0.55))

        result = _infer.predict_bytes(
            img_bytes,
            score_thresh=score_thresh,
            include_gradcam=True,
        )

        return jsonify({
            "status":        "success",
            "verdict":       result["verdict"],
            "class_report":  result["class_report"],
            "inference_ms":  result["inference_ms"],
            "image_boxes":   encode_img_base64(result["vis_boxes"]),
            "image_gradcam": encode_img_base64(result["vis_gradcam"]),
            "boxes":         result["boxes"],
            "num_boxes":     len(result["boxes"]),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════
# RUTAS — ÁRBOL DE DECISIÓN
# ════════════════════════════════════════════════════════════════════════════

@tb_detector_bp.route('/arbol')
def arbol_index():
    """Página principal del módulo Árbol de Decisión TB."""
    return render_template('arbol_tb.html', metrics=_arbol_metrics, depth_cfg=_arbol_depth_cfg)


@tb_detector_bp.route('/arbol/predict', methods=['POST'])
def arbol_predict():
    """Clasificación de radiografía usando el árbol de decisión."""
    try:
        if _arbol_model is None:
            return jsonify({"status": "error",
                            "message": "Modelo no entrenado. Ejecuta train_arbol_tb.py"}), 503

        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No se subió ninguna imagen"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Archivo no seleccionado"}), 400

        img_bytes = file.read()
        X         = _preprocesar_imagen(img_bytes)

        pred_class = int(_arbol_model.predict(X)[0])
        proba      = _arbol_model.predict_proba(X)[0]

        clases  = ['health', 'sick', 'tb']
        etiqueta = clases[pred_class]
        iconos   = {'health': '✅', 'sick': '⚠️', 'tb': '🦠'}
        colores  = {'health': 'success', 'sick': 'warning', 'tb': 'danger'}

        return jsonify({
            "status":     "success",
            "prediccion": pred_class,
            "etiqueta":   etiqueta,
            "icono":      iconos[etiqueta],
            "color":      colores[etiqueta],
            "probabilidades": {
                "health": round(float(proba[0]) * 100, 1),
                "sick":   round(float(proba[1]) * 100, 1),
                "tb":     round(float(proba[2]) * 100, 1),
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@tb_detector_bp.route('/arbol/tree-image')
def arbol_tree_image():
    """Sirve el PNG del árbol de decisión entrenado."""
    if os.path.exists(_tree_png):
        return send_file(_tree_png, mimetype='image/png')
    return jsonify({"error": "Imagen no encontrada"}), 404


@tb_detector_bp.route('/arbol/confusion-matrix')
def arbol_confusion_matrix():
    """Sirve el PNG de la matriz de confusión."""
    if os.path.exists(_cm_png):
        return send_file(_cm_png, mimetype='image/png')
    return jsonify({"error": "Imagen no encontrada"}), 404
