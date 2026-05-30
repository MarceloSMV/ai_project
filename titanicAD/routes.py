"""
routes.py — Blueprint Flask para el módulo Titanic AD (Árbol de Decisión)
==========================================================================
Rutas:
  GET  /            → Página principal con formulario y visualización
  POST /predict     → Predicción individual con el modelo entrenado
  GET  /tree-image  → Imagen PNG del árbol de decisión
  GET  /metrics     → Métricas del modelo en JSON
"""

import os
import json
import pickle
from flask import Blueprint, render_template, request, jsonify, send_file

titanic_ad_bp = Blueprint('titanic_ad', __name__, template_folder='templates')

# ── Directorio base del módulo ────────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))

# ── Cargar modelo entrenado ───────────────────────────────────────────────────
_pkl_path = os.path.join(_base_dir, 'miarboltitanic.pkl')
_model = None

def _load_model():
    global _model
    if _model is None and os.path.exists(_pkl_path):
        with open(_pkl_path, 'rb') as f:
            _model = pickle.load(f)
    return _model

# ── Cargar métricas ───────────────────────────────────────────────────────────
_metrics_path = os.path.join(_base_dir, 'metrics.json')

def _load_metrics():
    if os.path.exists(_metrics_path):
        with open(_metrics_path, 'r') as f:
            return json.load(f)
    return None

# ── Función diagnóstico (del notebook) ────────────────────────────────────────
def diagnostico(valor):
    if(valor == 1):
        return "Si ha sobrevivido"
    else:
        return "No ha sobrevivido"


# ── Rutas ─────────────────────────────────────────────────────────────────────

@titanic_ad_bp.route('/')
def index():
    metrics = _load_metrics()
    return render_template('titanic_ad.html', metrics=metrics)


@titanic_ad_bp.route('/predict', methods=['POST'])
def predict():
    try:
        model = _load_model()
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Modelo no entrenado. Ejecuta train_titanic.py primero."
            }), 500

        data = request.json

        # columnas = ["Fare","Pclass","Gender","Age","SibSp"]
        fare   = float(data['fare'])
        pclass = float(data['pclass'])
        gender = float(data['gender'])   # 0=male, 1=female
        age    = float(data['age'])
        sibsp  = float(data['sibsp'])

        # Predicción con el árbol
        entrada = [[fare, pclass, gender, age, sibsp]]
        respuesta = model.predict(entrada)
        valor = int(respuesta[0])

        # Probabilidades (proporción de clases en la hoja)
        probas = model.predict_proba(entrada)[0].tolist()

        return jsonify({
            "status": "success",
            "prediccion": valor,
            "diagnostico": diagnostico(valor),
            "probabilidades": {
                "no_sobrevivio": round(probas[0] * 100, 2),
                "sobrevivio": round(probas[1] * 100, 2),
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


@titanic_ad_bp.route('/tree-image')
def tree_image():
    """Devuelve la imagen PNG del árbol de decisión."""
    img_path = os.path.join(_base_dir, 'titanic_tree.png')
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    return jsonify({"status": "error", "message": "Imagen no generada. Ejecuta train_titanic.py primero."}), 404


@titanic_ad_bp.route('/confusion-matrix')
def confusion_matrix_img():
    """Devuelve la imagen PNG de la matriz de confusión."""
    img_path = os.path.join(_base_dir, 'confusion_matrix.png')
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    return jsonify({"status": "error", "message": "Imagen no generada. Ejecuta train_titanic.py primero."}), 404


@titanic_ad_bp.route('/metrics')
def metrics():
    """Devuelve las métricas del modelo en JSON."""
    m = _load_metrics()
    if m:
        return jsonify(m)
    return jsonify({"status": "error", "message": "Métricas no disponibles."}), 404
