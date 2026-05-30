import os
import numpy as np
import tensorflow as tf
from flask import Blueprint, render_template, request, jsonify

aereo_bp = Blueprint('aereo', __name__, template_folder='templates')

# ── Carga del modelo al importar el módulo ──────────────────────────────────
_ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo_entrenado.h5')
_modelo = tf.keras.models.load_model(_ruta_modelo)

# ── Rutas ────────────────────────────────────────────────────────────────────

@aereo_bp.route('/')
def index():
    return render_template('aereo.html')


@aereo_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = {
            "sex":   np.array([data['sex']],          dtype=object),
            "age":   np.array([float(data['age'])],   dtype=np.float32),
            "fare":  np.array([float(data['fare'])],  dtype=np.float32),
            "class": np.array([data['class']],        dtype=object),
            "deck":  np.array([data['deck']],         dtype=object),
        }
        prediccion   = _modelo.predict(input_data, verbose=0)
        probabilidad = float(tf.nn.sigmoid(prediccion).numpy()[0][0])
        resultado    = "SOBREVIVE" if probabilidad > 0.5 else "NO SOBREVIVE"
        return jsonify({
            "status":      "success",
            "probabilidad": probabilidad,
            "estado":       resultado,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
