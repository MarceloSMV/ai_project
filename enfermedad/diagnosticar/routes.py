"""
enfermedad/diagnosticar/routes.py
==================================
Blueprint que expone:
  GET  /enfermedad/diagnosticar/          → UI de diagn\u00f3stico
  GET  /enfermedad/diagnosticar/sintomas  → lista de s\u00edntomas (JSON)
  POST /enfermedad/diagnosticar/predict   → inferencia Prolog (JSON)
"""

import sqlite3
from flask import Blueprint, render_template, request, jsonify
from enfermedad.db import get_prolog, DB_PATH

diagnosticar_bp = Blueprint(
    'diagnosticar',
    __name__,
    template_folder='templates',
)


# ── UI ────────────────────────────────────────────────────────────────────────

@diagnosticar_bp.route('/')
def index():
    return render_template('diagnosticar.html')


# ── API: Cat\u00e1logo de s\u00edntomas ──────────────────────────────────────────────────

@diagnosticar_bp.route('/sintomas', methods=['GET'])
def get_sintomas():
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT idsintoma, sintoma FROM sintoma ORDER BY idsintoma"
        )
        sintomas = [{"idsintoma": r[0], "sintoma": r[1]} for r in cursor.fetchall()]
        conn.close()
        return jsonify({"status": "success", "sintomas": sintomas})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ── API: Inferencia Prolog ────────────────────────────────────────────────────

@diagnosticar_bp.route('/predict', methods=['POST'])
def predict():
    try:
        prolog   = get_prolog()
        data     = request.json
        sintomas = data.get('sintomas', [])   # lista de ids enteros

        # Limpiar hechos previos
        list(prolog.query("limpiar"))

        # Insertar s\u00edntomas marcados como hechos s\u00edmbolos (s1, s2, ...)
        for sid in sintomas:
            token = f"s{sid}" if not str(sid).startswith('s') else str(sid)
            list(prolog.query(f"assertz(tiene({token}))"))

        # Consultar enfermedades inferidas
        resultados  = list(prolog.query("enfermedad(E)"))
        enfermedades = list(set(r['E'] for r in resultados))

        # Eliminar fallback si hay diagn\u00f3sticos espec\u00edficos
        fallback = 'No Determinado (Sin patron claro)'
        if len(enfermedades) > 1 and fallback in enfermedades:
            enfermedades.remove(fallback)

        if not enfermedades:
            enfermedades = ["Ninguna enfermedad detectada con esos s\u00edntomas."]

        return jsonify({"status": "success", "enfermedades": enfermedades})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
