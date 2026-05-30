"""
enfermedad/configurar/routes.py
================================
Blueprint que expone:
  GET    /enfermedad/configurar/                    → UI de configuraci\u00f3n
  GET    /enfermedad/configurar/sintomas            → lista completa
  POST   /enfermedad/configurar/sintomas            → agregar nuevo
  PUT    /enfermedad/configurar/sintomas/<id>       → editar
  DELETE /enfermedad/configurar/sintomas/<id>       → eliminar
  GET    /enfermedad/configurar/enfermedades        → lista completa
  POST   /enfermedad/configurar/enfermedades        → agregar nueva
  DELETE /enfermedad/configurar/enfermedades/<id>   → eliminar
  GET    /enfermedad/configurar/matriz              → relaciones actuales
  POST   /enfermedad/configurar/matriz              → guardar (reemplaza todo)
"""

import sqlite3
from flask import Blueprint, render_template, request, jsonify
from enfermedad.db import DB_PATH, reload_prolog

configurar_bp = Blueprint(
    'configurar',
    __name__,
    template_folder='templates',
)


# ── UI ────────────────────────────────────────────────────────────────────────

@configurar_bp.route('/')
def index():
    return render_template('configurar.html')


# ─────────────────────────────────────────────────────────────────────────────
# SÍNTOMAS
# ─────────────────────────────────────────────────────────────────────────────

@configurar_bp.route('/sintomas', methods=['GET'])
def get_sintomas():
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT idsintoma, sintoma, estado FROM sintoma ORDER BY idsintoma")
        sintomas = [{"idsintoma": r[0], "sintoma": r[1], "estado": r[2]} for r in cursor.fetchall()]
        conn.close()
        return jsonify({"status": "success", "sintomas": sintomas})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@configurar_bp.route('/sintomas', methods=['POST'])
def add_sintoma():
    try:
        nombre = (request.json or {}).get('sintoma', '').strip()
        if not nombre:
            return jsonify({"status": "error", "message": "Nombre requerido"}), 400
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sintoma (sintoma, estado) VALUES (?, 'a')", (nombre,))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        reload_prolog()
        return jsonify({"status": "success", "idsintoma": new_id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@configurar_bp.route('/sintomas/<int:sid>', methods=['PUT'])
def update_sintoma(sid):
    try:
        nombre = (request.json or {}).get('sintoma', '').strip()
        if not nombre:
            return jsonify({"status": "error", "message": "Nombre requerido"}), 400
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE sintoma SET sintoma=? WHERE idsintoma=?", (nombre, sid))
        conn.commit()
        conn.close()
        reload_prolog()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@configurar_bp.route('/sintomas/<int:sid>', methods=['DELETE'])
def delete_sintoma(sid):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("DELETE FROM sintoma WHERE idsintoma=?", (sid,))
        conn.commit()
        conn.close()
        reload_prolog()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
# ENFERMEDADES
# ─────────────────────────────────────────────────────────────────────────────

@configurar_bp.route('/enfermedades', methods=['GET'])
def get_enfermedades():
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT idenfermedad, nombre FROM enfermedad ORDER BY idenfermedad")
        enfermedades = [{"idenfermedad": r[0], "nombre": r[1]} for r in cursor.fetchall()]
        conn.close()
        return jsonify({"status": "success", "enfermedades": enfermedades})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@configurar_bp.route('/enfermedades', methods=['POST'])
def add_enfermedad():
    try:
        nombre = (request.json or {}).get('nombre', '').strip()
        if not nombre:
            return jsonify({"status": "error", "message": "Nombre requerido"}), 400
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO enfermedad (nombre) VALUES (?)", (nombre,))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        reload_prolog()
        return jsonify({"status": "success", "idenfermedad": new_id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@configurar_bp.route('/enfermedades/<int:eid>', methods=['DELETE'])
def delete_enfermedad(eid):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("DELETE FROM enfermedad WHERE idenfermedad=?", (eid,))
        conn.commit()
        conn.close()
        reload_prolog()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
# MATRIZ DE RELACIONES
# ─────────────────────────────────────────────────────────────────────────────

@configurar_bp.route('/matriz', methods=['GET'])
def get_matriz():
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT idsintoma, sintoma FROM sintoma ORDER BY idsintoma")
        sintomas = [{"idsintoma": r[0], "sintoma": r[1]} for r in cursor.fetchall()]

        cursor.execute("SELECT idenfermedad, nombre FROM enfermedad ORDER BY idenfermedad")
        enfermedades = [{"idenfermedad": r[0], "nombre": r[1]} for r in cursor.fetchall()]

        cursor.execute("SELECT idenfermedad, idsintoma FROM enfermedad_sintoma")
        mapeos = [{"idenfermedad": r[0], "idsintoma": r[1]} for r in cursor.fetchall()]

        conn.close()
        return jsonify({
            "status": "success",
            "sintomas": sintomas,
            "enfermedades": enfermedades,
            "mapeos": mapeos,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@configurar_bp.route('/matriz', methods=['POST'])
def save_matriz():
    try:
        mapeos = (request.json or {}).get('mapeos', [])
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM enfermedad_sintoma")
        if mapeos:
            cursor.executemany(
                "INSERT INTO enfermedad_sintoma (idenfermedad, idsintoma) VALUES (:idenfermedad, :idsintoma)",
                mapeos,
            )
        conn.commit()
        conn.close()
        reload_prolog()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
