"""
app.py — Dispatcher principal
==============================
Este archivo SOLO registra blueprints y redirige la raíz.
Toda la lógica de negocio vive dentro de cada módulo:
  · home/routes.py
  · aereo/routes.py
  · enfermedad/diagnosticar/routes.py
  · enfermedad/configurar/routes.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, redirect, url_for

from home.routes                     import home_bp
from aereo.routes                    import aereo_bp
from enfermedad.db                   import init_all
from enfermedad.diagnosticar.routes  import diagnosticar_bp
from enfermedad.configurar.routes    import configurar_bp
from tb_detector.routes              import tb_detector_bp
from titanicAD.routes                import titanic_ad_bp

# ── Crear la app ──────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Inicializar BD SQLite + generar seic.pl + arrancar PySwip ─────────────────
init_all()

# ── Registrar blueprints ──────────────────────────────────────────────────────
app.register_blueprint(home_bp,         url_prefix='/home')
app.register_blueprint(aereo_bp,        url_prefix='/aereo')
app.register_blueprint(diagnosticar_bp, url_prefix='/enfermedad/diagnosticar')
app.register_blueprint(configurar_bp,   url_prefix='/enfermedad/configurar')
app.register_blueprint(tb_detector_bp,   url_prefix='/tb_detector')
app.register_blueprint(titanic_ad_bp,    url_prefix='/titanic_ad')


# ── Ruta raíz: muestra la pantalla de inicio con sidebar ─────────────────────
@app.route('/')
def index():
    return redirect(url_for('home.index'))


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    # use_reloader=False evita doble carga del modelo TF y de PySwip
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)