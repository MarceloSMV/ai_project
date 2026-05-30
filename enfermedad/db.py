"""
enfermedad/db.py
================
Capa de datos compartida entre los subm\u00f3dulos diagnosticar/ y configurar/.
Gestiona:
  - Inicializaci\u00f3n de la base de datos SQLite (enfermedades.db)
  - Generaci\u00f3n din\u00e1mica del archivo de reglas Prolog (seic.pl)
  - Singleton de la instancia PySwip Prolog
"""

import os
import sqlite3
from pyswip import Prolog

# ── Rutas absolutas ───────────────────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(_DIR, 'enfermedades.db')
PROLOG_PATH  = os.path.join(_DIR, 'seic.pl')

# Singleton Prolog (se inicializa en init_all(), llamado desde app.py)
_prolog: Prolog | None = None


# ── Acceso al singleton ───────────────────────────────────────────────────────

def get_prolog() -> Prolog:
    """Devuelve la instancia global de Prolog."""
    return _prolog


# ── Inicializaci\u00f3n de la base de datos ─────────────────────────────────────────

def init_db() -> None:
    """Crea las tablas y siembra los datos iniciales si la BD est\u00e1 vac\u00eda."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sintoma (
            idsintoma INTEGER PRIMARY KEY AUTOINCREMENT,
            sintoma   TEXT NOT NULL,
            estado    TEXT DEFAULT 'a'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enfermedad (
            idenfermedad INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre       TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enfermedad_sintoma (
            idenfermedad INTEGER,
            idsintoma    INTEGER,
            PRIMARY KEY (idenfermedad, idsintoma),
            FOREIGN KEY (idenfermedad) REFERENCES enfermedad(idenfermedad) ON DELETE CASCADE,
            FOREIGN KEY (idsintoma)    REFERENCES sintoma(idsintoma)       ON DELETE CASCADE
        )
    """)

    # ── Siembra de s\u00edntomas ────────────────────────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM sintoma")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO sintoma (idsintoma, sintoma, estado) VALUES (?, ?, ?)",
            [
                (1,  'Fiebre alta de inicio s\u00fabito', 'a'),
                (2,  'Dolor retroocular (dolor intenso detr\u00e1s de los globos oculares)', 'a'),
                (3,  'Dolor articular severo e incapacitante (dificultad para mover las articulaciones)', 'a'),
                (4,  'Conjuntivitis no purulenta (ojos enrojecidos sin secreci\u00f3n ni laga\u00f1as)', 'a'),
                (5,  'Erupci\u00f3n cut\u00e1nea o sarpullido (rash maculopapular)', 'a'),
                (6,  'Dificultad respiratoria severa y repentina (sensaci\u00f3n de ahogo)', 'a'),
                (7,  'Dolor muscular agudo, focalizado principalmente en las pantorrillas', 'a'),
                (8,  'Ictericia (coloraci\u00f3n amarillenta en la piel y la parte blanca de los ojos)', 'a'),
                (9,  'Sangrado espont\u00e1neo (en mucosas, enc\u00edas, nariz o bajo la piel)', 'a'),
                (10, 'V\u00f3mitos persistentes y dolor abdominal intenso', 'a'),
                (11, 'Aparici\u00f3n de petequias (peque\u00f1os puntos rojos en la piel) o prueba del torniquete positiva', 'a'),
                (12, 'Disminuci\u00f3n dr\u00e1stica en la cantidad de orina (oliguria)', 'a'),
                (13, 'Debilidad muscular progresiva u hormigueo que asciende por las piernas', 'a'),
                (14, 'Fotofobia (sensibilidad extrema a la luz)', 'a'),
                (15, 'Tos seca persistente acompa\u00f1ada de mareos profundos', 'a'),
            ]
        )

    # ── Siembra de enfermedades ───────────────────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM enfermedad")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO enfermedad (idenfermedad, nombre) VALUES (?, ?)",
            [
                (1, 'DENGUE GRAVE'),
                (2, 'SINDROME PULMONAR POR HANTAVIRUS'),
                (3, 'LEPTOSPIROSIS (ENFERMEDAD DE WEIL)'),
                (4, 'COMPLICACION NEUROLOGICA POR ZIKA (GUILLAIN-BARRE)'),
                (5, 'ZIKA'),
                (6, 'CHIKUNGUNYA'),
                (7, 'DENGUE CLASICO'),
            ]
        )

    # ── Siembra de relaciones ─────────────────────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM enfermedad_sintoma")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO enfermedad_sintoma (idenfermedad, idsintoma) VALUES (?, ?)",
            [
                (1, 1), (1, 9), (1, 10), (1, 11),
                (2, 1), (2, 6), (2, 15),
                (3, 1), (3, 7), (3, 8),  (3, 12),
                (4, 1), (4, 4), (4, 13),
                (5, 1), (5, 4), (5, 5),
                (6, 1), (6, 3), (6, 5),
                (7, 1), (7, 2), (7, 5),  (7, 14),
            ]
        )

    conn.commit()
    conn.close()


# ── Generaci\u00f3n del archivo Prolog ────────────────────────────────────────────

def generar_prolog_desde_db() -> None:
    """Regenera seic.pl a partir de las relaciones en la BD."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.nombre, GROUP_CONCAT(es.idsintoma)
        FROM   enfermedad e
        LEFT JOIN enfermedad_sintoma es ON e.idenfermedad = es.idenfermedad
        GROUP BY e.idenfermedad
        ORDER BY e.idenfermedad
    """)
    rows = cursor.fetchall()
    conn.close()

    lines = [
        ":-dynamic tiene/1.",
        ":-dynamic enfermedad/1.",
        "",
        "lista([]):-enfermedad(E), write(E).",
        "lista([H|T]):-assert(tiene(H)), lista(T).",
        "",
        "test(X) :- limpiar, lista(X).",
        "",
    ]

    for nombre, sintomas_str in rows:
        if sintomas_str:
            conds = ",".join(f"tiene(s{s})" for s in sintomas_str.split(','))
            lines.append(f"enfermedad('{nombre}'):-{conds}.")

    lines += [
        "",
        "enfermedad('No Determinado (Sin patron claro)').",
        "",
        "limpiar:-retract(tiene(_)), fail.",
        "limpiar.",
    ]

    with open(PROLOG_PATH, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))


def reload_prolog() -> None:
    """Regenera el archivo .pl y lo recarga en la instancia activa de Prolog."""
    generar_prolog_desde_db()
    if _prolog is not None:
        _prolog.consult(PROLOG_PATH.replace('\\', '/'))


# ── Inicializaci\u00f3n completa (llamada desde app.py) ───────────────────────────

def init_all() -> Prolog:
    """
    Inicializa la BD, genera el archivo Prolog y crea la instancia singleton.
    Debe llamarse UNA vez antes de registrar los blueprints.
    """
    global _prolog
    init_db()
    generar_prolog_desde_db()
    _prolog = Prolog()
    _prolog.consult(PROLOG_PATH.replace('\\', '/'))
    return _prolog
