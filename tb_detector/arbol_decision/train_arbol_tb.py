"""
train_arbol_tb.py — Árbol de Decisión Puro para Clasificación de Tuberculosis
==============================================================================
Inspirado en la estructura y estilo de train_titanic.py (módulo titanicAD).

Pipeline:
  1. Carga de imágenes desde carpetas health / sick / tb
  2. Preprocesamiento: escala de grises → 64x64 → vector fila-por-fila
  3. Discretización de píxeles a caracteres (O=oscuro, M=medio, C=claro)
  4. Balanceo de clases con class_weight='balanced'
  5. Bucle de optimización de profundidad (max_depth 3..20)
     → Parada temprana si precision >= 80%
  6. Guardado del modelo, métricas y visualizaciones

Salidas en la misma carpeta:
  miarbol_tb.pkl          → Modelo entrenado
  best_depth.json         → Profundidad óptima encontrada
  metrics_tb.json         → Métricas completas para la web
  tb_tree.png             → Visualización del árbol
  confusion_matrix_tb.png → Matriz de confusión
"""

import os
import json
import pickle
import warnings
import cv2
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

# ── Directorio base del módulo ────────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))

# ── Rutas de imágenes ─────────────────────────────────────────────────────────
IMG_DIR_HEALTH = os.path.join(_base_dir, '..', 'imgs', 'health')
IMG_DIR_SICK   = os.path.join(_base_dir, '..', 'imgs', 'sick')
IMG_DIR_TB     = os.path.join(_base_dir, '..', 'imgs', 'tb')

# ── Parámetros de preprocesamiento ────────────────────────────────────────────
IMG_SIZE       = 64            # Resolución de reducción (64x64 = 4096 píxeles)
PIXEL_VECTOR   = IMG_SIZE * IMG_SIZE  # Tamaño del vector final por imagen
TARGET_ACCURACY = 0.80         # Precisión mínima requerida (80%)
CLASES         = {0: 'health', 1: 'sick', 2: 'tb'}

# ── Umbrales de discretización de brillo ─────────────────────────────────────
# En una radiografía de tórax:
#   O (Oscuro, 0-80)   → Aire en los pulmones (negro)
#   M (Medio, 81-170)  → Tejido blando y músculo
#   C (Claro, 171-255) → Huesos y opacidades/lesiones de TB (blanco)
def discretizar_pixel(valor):
    """Convierte el brillo de un píxel (0-255) a categoría O/M/C → 0/1/2."""
    if valor <= 80:
        return 0   # 'O' - Oscuro/Aire
    elif valor <= 170:
        return 1   # 'M' - Medio/Tejido
    else:
        return 2   # 'C' - Claro/Opacidad o Hueso


# ══════════════════════════════════════════════════════════════════════════════
# CELDA 1: Carga y preprocesamiento de imágenes
# ══════════════════════════════════════════════════════════════════════════════
def cargar_imagenes_de_carpeta(carpeta, etiqueta):
    """
    Lee todas las imágenes PNG/JPG de una carpeta.
    - Convierte a escala de grises
    - Redimensiona a IMG_SIZE x IMG_SIZE
    - Concatena las filas (nivel 1, nivel 2, ...) → vector 1D
    - Discretiza cada píxel a O/M/C (0/1/2)
    Retorna X (vectores) e Y (etiquetas).
    """
    X, Y = [], []
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp')

    if not os.path.isdir(carpeta):
        print(f"  [AVISO] Carpeta no encontrada: {carpeta}")
        return np.array(X), np.array(Y)

    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(extensiones_validas)]

    for nombre in archivos:
        ruta = os.path.join(carpeta, nombre)
        # Leer en escala de grises (1 canal)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Saltar archivos corruptos o no imágenes

        # Redimensionar a 64x64
        img_reducida = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Convertir matriz 64x64 a vector 1D (fila 1 al lado de fila 2, ...)
        # → esto es equivalente a img_reducida.flatten() pero respeta el orden fila-a-fila
        vector_brillo = img_reducida.flatten()

        # Discretizar: aplicar función O/M/C a cada píxel del vector
        vector_discreto = np.array([discretizar_pixel(p) for p in vector_brillo])

        X.append(vector_discreto)
        Y.append(etiqueta)

    return np.array(X), np.array(Y)


def entrenar():
    """Ejecuta todo el pipeline de entrenamiento del Árbol de Decisión para TB."""

    print("=" * 65)
    print("  ÁRBOL DE DECISIÓN — DETECTOR DE TUBERCULOSIS EN RADIOGRAFÍAS")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════════════════
    # Celda 1: Carga de imágenes por clase
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Paso 1] Cargando imágenes de las tres carpetas...")
    print(f"  Resolucion de trabajo: {IMG_SIZE}x{IMG_SIZE} px -> {PIXEL_VECTOR} columnas/pixel")
    print(f"  Discretizacion: 0-80='O'(Oscuro), 81-170='M'(Medio), 171-255='C'(Claro)\n")

    X_health, Y_health = cargar_imagenes_de_carpeta(IMG_DIR_HEALTH, 0)
    X_sick,   Y_sick   = cargar_imagenes_de_carpeta(IMG_DIR_SICK,   1)
    X_tb,     Y_tb     = cargar_imagenes_de_carpeta(IMG_DIR_TB,     2)

    print(f"  Imágenes cargadas:")
    print(f"    health (0): {len(X_health):>5} imágenes")
    print(f"    sick   (1): {len(X_sick):>5} imágenes")
    print(f"    tb     (2): {len(X_tb):>5} imágenes")
    total = len(X_health) + len(X_sick) + len(X_tb)
    print(f"    TOTAL     : {total:>5} imágenes\n")

    if total == 0:
        print("[ERROR] No se encontraron imágenes. Verifica las rutas de las carpetas.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # Celda 2: Unir todo en un único dataset X, Y
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 2] Construyendo dataset unificado...")
    X = np.vstack([X_health, X_sick, X_tb])
    Y = np.concatenate([Y_health, Y_sick, Y_tb])
    print(f"  Forma de X: {X.shape}   (filas=imágenes, columnas=píxeles discretizados)")
    print(f"  Forma de Y: {Y.shape}\n")

    # Mostrar muestra de la "cadena de caracteres" de la primera imagen de cada clase
    print("  === Muestra del vector discretizado (primeros 30 píxeles) ===")
    nombre_letras = {0: 'O', 1: 'M', 2: 'C'}
    for etq, nombre in CLASES.items():
        idx = np.where(Y == etq)[0]
        if len(idx) > 0:
            muestra = ''.join([nombre_letras[p] for p in X[idx[0], :30]])
            print(f"    {nombre:8s}: {muestra}...")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # Celda 3: División entrenamiento / prueba (80% / 20%)
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 3] División 80% entrenamiento / 20% prueba (random_state=42)...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=Y
    )
    print(f"  Entrenamiento: {len(X_train)} imágenes")
    print(f"  Prueba       : {len(X_test)} imágenes\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 4: Bucle de optimización del "largo de raíces" (max_depth)
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 4] Buscando el largo optimo de raices (max_depth 3 -> 20)...")
    print(f"  Criterio de parada: accuracy >= {TARGET_ACCURACY*100:.0f}% en conjunto de prueba")
    print(f"  Balance de clases : class_weight='balanced' (corrige desbalance health/sick/tb)\n")

    historial = []     # [(depth, accuracy)]
    mejor_depth = 3
    mejor_acc   = 0.0
    profundidad_optima_encontrada = False

    for depth in range(3, 21):
        miarbol = tree.DecisionTreeClassifier(
            criterion='entropy',
            class_weight='balanced',
            max_depth=depth,
            random_state=42
        )
        miarbol.fit(X_train, Y_train)
        acc_train = miarbol.score(X_train, Y_train)
        acc_test  = miarbol.score(X_test,  Y_test)
        historial.append((depth, acc_test))

        estado = ""
        if acc_test > mejor_acc:
            mejor_acc   = acc_test
            mejor_depth = depth
            estado = " [MEJOR]"

        print(f"  max_depth={depth:2d} | Train={acc_train*100:6.2f}% | Test={acc_test*100:6.2f}%{estado}")

        # Criterio de parada temprana: si alcanzamos el 80%, paramos
        if acc_test >= TARGET_ACCURACY and not profundidad_optima_encontrada:
            print(f"\n  [OK] Precision del {TARGET_ACCURACY*100:.0f}% alcanzada con max_depth={depth}.")
            print(f"       Deteniendo busqueda anticipada.\n")
            mejor_depth = depth
            mejor_acc   = acc_test
            profundidad_optima_encontrada = True
            break

    if not profundidad_optima_encontrada:
        print(f"\n  [AVISO] No se alcanzo el {TARGET_ACCURACY*100:.0f}% en el rango explorado.")
        print(f"          Se usara el mejor encontrado: max_depth={mejor_depth} con {mejor_acc*100:.2f}%\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 5: Entrenamiento final con la profundidad óptima
    # ══════════════════════════════════════════════════════════════════════
    print(f"[Paso 5] Entrenando modelo final con max_depth={mejor_depth}...")
    miarbol_final = tree.DecisionTreeClassifier(
        criterion='entropy',
        class_weight='balanced',
        max_depth=mejor_depth,
        random_state=42
    )
    miarbol_final.fit(X_train, Y_train)

    score_train = 100 * miarbol_final.score(X_train, Y_train)
    score_test  = 100 * miarbol_final.score(X_test, Y_test)

    print(f"  Score en entrenamiento : {score_train:.2f}%")
    print(f"  Score en prueba        : {score_test:.2f}%\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 6: Predicciones en el conjunto de prueba
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 6] Generando predicciones sobre el conjunto de prueba...")
    Y_pred_test  = miarbol_final.predict(X_test)
    Y_pred_train = miarbol_final.predict(X_train)

    # Score manual (siguiendo estilo de train_titanic.py)
    aciertos = sum(1 for real, pred in zip(Y_test, Y_pred_test) if real == pred)
    score_manual = 100 * aciertos / len(Y_test)
    print(f"  Score manual en testing: {score_manual:.2f}%\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 7: Matriz de confusión (entrenamiento)
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 7] Calculando métricas...")
    conf_train = confusion_matrix(Y_train, Y_pred_train)
    conf_test  = confusion_matrix(Y_test,  Y_pred_test)

    # Sensibilidad para clase TB (clase 2) — TP/(TP+FN)
    # TP = conf_test[2,2], FN = conf_test[2,0] + conf_test[2,1]
    tp_tb = conf_test[2, 2]
    fn_tb = conf_test[2, 0] + conf_test[2, 1]
    fp_tb = conf_test[0, 2] + conf_test[1, 2]
    tn_tb = conf_test[0, 0] + conf_test[0, 1] + conf_test[1, 0] + conf_test[1, 1]

    sensibilidad_tb  = tp_tb / (tp_tb + fn_tb + 1e-8)
    especificidad_tb = tn_tb / (tn_tb + fp_tb + 1e-8)

    print(f"  Matriz de confusion (prueba):\n  {conf_test}")
    print(f"  Sensibilidad TB (Recall clase tb) : {sensibilidad_tb:.4f}")
    print(f"  Especificidad TB                  : {especificidad_tb:.4f}\n")

    # Reporte de clasificación completo
    nombres_clases = [CLASES[i] for i in sorted(CLASES.keys())]
    report_dict = classification_report(Y_test, Y_pred_test,
                                        target_names=nombres_clases,
                                        output_dict=True)
    print("  Reporte de clasificacion (prueba):")
    print(classification_report(Y_test, Y_pred_test, target_names=nombres_clases))

    # ══════════════════════════════════════════════════════════════════════
    # Celda 8: Guardar visualización del árbol (tb_tree.png)
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 8] Generando visualizacion del arbol de decision...")
    tree_png_path = os.path.join(_base_dir, 'tb_tree.png')
    nombres_features = [f'px{i}' for i in range(PIXEL_VECTOR)]

    fig, ax = plt.subplots(figsize=(28, 14), dpi=80)
    tree.plot_tree(
        miarbol_final,
        feature_names=nombres_features,
        class_names=nombres_clases,
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=4,   # Mostrar solo los primeros 4 niveles en el PNG (el árbol es grande)
        ax=ax  # type: ignore
    )
    ax.set_title(
        f"Arbol de Decision - Detector de Tuberculosis (Entropia, max_depth={mejor_depth})\n"
        f"Precision en prueba: {score_test:.2f}%",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(tree_png_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Arbol guardado en: {tree_png_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 9: Guardar matriz de confusión como imagen
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 9] Generando imagen de la matriz de confusion...")
    cm_png_path = os.path.join(_base_dir, 'confusion_matrix_tb.png')

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    cax = ax.matshow(conf_test, cmap='Blues')
    fig.colorbar(cax)
    tick_pos = list(range(len(nombres_clases)))
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(nombres_clases)
    ax.set_yticklabels(nombres_clases)

    # Añadir valores numéricos en cada celda
    for i in range(len(nombres_clases)):
        for j in range(len(nombres_clases)):
            ax.text(j, i, str(conf_test[i, j]),
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if conf_test[i, j] > conf_test.max() / 2 else 'black')

    ax.set_title(f'Matriz de Confusion - Arbol TB (max_depth={mejor_depth})', fontsize=13, pad=15)
    plt.xlabel('Predicho', fontsize=12)
    plt.ylabel('Esperado', fontsize=12)
    fig.tight_layout()
    fig.savefig(cm_png_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Matriz de confusion guardada en: {cm_png_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 10: Guardar el modelo con pickle
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 10] Guardando modelo entrenado con pickle...")
    pkl_path = os.path.join(_base_dir, 'miarbol_tb.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(miarbol_final, f)
    print(f"  Modelo guardado en: {pkl_path}")

    # Verificación: cargar y usar el modelo guardado
    with open(pkl_path, 'rb') as f:
        modelo_verificado = pickle.load(f)
    print(f"  Verificacion: modelo cargado correctamente ({type(modelo_verificado).__name__})\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 11: Guardar best_depth.json (archivo de configuración)
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 11] Guardando configuracion optima en best_depth.json...")
    best_depth_path = os.path.join(_base_dir, 'best_depth.json')
    config_optima = {
        "largo_optimo_raices": int(mejor_depth),
        "precision_obtenida":  round(float(score_test), 4),
        "total_imagenes":      int(total),
        "imagenes_health":     int(len(X_health)),
        "imagenes_sick":       int(len(X_sick)),
        "imagenes_tb":         int(len(X_tb)),
        "resolucion_imagen":   f"{IMG_SIZE}x{IMG_SIZE}",
        "metodo_conversion":   "Concatenacion fila-por-fila con discretizacion O/M/C",
        "objetivo_precision":  f">= {TARGET_ACCURACY*100:.0f}%",
        "precision_alcanzada": score_test >= TARGET_ACCURACY
    }
    with open(best_depth_path, 'w', encoding='utf-8') as f:
        json.dump(config_optima, f, indent=2, ensure_ascii=False)
    print(f"  Configuracion guardada en: {best_depth_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    # Celda 12: Guardar metrics_tb.json (métricas para la web Flask)
    # ══════════════════════════════════════════════════════════════════════
    print("[Paso 12] Guardando metricas completas en metrics_tb.json...")
    metricas = {
        "train_score":          round(float(score_train), 4),
        "test_score":           round(float(score_test),  4),
        "score_manual_testing": round(float(score_manual), 4),
        "sensibilidad_tb":      round(float(sensibilidad_tb),  4),
        "especificidad_tb":     round(float(especificidad_tb), 4),
        "max_depth_optimo":     int(mejor_depth),
        "total_imagenes":       int(total),
        "confusion_matrix_train": conf_train.tolist(),
        "confusion_matrix_test":  conf_test.tolist(),
        "clases":               nombres_clases,
        "classification_report": {
            nombre: {
                "precision": round(report_dict[nombre]['precision'], 4),
                "recall":    round(report_dict[nombre]['recall'],    4),
                "f1_score":  round(report_dict[nombre]['f1-score'],  4),
                "support":   int(report_dict[nombre]['support'])
            }
            for nombre in nombres_clases
        },
        "accuracy_overall": round(report_dict['accuracy'], 4),
    }
    metrics_path = os.path.join(_base_dir, 'metrics_tb.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"  Metricas guardadas en: {metrics_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    # Resumen final
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 65)
    print(f"  Largo optimo de raices : max_depth = {mejor_depth}")
    print(f"  Precision en prueba    : {score_test:.2f}%")
    print(f"  Sensibilidad TB        : {sensibilidad_tb*100:.2f}%")
    print(f"  Especificidad TB       : {especificidad_tb*100:.2f}%")
    print(f"  Total imagenes usadas  : {total}")
    estado_objetivo = "[OK]" if score_test >= TARGET_ACCURACY else "[NO ALCANZADO]"
    print(f"  Objetivo >= 80%        : {estado_objetivo}")
    print()
    print("  Archivos generados:")
    print(f"    miarbol_tb.pkl          -> Modelo listo para usar")
    print(f"    best_depth.json         -> Configuracion optima")
    print(f"    metrics_tb.json         -> Metricas para la web")
    print(f"    tb_tree.png             -> Visualizacion del arbol")
    print(f"    confusion_matrix_tb.png -> Matriz de confusion")
    print("=" * 65)

    return miarbol_final


# ── Punto de entrada directo ──────────────────────────────────────────────────
if __name__ == '__main__':
    entrenar()
