"""
train_titanic.py — Entrenamiento del Árbol de Decisión para Titanic
=====================================================================
Lógica trasladada del notebook Titanic_AD.ipynb celda por celda.
Genera: miarboltitanic.pkl (modelo) + titanic_tree.png (visualización)
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import io
import base64

# ── Directorio base ──────────────────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))

def entrenar():
    """Ejecuta todo el pipeline de entrenamiento del notebook."""

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 1: Voy a agregar las librerías + cargar CSV
    # ══════════════════════════════════════════════════════════════════════════
    csv_path = os.path.join(_base_dir, 'titanic.csv')
    training = pd.read_csv(csv_path)

    # Renombrar columnas para que coincidan con la lógica del notebook
    training = training.rename(columns={
        'survived': 'Survived',
        'sex': 'Gender',
        'age': 'Age',
        'n_siblings_spouses': 'SibSp',
        'fare': 'Fare',
        'class': 'Pclass',
    })

    # Convertir Pclass de texto a numérico (First=1, Second=2, Third=3)
    pclass_map = {'First': 1, 'Second': 2, 'Third': 3}
    training['Pclass'] = training['Pclass'].map(pclass_map)

    print("=== INFO DEL DATASET ===")
    training.info()

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 2: training.head(5)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== HEAD (5) ===")
    print(training.head(5))

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 3: Convertir Gender a 0/1
    # ══════════════════════════════════════════════════════════════════════════
    training["Gender"] = training["Gender"].apply(lambda toLabel: 0 if toLabel == 'male' else 1)

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 4: training.head(5) después de conversión
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== HEAD (5) después de convertir Gender ===")
    print(training.head(5))

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 5: Rellenar Age nulos con la media
    # ══════════════════════════════════════════════════════════════════════════
    training["Age"].fillna(training["Age"].mean(), inplace=True)
    print("\n=== HEAD (5) después de fillna Age ===")
    print(training.head(5))

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 6: Definir X e Y
    # ══════════════════════════════════════════════════════════════════════════
    Y = training["Survived"].values
    #print(Y)
    columnas = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
    X = training[list(columnas)].values
    print("\n=== X (features) ===")
    print(X)

    # ══════════════════════════════════════════════════════════════════════════
    # Split train/test (el notebook usaba 2 CSVs, aquí hacemos split 80/20)
    # ══════════════════════════════════════════════════════════════════════════
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 7: Entrenar el árbol de decisión
    # ══════════════════════════════════════════════════════════════════════════
    miarbol = tree.DecisionTreeClassifier(criterion="entropy")
    #miarbol = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
    miarbol = miarbol.fit(X_train, Y_train)
    print("\n=== Score en entrenamiento ===")
    print(100 * miarbol.score(X_train, Y_train))
    Ypredecido = miarbol.predict(X_train)

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 8: Predicción de prueba
    # ══════════════════════════════════════════════════════════════════════════
    #columnas = ["Fare","Pclass","Gender","Age","SibSp"]
    respuesta = miarbol.predict([[52.0, 1.0, 0.0, 31.0, 1.0]])
    print("\n=== Predicción de prueba [52.0, 1.0, 0.0, 31.0, 1.0] ===")
    print(respuesta)

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 9: Exportar árbol como imagen
    # ══════════════════════════════════════════════════════════════════════════
    dot_path = os.path.join(_base_dir, 'titanic.dot')
    png_path = os.path.join(_base_dir, 'titanic_tree.png')

    # Generar visualización con matplotlib (no requiere graphviz instalado)
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    tree.plot_tree(
        miarbol,
        feature_names=columnas,
        class_names=["No Sobrevivió", "Sobrevivió"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    ax.set_title("Árbol de Decisión — Titanic (Entropía)", fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n=== Árbol guardado en {png_path} ===")

    # También exportar .dot por si se necesita
    with open(dot_path, 'w') as f:
        tree.export_graphviz(miarbol, out_file=f, feature_names=columnas)
    print(f"=== DOT guardado en {dot_path} ===")

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 10: Testing con datos de prueba
    # ══════════════════════════════════════════════════════════════════════════
    Y_predecido_test = miarbol.predict(X_test)
    score = 0
    for i in range(len(Y_test)):
        if Y_test[i] == Y_predecido_test[i]:
            score = score + 1
    score = 100 * score / len(Y_test)
    print(f"\n=== Score en testing (manual) ===")
    print(score)

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 11: Matriz de confusión - sensibilidad
    # ══════════════════════════════════════════════════════════════════════════
    Matriz_de_confision = confusion_matrix(Y_train, Ypredecido)
    # sensibilidad
    sensibilidad = Matriz_de_confision[0, 0] / np.sum(Matriz_de_confision[0, 0] + Matriz_de_confision[1, 0])
    print(f"\n=== Sensibilidad ===")
    print(sensibilidad)

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 12: Especificidad
    # ══════════════════════════════════════════════════════════════════════════
    # especificidad
    especificidad = Matriz_de_confision[1, 1] / np.sum(Matriz_de_confision[1, 1] + Matriz_de_confision[0, 1])
    print(f"\n=== Especificidad ===")
    print(especificidad)

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 13: Reporte completo de clasificación + gráfico de matriz
    # ══════════════════════════════════════════════════════════════════════════
    conf_mat = confusion_matrix(y_true=Y_train, y_pred=Ypredecido)
    print("\nMatriz de confusion - Datos originales \n", conf_mat)
    print("Metricas de matriz de confusión - Datos originales \n", classification_report(Y_train, Ypredecido))
    labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predecido')
    plt.ylabel('Esperado')
    confusion_png_path = os.path.join(_base_dir, 'confusion_matrix.png')
    fig.savefig(confusion_png_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"=== Matriz de confusión guardada en {confusion_png_path} ===")

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 14: Guardar modelo con pickle
    # ══════════════════════════════════════════════════════════════════════════
    pkl_path = os.path.join(_base_dir, 'miarboltitanic.pkl')
    with open(pkl_path, 'wb') as outfile:
        pickle.dump(miarbol, outfile)
    print("Arbol aprendido guardado correctamente")

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 15: Cargar modelo guardado (verificación)
    # ══════════════════════════════════════════════════════════════════════════
    # abrir lo aprendido
    with open(pkl_path, "rb") as tf:
        miarbol_cargado = pickle.load(tf)
    print("Arbol aprendido cargado correctamente")

    # ══════════════════════════════════════════════════════════════════════════
    # Celda 16-17: Función diagnóstico + predicción final
    # ══════════════════════════════════════════════════════════════════════════
    def diagnostico(valor):
        print(valor)
        if(valor == 1):
            return "Si ha sobrevivido"
        else:
            return "No ha sobrevivido"

    respuesta = miarbol_cargado.predict([[52.0, 1.0, 0.0, 31.0, 1.0]])
    print(diagnostico(respuesta[0]))

    # ══════════════════════════════════════════════════════════════════════════
    # Guardar métricas en un JSON para la web
    # ══════════════════════════════════════════════════════════════════════════
    import json

    # Score de testing con sklearn
    test_score = 100 * miarbol.score(X_test, Y_test)
    train_score = 100 * miarbol.score(X_train, Y_train)

    # Reporte de testing
    report_test = classification_report(Y_test, Y_predecido_test, output_dict=True)
    conf_mat_test = confusion_matrix(Y_test, Y_predecido_test).tolist()

    metrics = {
        "train_score": round(train_score, 2),
        "test_score": round(test_score, 2),
        "sensibilidad": round(float(sensibilidad), 4),
        "especificidad": round(float(especificidad), 4),
        "confusion_matrix_train": conf_mat.tolist(),
        "confusion_matrix_test": conf_mat_test,
        "classification_report_test": {
            "precision_0": round(report_test['0']['precision'], 4),
            "recall_0": round(report_test['0']['recall'], 4),
            "f1_0": round(report_test['0']['f1-score'], 4),
            "precision_1": round(report_test['1']['precision'], 4),
            "recall_1": round(report_test['1']['recall'], 4),
            "f1_1": round(report_test['1']['f1-score'], 4),
            "accuracy": round(report_test['accuracy'], 4),
        },
        "columnas": columnas,
    }

    metrics_path = os.path.join(_base_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"=== Métricas guardadas en {metrics_path} ===")

    print("\n[OK] Entrenamiento completo.")
    return miarbol


# ── Punto de entrada directo ─────────────────────────────────────────────────
if __name__ == '__main__':
    entrenar()
