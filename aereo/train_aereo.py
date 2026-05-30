import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

def train_model():
    # 1. Carga de datos
    # Usar el archivo CSV local
    csv_path = os.path.join(os.path.dirname(__file__), 'accidenteaereo.csv')
    air_accident = pd.read_csv(csv_path)

    # 2. Selección de Características (Filtro del profesor)
    columnas_a_eliminar = ['embark_town', 'n_siblings_spouses', 'parch', 'alone']
    air_accident = air_accident.drop(columns=columnas_a_eliminar)

    air_accident_features = air_accident.copy()
    air_accident_labels = air_accident_features.pop('survived')

    # 3. Definición de Entradas Simbólicas
    inputs = {}
    for name, column in air_accident_features.items():
        dtype = tf.string if pd.api.types.is_string_dtype(column) or pd.api.types.is_object_dtype(column) else tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    # 4. Pipeline de Preprocesamiento
    numeric_inputs = {name: val for name, val in inputs.items() if val.dtype == tf.float32}
    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(air_accident[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)

    preprocessed_inputs = [all_numeric_inputs]
    for name, input_item in inputs.items():
        if input_item.dtype == tf.float32:
            continue
        lookup = layers.StringLookup(vocabulary=np.unique(air_accident_features[name]))
        one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
        preprocessed_inputs.append(one_hot(lookup(input_item)))

    # 5. Arquitectura del Modelo
    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
    body = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Salida Logit
    ])
    result = body(preprocessed_inputs_cat)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam', metrics=['accuracy'])

    # 6. Entrenamiento
    features_dict = {name: np.array(value) for name, value in air_accident_features.items()}
    model.fit(x=features_dict, y=air_accident_labels, epochs=15)

    # 7. Guardar el modelo con el nombre correcto
    model_path = os.path.join(os.path.dirname(__file__), 'modelo_entrenado.h5')
    model.save(model_path)
    print(f"\nModelo guardado exitosamente en '{model_path}'")

if __name__ == '__main__':
    train_model()
