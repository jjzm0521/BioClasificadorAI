import pandas as pd
import numpy as np
from data_preparation import preprocess_text, binarize_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split as standard_train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import joblib
import os
import time

# --- Configuración ---
# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))
# Subir un nivel a la raíz del proyecto (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

DATA_FILE = os.path.join(project_root, 'data', 'raw', 'challenge_data-18-ago.csv')
PROCESSED_DATA_FILE = os.path.join(project_root, 'data', 'processed', 'processed_data.pkl')
MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
TEST_DATA_FILE = os.path.join(project_root, 'results', 'models', 'test_data.joblib')

DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
TEXT_COLUMN = 'text'
RANDOM_STATE = 42

# --- Script Principal de Entrenamiento ---
def train_baseline():
    """
    Función principal para cargar, preprocesar y entrenar el modelo de referencia.
    """
    # 1. Cargar y Preprocesar Datos
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Cargando datos preprocesados desde {PROCESSED_DATA_FILE}...")
        df = pd.read_pickle(PROCESSED_DATA_FILE)
    else:
        print(f"Cargando y preprocesando datos desde {DATA_FILE}...")
        try:
            df = pd.read_csv(DATA_FILE, sep=';', engine='python')
        except FileNotFoundError:
            print(f"Error: El archivo {DATA_FILE} no fue encontrado.")
            return

        # Binarizar etiquetas
        df = binarize_labels(df, DOMAINS)

        # Combinar título y resumen
        df['abstract'] = df['abstract'].fillna('')
        df[TEXT_COLUMN] = df['title'] + ' ' + df['abstract']

        # Preprocesar texto (esta es la parte lenta)
        print("Aplicando preprocesamiento de texto a todos los artículos. Esto puede tardar unos minutos...")
        start_time = time.time()
        df['processed_text'] = df[TEXT_COLUMN].apply(preprocess_text)
        end_time = time.time()
        print(f"Preprocesamiento de texto finalizado en {end_time - start_time:.2f} segundos.")

        # Guardar datos procesados para evitar repetir este paso
        df.to_pickle(PROCESSED_DATA_FILE)
        print(f"Datos procesados guardados en {PROCESSED_DATA_FILE}")

    # 2. Dividir Datos
    print("Dividiendo datos en conjuntos de entrenamiento, validación y prueba...")

    X = df[['processed_text']]
    y = df[DOMAINS].values

    # skmultilearn requiere arrays de numpy
    X_np = np.array(X.index).reshape(-1, 1) # Usar el índice para dividir, luego recuperar el texto
    y_np = np.array(y)

    # Primera división: 70% entrenamiento, 30% temporal (validación + prueba)
    X_train_idx, y_train, X_temp_idx, y_temp = iterative_train_test_split(X_np, y_np, test_size=0.3)

    # Segunda división: 15% validación, 15% prueba (50% del 30% temporal)
    X_val_idx, y_val, X_test_idx, y_test = iterative_train_test_split(X_temp_idx, y_temp, test_size=0.5)

    # Recuperar datos de texto usando los índices
    X_train = df.loc[X_train_idx.flatten(), 'processed_text']
    X_val = df.loc[X_val_idx.flatten(), 'processed_text']
    X_test = df.loc[X_test_idx.flatten(), 'processed_text']

    print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"Conjunto de validación: {len(X_val)} muestras")
    print(f"Conjunto de prueba: {len(X_test)} muestras")

    # 3. Definir y Entrenar el Pipeline del Modelo
    print("Definiendo y entrenando el pipeline del modelo de referencia (TF-IDF + Regresión Logística)...")

    # Definir el pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE), n_jobs=-1))
    ])

    # Entrenar el modelo
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    print(f"Entrenamiento del modelo finalizado en {end_time - start_time:.2f} segundos.")

    # 4. Guardar el Modelo
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    print(f"Guardando el modelo entrenado en {MODEL_FILE}...")
    joblib.dump(pipeline, MODEL_FILE)

    # También guardar el conjunto de prueba para la evaluación
    os.makedirs(os.path.dirname(TEST_DATA_FILE), exist_ok=True)
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(test_data, TEST_DATA_FILE)
    print(f"Datos de prueba guardados para evaluación en {TEST_DATA_FILE}")

    print("\n--- ¡Script de entrenamiento finalizado exitosamente! ---")

if __name__ == '__main__':
    train_baseline()
