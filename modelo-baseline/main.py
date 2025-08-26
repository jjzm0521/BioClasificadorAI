import sys
import os

# Agrega el directorio 'src' al path de Python si es necesario
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# --- 1. Importa las funciones de cada script ---
# Asegúrate de que los nombres de las funciones (ej. prepare_data, run_eda)
# coincidan con los que tienes en tus archivos.
from data_preparation import prepare_data
from eda import run_eda
from train_model import train_baseline
from evaluate_model import evaluate_model

def main():
    """
    Función principal para correr el pipeline de ML completo.
    1. Preparar los datos.
    2. Realizar análisis exploratorio (EDA).
    3. Entrenar el modelo.
    4. Evaluar el modelo.
    """
    print("--- Iniciando Pipeline de ML ---")

    # Paso 1: Preparación de Datos
    print("\n--- Ejecutando Preparación de Datos ---")
    prepare_data()

    # Paso 2: Análisis Exploratorio de Datos (EDA)
    print("\n--- Ejecutando EDA ---")
    run_eda()

    # Paso 3: Entrenamiento del modelo
    print("\n--- Ejecutando Entrenamiento ---")
    train_baseline()

    # Paso 4: Evaluación del modelo
    print("\n--- Ejecutando Evaluación ---")
    # Se mantiene la verificación de que el modelo y los datos de prueba existan
    model_path = os.path.join(os.path.dirname(__file__), 'results', 'models', 'baseline_model.joblib')
    test_data_path = os.path.join(os.path.dirname(__file__), 'results', 'models', 'test_data.joblib')
    if os.path.exists(model_path) and os.path.exists(test_data_path):
        evaluate_model()
    else:
        print("Evaluación omitida: no se encontró el modelo o los datos de prueba.")

    print("\n--- Pipeline de ML Finalizado ---")

if __name__ == '__main__':
    main()