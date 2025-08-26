import joblib
from sklearn.metrics import f1_score, classification_report, hamming_loss
import pandas as pd
import os

# --- Configuración ---
# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))
# Subir un nivel a la raíz del proyecto (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
TEST_DATA_FILE = os.path.join(project_root, 'results', 'models', 'test_data.joblib')
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Script Principal de Evaluación ---
def evaluate_model():
    """
    Carga un modelo entrenado y datos de prueba, evalúa el modelo
    e imprime las métricas de rendimiento.
    """
    # 1. Cargar Modelo y Datos
    print(f"Cargando modelo desde {MODEL_FILE} y datos de prueba desde {TEST_DATA_FILE}...")
    try:
        pipeline = joblib.load(MODEL_FILE)
        test_data = joblib.load(TEST_DATA_FILE)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar un archivo requerido. {e}")
        print("Por favor, ejecute primero el script de entrenamiento (train_model.py) para generar el modelo y los datos de prueba.")
        return

    # 2. Realizar Predicciones
    print("Realizando predicciones en el conjunto de prueba...")
    y_pred = pipeline.predict(X_test)

    # 3. Calcular e Imprimir Métricas
    print("\n--- Resultados de la Evaluación del Modelo ---")

    # F1-Score Ponderado
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"\nF1-Score Ponderado: {f1_weighted:.4f}")

    # Pérdida de Hamming
    hamming = hamming_loss(y_test, y_pred)
    print(f"Pérdida de Hamming: {hamming:.4f}")
    print("(Menor es mejor. Es la fracción de etiquetas que se predicen incorrectamente.)")

    # Reporte de Clasificación (métricas por clase)
    print("\nReporte de Clasificación (rendimiento por clase):")
    # Usar target_names para etiquetar las clases en el reporte
    report = classification_report(y_test, y_pred, target_names=DOMAINS, zero_division=0)
    print(report)

    print("--- Script de evaluación finalizado. ---")

if __name__ == '__main__':
    evaluate_model()
