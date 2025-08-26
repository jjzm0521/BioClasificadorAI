import joblib
from sklearn.metrics import (f1_score, classification_report,
                             precision_score, recall_score, accuracy_score,
                             multilabel_confusion_matrix)
import pandas as pd
import os

# --- Configuración ---
# Obtiene el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un nivel a la raíz del proyecto (de src a la carpeta principal)
project_root = os.path.dirname(script_dir)

MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
TEST_DATA_FILE = os.path.join(project_root, 'results', 'models', 'test_data.joblib')
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Script Principal de Evaluación ---
def evaluate_model():
    """
    Carga un modelo entrenado y datos de prueba, evalúa el modelo
    e imprime un informe de texto formateado con las métricas de rendimiento.
    """
    # 1. Cargar Modelo y Datos
    print("--- Cargando Modelo y Datos ---")
    try:
        pipeline = joblib.load(MODEL_FILE)
        test_data = joblib.load(TEST_DATA_FILE)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        print("Modelo y datos cargados exitosamente.")
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar un archivo requerido. {e}")
        print("Por favor, ejecuta el script de entrenamiento (train_model.py) primero para generar el modelo y los datos de prueba.")
        return

    # 2. Realizar Predicciones
    print("\n--- Realizando Predicciones ---")
    y_pred = pipeline.predict(X_test)
    print("Predicciones realizadas en el conjunto de prueba.")

    # 3. Calcular Todas las Métricas
    print("\n--- Calculando Métricas ---")
    # Métricas Globales
    global_metrics = {
        "f1_score_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "subset_accuracy": accuracy_score(y_test, y_pred)
    }

    # Métricas por Categoría obtenidas de classification_report
    class_report = classification_report(y_test, y_pred, target_names=DOMAINS, zero_division=0, output_dict=True)
    category_metrics = {k: v for k, v in class_report.items() if k in DOMAINS}

    # Matrices de Confusión
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    confusion_matrices = {domain: mcm[i] for i, domain in enumerate(DOMAINS)}
    print("Todas las métricas han sido calculadas.")

    # 4. Generar e Imprimir Informe de Texto
    print("\n--- Generando Informe Final ---")
    text_report = generate_text_report(global_metrics, category_metrics, confusion_matrices)
    print("\n" + "="*80)
    print(text_report)
    print("="*80 + "\n")

    print("--- Script de evaluación finalizado. ---")


def generate_text_report(global_metrics, category_metrics, confusion_matrices):
    """
    Genera un bloque de texto formateado con las métricas de rendimiento del modelo.
    """
    # Título
    report = "Análisis de Rendimiento del Modelo de Clasificación Biomédica\n\n"

    # Sección de Métricas Globales
    report += "Métricas Globales\n"
    report += "-"*20 + "\n"
    report += f"- F1-Score Ponderado: {global_metrics['f1_score_weighted']:.3f}\n"
    report += f"- Precisión Ponderada: {global_metrics['precision_weighted']:.3f}\n"
    report += f"- Recall Ponderado: {global_metrics['recall_weighted']:.3f}\n"
    report += f"- Exactitud de Subconjunto: {global_metrics['subset_accuracy']:.3f}\n\n"

    # Sección de Rendimiento por Categoría
    report += "Rendimiento por Categoría\n"
    report += "-"*28 + "\n"
    # Crea un encabezado de tabla formateado
    header = f"{'Categoría':<15} | {'Precisión':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Soporte':>10}\n"
    report += header
    report += "-"*len(header) + "\n"
    # Crea las filas de la tabla
    for domain, metrics in category_metrics.items():
        report += (f"{domain:<15} | {metrics['precision']:>10.3f} | "
                   f"{metrics['recall']:>10.3f} | {metrics['f1-score']:>10.3f} | "
                   f"{metrics['support']:>10.0f}\n")
    report += "\n"

    # Sección de Matrices de Confusión
    report += "Matrices de Confusión\n"
    report += "-"*24 + "\n"
    for domain, matrix in confusion_matrices.items():
        tn, fp, fn, tp = matrix.ravel()
        report += f"- Matriz para {domain}:\n"
        report += f"  [TN: {tn}, FP: {fp}]\n"
        report += f"  [FN: {fn}, TP: {tp}]\n\n"

    return report.strip()


if __name__ == '__main__':
    evaluate_model()