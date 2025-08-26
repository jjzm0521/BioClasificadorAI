import joblib
import os
from data_preparation import preprocess_text

# --- Configuración ---
# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))
# Subir un nivel a la raíz del proyecto (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Función de Predicción ---
def predict_domains(text: str):
    """
    Carga el modelo entrenado y predice los dominios para un texto dado.

    Args:
        text (str): El texto de entrada a clasificar.

    Returns:
        dict: Un diccionario con las etiquetas predichas y sus probabilidades.
              Devuelve None si no se encuentra el modelo.
    """
    # 1. Cargar Modelo
    try:
        pipeline = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"Error: Archivo del modelo no encontrado en {MODEL_FILE}")
        print("Por favor, ejecute primero el script de entrenamiento (train_model.py).")
        return None

    # 2. Preprocesar el texto de entrada
    processed_text = preprocess_text(text)

    # El pipeline espera un iterable (como una lista o una Serie de pandas)
    text_to_predict = [processed_text]

    # 3. Realizar Predicción
    # Usar predict_proba para obtener las probabilidades de cada clase
    probabilities = pipeline.predict_proba(text_to_predict)

    # Usar predict para obtener las predicciones binarias (0 o 1)
    predictions = pipeline.predict(text_to_predict)

    # 4. Formatear la salida
    # La salida de predict_proba es una lista de arrays, uno por cada clase
    # Tomamos el primer elemento ya que solo predecimos sobre una muestra
    # Las probabilidades son para [clase_0, clase_1] para cada etiqueta. Queremos la prob de la clase 1.

    # La salida de predict es un array 2D, tomamos la primera fila
    predicted_labels = [DOMAINS[i] for i, prediction in enumerate(predictions[0]) if prediction == 1]

    # Crear un diccionario de todas las probabilidades de dominio
    domain_probabilities = {}
    for i, domain in enumerate(DOMAINS):
        # La segunda columna [:, 1] es la probabilidad de la clase positiva (1)
        domain_probabilities[domain] = probabilities[i][0][1]


    return {
        "predicted_labels": predicted_labels,
        "probabilities": domain_probabilities
    }

if __name__ == '__main__':
    # Ejemplo de Uso
    sample_text = "The study investigated the effects of a new drug on cardiac rhythm in patients with heart failure."
    print(f"--- Prediciendo para el texto de ejemplo ---\n'{sample_text}'\n")

    # Esto requiere que el modelo se haya entrenado primero.
    # Agregamos una verificación para ver si el modelo existe antes de ejecutar el ejemplo.
    if not os.path.exists(MODEL_FILE):
        print("Modelo no encontrado. Por favor, entrene el modelo ejecutando 'python main.py' en el directorio 'modelo-baseline' primero.")
    else:
        results = predict_domains(sample_text)
        if results:
            print("--- Resultados de la Predicción ---")
            print(f"Etiquetas Predichas: {results['predicted_labels']}")
            print("\nProbabilidades por Dominio:")
            for domain, prob in results['probabilities'].items():
                print(f"- {domain}: {prob:.4f}")
