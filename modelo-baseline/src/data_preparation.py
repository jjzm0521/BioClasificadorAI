import spacy
import re
import pandas as pd

# Es una buena práctica cargar el modelo una vez y pasarlo a las funciones.
# Lo cargaremos aquí y manejaremos los errores potenciales.
try:
    NLP = spacy.load("en_core_sci_lg", disable=["parser", "ner"])
    NLP.max_length = 2000000 # Aumentar la longitud máxima para resúmenes largos
except OSError:
    print("Error: Modelo 'en_core_sci_lg' no encontrado.")
    print("Por favor, ejecute: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz")
    NLP = None

def preprocess_text(text: str) -> str:
    """
    Limpia, tokeniza y lematiza el texto usando el modelo de NLP scispaCy.
    Elimina las palabras vacías (stop words) y los tokens no alfabéticos.

    Args:
        text (str): La cadena de texto de entrada a procesar.

    Returns:
        str: El texto procesado con los lemas unidos por espacios.
    """
    if NLP is None or not isinstance(text, str):
        return ""

    # Limpieza básica
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Mantener solo letras y espacios
    text = re.sub(r'\s+', ' ', text).strip()

    # Procesar con scispaCy
    doc = NLP(text)

    # Lematizar y eliminar palabras vacías y tokens no alfabéticos
    lemmas = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]

    return " ".join(lemmas)

def binarize_labels(df: pd.DataFrame, domains: list) -> pd.DataFrame:
    """
    Convierte la columna 'group' que contiene cadenas de etiquetas en columnas binarias.

    Args:
        df (pd.DataFrame): El DataFrame de entrada con una columna 'group'.
        domains (list): Una lista de los nombres de los dominios objetivo.

    Returns:
        pd.DataFrame: El DataFrame con nuevas columnas binarias para cada dominio.
    """
    df_copy = df.copy()
    for domain in domains:
        df_copy[domain] = df_copy['group'].apply(
            lambda x: 1 if isinstance(x, str) and domain.lower() in x.lower() else 0
        )
    return df_copy

if __name__ == '__main__':
    # Ejemplo de Uso
    print("Probando funciones de preprocesamiento...")

    # Probar preprocesamiento de texto
    sample_text = "Effects of suprofen on the isolated perfused rat kidney. Although suprofen has been associated with the development of acute renal failure."
    print(f"Original:  {sample_text}")
    processed_text = preprocess_text(sample_text)
    print(f"Procesado: {processed_text}")

    # Probar binarización de etiquetas
    sample_data = {
        'title': ['t1', 't2', 't3', 't4'],
        'abstract': ['a1', 'a2', 'a3', 'a4'],
        'group': ['cardiovascular', 'neurological|hepatorenal', 'oncological', 'cardiovascular|oncological']
    }
    sample_df = pd.DataFrame(sample_data)
    domains_list = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

    print("\nDataFrame Original:")
    print(sample_df)

    binarized_df = binarize_labels(sample_df, domains_list)
    print("\nDataFrame Binarizado:")
    print(binarized_df)

    # Verificar las columnas de salida
    print(f"\nColumnas en el nuevo DataFrame: {binarized_df.columns.tolist()}")
    print("\nPruebas del módulo de preprocesamiento completadas.")
