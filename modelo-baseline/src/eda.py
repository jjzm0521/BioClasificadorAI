import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re

# --- Configuración ---
DATA_FILE = r'modelo-baseline\data\raw\challenge_data-18-ago.csv'
OUTPUT_DIR = r'modelo-baseline\results\images'
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Funciones de Ayuda ---
def clean_text(text):
    """Limpieza básica del texto."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Eliminar etiquetas HTML
    text = re.sub(r'[^a-z\s]', '', text) # Eliminar caracteres especiales y números
    text = re.sub(r'\s+', ' ', text).strip() # Eliminar espacios en blanco adicionales
    return text

def plot_text_length_distribution(df, column_name, output_filename):
    """Grafica y guarda la distribución de la longitud del texto para una columna dada."""
    plt.figure(figsize=(10, 6))
    df[f'{column_name}_length'] = df[column_name].str.len()
    sns.histplot(df[f'{column_name}_length'], bins=50, kde=True)
    plt.title(f'Distribución de la Longitud de {column_name.capitalize()}')
    plt.xlabel('Longitud (número de caracteres)')
    plt.ylabel('Frecuencia')
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()
    print(f"Gráfico de distribución de longitud de {column_name} guardado en {output_filename}")

def generate_word_cloud(text, output_filename):
    """Genera y guarda una nube de palabras a partir de un bloque de texto."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()
    print(f"Nube de palabras guardada en {output_filename}")

# --- Script Principal de EDA ---
def perform_eda():
    """Función principal para ejecutar el Análisis Exploratorio de Datos (EDA)."""
    # Crear el directorio de salida si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Cargar Datos
    print(f"Cargando datos desde {DATA_FILE}...")
    try:
        # Se corrigió el analizador agregando el separador y el motor
        df = pd.read_csv(DATA_FILE, sep=';', engine='python', on_bad_lines='warn')
    except FileNotFoundError:
        print(f"Error: El archivo {DATA_FILE} no fue encontrado en el directorio raíz.")
        return

    print("\n--- Inspección Inicial de Datos ---")
    print("Encabezado de los Datos:")
    print(df.head())
    print("\nInformación de los Datos:")
    df.info()
    print("\nValores Faltantes:")
    print(df.isnull().sum())

    # Rellenar resúmenes faltantes con una cadena vacía
    df['abstract'] = df['abstract'].fillna('')

    # 2. Analizar Etiquetas (Dominios)
    print("\n--- Análisis de Dominios/Etiquetas ---")
    # Binarizar etiquetas
    for domain in DOMAINS:
        df[domain] = df['group'].apply(lambda x: 1 if domain.lower() in x.lower() else 0)

    label_counts = df[DOMAINS].sum().sort_values(ascending=False)
    print("\nConteo de artículos por dominio único:")
    print(label_counts)

    # Graficar la distribución de etiquetas
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Conteo de Artículos por Dominio Médico')
    plt.ylabel('Número de Artículos')
    plt.xlabel('Dominio')
    plt.savefig(os.path.join(OUTPUT_DIR, 'domain_distribution.png'))
    plt.close()
    print("Gráfico de distribución de dominios guardado en domain_distribution.png")

    # Analizar combinaciones de etiquetas
    df['domain_combination'] = df[DOMAINS].apply(lambda row: '|'.join(row.index[row == 1]), axis=1)
    combo_counts = df['domain_combination'].value_counts()
    print("\nConteo de artículos por combinación de dominios:")
    print(combo_counts)

    # 3. Analizar Longitud del Texto
    print("\n--- Análisis de Longitud del Texto ---")
    df['text'] = df['title'] + ' ' + df['abstract']
    plot_text_length_distribution(df, 'title', 'title_length_dist.png')
    plot_text_length_distribution(df, 'abstract', 'abstract_length_dist.png')
    plot_text_length_distribution(df, 'text', 'full_text_length_dist.png')


    # 4. Generar Nubes de Palabras
    print("\n--- Generación de Nubes de Palabras ---")
    # Limpiar texto para las nubes de palabras
    df['cleaned_text'] = df['text'].apply(clean_text)

    for domain in DOMAINS:
        print(f"Generando nube de palabras para {domain}...")
        # Concatenar todo el texto para el dominio dado
        domain_text = " ".join(df[df[domain] == 1]['cleaned_text'])
        if domain_text:
            generate_word_cloud(domain_text, f'wordcloud_{domain.lower()}.png')
        else:
            print(f"No hay texto disponible para generar la nube de palabras para {domain}")

    print("\nScript de EDA finalizado exitosamente.")

if __name__ == '__main__':
    perform_eda()
