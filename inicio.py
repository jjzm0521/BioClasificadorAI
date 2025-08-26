import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
import joblib
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# ⚙️ 1. CONFIGURACIÓN INICIAL Y CARGA DE RECURSOS
# ==============================================================================

st.set_page_config(page_title="Clasificador Biomédico", layout="wide")

# --- Rutas y Nombres de Modelo ---
REPO_ID = "Rypsor/biomedical-classifier"
BASELINE_MODEL_PATH = "modelo-baseline/results/models/baseline_model.joblib"

# --- Definir listas de etiquetas separadas ---
ADVANCED_MODEL_LABELS = ['cardiovascular', 'hepatorenal', 'neurological', 'oncological']
BASELINE_MODEL_LABELS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Carga de Recursos con Caché ---
@st.cache_resource
def load_advanced_model():
    """Carga el modelo SciBERT, tokenizador y umbrales desde Hugging Face Hub."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(REPO_ID).to(device)
        tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
        thresholds_path = hf_hub_download(repo_id=REPO_ID, filename="thresholds.json")
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
        return model, tokenizer, thresholds, device
    except Exception as e:
        st.error(f"Error al cargar el modelo avanzado: {e}")
        return None, None, None, None

@st.cache_resource
def load_baseline_model():
    """Carga el pipeline del modelo baseline desde un archivo .joblib."""
    try:
        pipeline = joblib.load(BASELINE_MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error al cargar el modelo baseline: {e}")
        return None

# ==============================================================================
# 헬 2. FUNCIONES AUXILIARES (COMPARTIDAS Y NUEVAS)
# ==============================================================================

def preprocess_dataframe(df, label_names):
    """Aplica el preprocesamiento al CSV cargado."""
    def unir_texto(title, abstract):
        title = str(title).strip()
        abstract = str(abstract).strip()
        return f"{title}. {abstract}" if not title.endswith(".") else f"{title} {abstract}"

    df["text"] = df.apply(lambda row: unir_texto(row["title"], row["abstract"]), axis=1)
    df["group"] = df["group"].str.lower().str.strip()
    
    for label in label_names:
        df[label.lower()] = df["group"].apply(lambda x: 1 if label.lower() in x else 0)
    return df

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])

# --- FUNCIÓN NUEVA Y CORREGIDA ---
def predict_on_df(df, text_column, model, tokenizer):
    """Realiza inferencia en un DataFrame usando el modelo y tokenizador de HF."""
    # Preparamos los argumentos del Trainer (necesarios para la predicción)
    training_args = TrainingArguments(
        output_dir='./temp_results', # Directorio temporal
        per_device_eval_batch_size=16, # Ajusta según la VRAM de tu GPU
        do_predict=True,
        report_to="none"
    )

    # Tokenizamos todos los textos del DataFrame
    texts = df[text_column].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

    # Creamos el dataset de inferencia usando tu clase
    dataset = InferenceDataset(encodings)

    # Creamos una instancia del Trainer solo para predicción
    trainer = Trainer(model=model, args=training_args)

    # Obtenemos las predicciones (que son logits)
    raw_predictions = trainer.predict(dataset)
    logits = raw_predictions.predictions

    # Para clasificación multietiqueta, aplicamos la función sigmoide para obtener probabilidades
    sigmoid = torch.nn.Sigmoid()
    probs_tensor = sigmoid(torch.from_numpy(logits))
    probabilities = probs_tensor.cpu().numpy()

    return probabilities

def labels_to_str(labels_array, class_names):
    predicted_groups = []
    for row in labels_array:
        names = [class_names[i] for i, label in enumerate(row) if label == 1]
        predicted_groups.append(", ".join(names) if names else "ninguna")
    return predicted_groups

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def display_metrics(y_true, y_pred, class_names):
    st.subheader("Métricas de Desempeño")
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    st.metric("F1-Score Ponderado (Métrica Principal)", f"{f1_w:.4f}")
    
    st.text("Reporte de Clasificación Detallado:")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    st.code(report)
    
    st.subheader("Matrices de Confusión por Clase")
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, len(class_names), figsize=(20, 5))
    for i, matrix in enumerate(conf_matrices):
        label = class_names[i].capitalize()
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['Real Neg', 'Real Pos'])
        axes[i].set_title(label)
    plt.tight_layout()
    st.pyplot(fig)

# ==============================================================================
# 🎨 3. INTERFAZ DE USUARIO PRINCIPAL (VERSIÓN CORREGIDA)
# ==============================================================================

st.title("🔬 Clasificador de Artículos de Investigación Biomédica")
st.markdown("---")
st.subheader("Paso 1: Elige el modelo a utilizar")

model_choice = st.selectbox(
    "Modelo:",
    ("SciBERT (Fine-Tuneado)", "TF-IDF + SVM Baseline")
)
st.markdown("---")

# --- LÓGICA PARA EL MODELO AVANZADO (SCI-BERT) ---
if "SciBERT" in model_choice:
    model, tokenizer, umbrales, device = load_advanced_model()
    if model:
        st.subheader("Paso 2: Elige el método de entrada")
        
        input_method = st.radio(
            "Método:", 
            ("Clasificar un solo texto", "Evaluar un archivo CSV"), 
            horizontal=True, 
            key="advanced_method"
        )

        if input_method == "Clasificar un solo texto":
            input_title = st.text_input("Título del Artículo", key="advanced_title")
            input_abstract = st.text_area("Resumen (Abstract)", height=200, key="advanced_abstract")
            if st.button("Clasificar", type="primary", key="advanced_classify_btn"):
                if input_title and input_abstract:
                    with st.spinner('Clasificando con SciBERT...'):
                        combined_text = f"{input_title}. {input_abstract}"
                        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            logits = model(**inputs).logits
                        probabilidades = torch.sigmoid(logits).cpu().numpy()[0]
                        resultados = {}
                        for i, label in enumerate(umbrales.keys()):
                            if probabilidades[i] >= umbrales[label]:
                                resultados[label.capitalize()] = f"**Sí** (Confianza: {probabilidades[i]:.2%})"
                            else:
                                resultados[label.capitalize()] = f"No (Confianza: {probabilidades[i]:.2%})"
                    st.subheader("Resultados de la Clasificación:")
                    st.table(resultados)
        
        elif input_method == "Evaluar un archivo CSV":
            uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv", key="advanced_uploader")
            if uploaded_file:
                with st.status("Procesando con modelo SciBERT...", expanded=True) as status:
                    status.update(label="Preprocesando datos...")
                    df_original = pd.read_csv(uploaded_file, sep=";")
                    df_original.columns = [col.lower().strip() for col in df_original.columns]
                    df_procesado = preprocess_dataframe(df_original.copy(), ADVANCED_MODEL_LABELS)
                    
                    status.update(label="Realizando predicciones...")
                    # --- LLAMADA CORREGIDA A LA FUNCIÓN ---
                    probabilities = predict_on_df(df_procesado, "text", model, tokenizer)
                    
                    status.update(label="Calculando métricas...")
                    umbrales_array = np.array([umbrales[name] for name in umbrales.keys()])
                    predicted_labels_binary = (probabilities >= umbrales_array).astype(int)
                    true_labels_binary = df_procesado[ADVANCED_MODEL_LABELS].values
                    df_procesado['group_predicted'] = labels_to_str(predicted_labels_binary, ADVANCED_MODEL_LABELS)
                    
                    status.update(label="¡Proceso completado!", state="complete", expanded=False)

                display_metrics(true_labels_binary, predicted_labels_binary, ADVANCED_MODEL_LABELS)
                
                st.subheader("Resultados con la Columna 'group_predicted'")
                st.dataframe(df_procesado[['title', 'abstract', 'group', 'group_predicted']])
                
                csv_final = convert_df_to_csv(df_procesado)
                st.download_button(label="Descargar CSV con Predicciones", data=csv_final, file_name="resultados_scibert.csv", mime="text/csv")


# --- LÓGICA PARA EL MODELO BASELINE (TF-IDF + SVM) ---
elif "Baseline" in model_choice:
    baseline_pipeline = load_baseline_model()
    if baseline_pipeline:
        st.subheader("Paso 2: Elige el método de entrada")
        input_method = st.radio(
            "Método:", 
            ("Clasificar un solo texto", "Evaluar un archivo CSV"), 
            horizontal=True, 
            key="baseline_method"
        )

        if input_method == "Clasificar un solo texto":
            input_title = st.text_input("Título del Artículo", key="baseline_title")
            input_abstract = st.text_area("Resumen (Abstract)", height=200, key="baseline_abstract")
            if st.button("Clasificar", type="primary", key="baseline_classify_btn"):
                if input_title and input_abstract:
                    with st.spinner('Clasificando con modelo Baseline...'):
                        combined_text = f"{input_title}. {input_abstract}"
                        prediction_binary = baseline_pipeline.predict([combined_text])
                        predicted_labels = labels_to_str(prediction_binary, BASELINE_MODEL_LABELS)
                    st.subheader("Clasificación Predicha:")
                    st.success(predicted_labels[0] if predicted_labels else "ninguna")
        
        elif input_method == "Evaluar un archivo CSV":
            uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv", key="baseline_uploader")
            if uploaded_file:
                with st.status("Procesando con modelo Baseline...", expanded=True) as status:
                    status.update(label="Preprocesando datos...")
                    df_original = pd.read_csv(uploaded_file, sep=";")
                    df_original.columns = [col.lower().strip() for col in df_original.columns]
                    df_procesado = preprocess_dataframe(df_original.copy(), BASELINE_MODEL_LABELS)
                    
                    status.update(label="Realizando predicciones...")
                    predicted_labels_binary = baseline_pipeline.predict(df_procesado['text'])
                    true_labels_binary = df_procesado[[label.lower() for label in BASELINE_MODEL_LABELS]].values
                    df_procesado['group_predicted'] = labels_to_str(predicted_labels_binary, BASELINE_MODEL_LABELS)
                    
                    status.update(label="¡Proceso completado!", state="complete", expanded=False)

                display_metrics(true_labels_binary, predicted_labels_binary, BASELINE_MODEL_LABELS)
                
                st.subheader("Resultados con la Columna 'group_predicted'")
                st.dataframe(df_procesado[['title', 'abstract', 'group', 'group_predicted']])
                
                csv_final = convert_df_to_csv(df_procesado)
                st.download_button(label="Descargar CSV con Predicciones", data=csv_final, file_name="resultados_baseline.csv", mime="text/csv")