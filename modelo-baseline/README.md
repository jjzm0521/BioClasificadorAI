# Reto-Tech-Sphere: Clasificación de Artículos Médicos

Este proyecto es una solución para el desafío "AI + Data Challenge – Tech Sphere 2025".

## Descripción del Desafío

El "AI + Data Challenge – Tech Sphere 2025" es un desafío de clasificación biomédica con inteligencia artificial. El objetivo es construir una solución de IA para clasificar artículos médicos en uno o varios de los siguientes dominios:
- Cardiovascular
- Neurológico
- Hepatorenal
- Oncológico

La clasificación se debe basar únicamente en el **título** y el **resumen** del artículo.

## Público Objetivo

El desafío está dirigido a:
- Perfiles junior y recién graduados en carreras STEM.
- Profesionales en transición hacia la ciencia de datos.
- Desarrolladores incursionando en el campo de la inteligencia artificial.

---

## Estructura del Proyecto

Este repositorio está organizado de la siguiente manera para asegurar un código limpio, modular y reproducible.

```
modelo-baseline/
│
├── README.md           # Documentación del proyecto.
├── requirements.txt    # Lista de librerías de Python necesarias.
├── main.py             # Script principal para ejecutar el pipeline completo.
│
├── data/
│   ├── raw/            # Aquí van los datos originales (ej. challenge_data-18-ago.csv).
│   └── processed/      # Aquí se guardan los datos limpios y listos para el modelo.
│
├── src/
│   ├── data_preparation.py # Scripts para limpiar y preparar los datos.
│   ├── train_model.py      # Script para entrenar el modelo baseline.
│   ├── evaluate_model.py   # Script para evaluar el modelo con el set de test.
│   └── predict.py          # Script para hacer predicciones con un modelo entrenado.
│
└── results/
    ├── figures/          # Para guardar gráficos y visualizaciones.
    └── models/           # Para guardar los modelos entrenados (ej. baseline_model.joblib).
```

## Cómo Empezar

Sigue estos pasos para poner en marcha el proyecto y entrenar el modelo.

### 1. Clonar el Repositorio

```bash
git clone <URL-del-repositorio>
cd modelo-baseline
```

### 2. Instalar Dependencias

Se recomienda crear un entorno virtual para mantener las dependencias aisladas.

```bash
# Crear un entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar las librerías
pip install -r requirements.txt
```

### 3. Ejecutar el Pipeline Completo

El script `main.py` se encarga de todo el proceso: entrena el modelo y luego lo evalúa.

```bash
python main.py
```

Esto ejecutará los siguientes pasos:
1.  **Carga y preprocesamiento de datos**: Lee `challenge_data-18-ago.csv`, limpia el texto, y guarda un archivo intermedio en `data/processed/`.
2.  **Entrenamiento del modelo**: Entrena un modelo TF-IDF con Regresión Logística y lo guarda en `results/models/`.
3.  **Evaluación del modelo**: Carga el modelo entrenado y el conjunto de datos de prueba para calcular y mostrar las métricas de rendimiento.

### 4. Realizar una Predicción

Si ya tienes un modelo entrenado, puedes usar `predict.py` para clasificar un nuevo texto.

```bash
# Ejemplo de cómo se podría usar (el script ya tiene un ejemplo)
python src/predict.py
```

---

## Stack Tecnológico

- **Lenguaje:** Python
- **Librerías Principales:**
  - `pandas` para manipulación de datos.
  - `scikit-learn` y `scikit-multilearn` para el modelo de machine learning.
  - `spacy` y `scispacy` para procesamiento de lenguaje natural (NLP).
