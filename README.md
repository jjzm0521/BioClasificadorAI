# BioClasificadorAI

Este proyecto es una solución para el desafío "AI + Data Challenge – Tech Sphere 2025". Contiene dos modelos para la clasificación de artículos médicos en los siguientes dominios:
- Cardiovascular
- Neurológico
- Hepatorenal
- Oncológico

La clasificación se basa en el **título** y el **resumen** del artículo.

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

```
.
├── README.md                 # Documentación del proyecto.
├── requirements.txt          # Dependencias de Python para ambos modelos.
│
├── modelo-baseline/
│   ├── main.py               # Script principal para ejecutar el pipeline del modelo baseline.
│   ├── data/                 # Datos brutos y procesados para el modelo baseline.
│   ├── src/                  # Código fuente del modelo baseline.
│   └── results/              # Modelos entrenados y visualizaciones del baseline.
│
└── Biomedical-Clasifier/
    └── notebooks/            # Notebooks de Jupyter con el pipeline del modelo BioBERT.
        ├── preprocesamiento.ipynb
        ├── entrenamiento.ipynb
        └── testing.ipynb
```

## Cómo Empezar

Siga estos pasos para poner en marcha el proyecto.

### 1. Clonar el Repositorio

```bash
git clone <URL-del-repositorio>
cd BioClasificadorAI
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

## Modelos

### Modelo Baseline

Este es un modelo simple basado en TF-IDF y Regresión Logística.

#### Para Entrenar el Modelo

```bash
python modelo-baseline/main.py
```

Esto ejecutará los siguientes pasos:
1.  **Carga y preprocesamiento de datos**: Lee `challenge_data-18-ago.csv`, limpia el texto, y guarda un archivo intermedio.
2.  **Entrenamiento del modelo**: Entrena un modelo TF-IDF con Regresión Logística y lo guarda.
3.  **Evaluación del modelo**: Carga el modelo entrenado y calcula las métricas de rendimiento.

#### Para Realizar una Predicción

Si ya tienes un modelo entrenado, puedes usar `predict.py` para clasificar un nuevo texto.

```bash
# Ejemplo de cómo se podría usar (el script ya tiene un ejemplo)
python modelo-baseline/src/predict.py
```

### Modelo BioBERT

Este es un modelo más avanzado basado en BioBERT, una versión de BERT pre-entrenada en textos biomédicos. El pipeline completo se encuentra en los notebooks de Jupyter en la carpeta `Biomedical-Clasifier/notebooks/`.

Para ejecutar este modelo, siga los notebooks en el siguiente orden:
1.  `preprocesamiento.ipynb`
2.  `entrenamiento.ipynb`
3.  `testing.ipynb`

## Stack Tecnológico

- **Lenguaje:** Python
- **Librerías Principales:**
  - `pandas` para manipulación de datos.
  - `scikit-learn` y `scikit-multilearn` para el modelo de machine learning.
  - `spacy` y `scispacy` para procesamiento de lenguaje natural (NLP).
  - `torch`, `transformers` y `datasets` para el modelo de deep learning.
  - `optuna` para la optimización de hiperparámetros.
