# 🔬 BioClasificadorAI - Clasificador de Artículos Biomédicos

**Diagrama de diseño:** [v0-ml-pipeline-diagrams.vercel.app](https://v0-ml-pipeline-diagrams.vercel.app)  
**🚀 Aplicación Streamlit en vivo:** [https://jjzm0521-bioclasificadorai-inicio-4px4ao.streamlit.app/](https://jjzm0521-bioclasificadorai-inicio-4px4ao.streamlit.app/)

> 💡 **¡Pruébalo ahora!** Puedes usar la aplicación directamente desde el enlace anterior, sin necesidad de instalar nada localmente.

Este proyecto es una solución para el desafío **"AI + Data Challenge – Tech Sphere 2025"**. Incluye una aplicación web interactiva desarrollada con Streamlit que permite clasificar artículos médicos en cuatro dominios especializados:

- 🫀 **Cardiovascular**
- 🧠 **Neurológico** 
- 🍃 **Hepatorenal**
- 🎗️ **Oncológico**

La clasificación se basa en el **título** y **resumen (abstract)** del artículo científico.

## 🎯 Características Principales

### 📊 **Dos Modelos Disponibles**
1. **SciBERT Fine-Tuneado** - Mayor precisión, ideal para investigación
2. **TF-IDF + SVM Baseline** - Mayor velocidad, ideal para uso general

### 🔧 **Dos Métodos de Uso**
1. **Clasificación Individual** - Introduce título y resumen manualmente
2. **Evaluación Masiva** - Carga un archivo CSV para procesar múltiples artículos

### 📈 **Métricas Detalladas**
- F1-Score ponderado
- Reporte de clasificación completo
- Matrices de confusión por clase
- Visualizaciones interactivas

## 🚀 Cómo Usar la Aplicación Streamlit

### Opción 1: Usar la Aplicación Online (Recomendado)

1. **Accede al enlace**: [https://jjzm0521-bioclasificadorai-inicio-4px4ao.streamlit.app/](https://jjzm0521-bioclasificadorai-inicio-4px4ao.streamlit.app/)
2. **¡Listo!** No necesitas instalar nada

### Opción 2: Ejecutar Localmente

#### Prerrequisitos
- Python 3.8 o superior
- Git

#### Instalación
```bash
# 1. Clonar el repositorio
git clone https://huggingface.co/Rypsor/biomedical-classifier
cd BioClasificadorAI

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación Streamlit
streamlit run inicio.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📝 Guía de Uso Paso a Paso

### 1. **Seleccionar Modelo**
Al iniciar la aplicación, elige entre:
- **SciBERT Fine-Tuneado**: Más preciso, basado en transformers
- **TF-IDF + SVM Baseline**: Más rápido, basado en métodos tradicionales

### 2. **Elegir Método de Entrada**

#### 🔤 **Clasificar un Solo Texto**
1. Selecciona "Clasificar un solo texto"
2. Completa los campos:
   - **Título del Artículo**: Introduce el título del paper científico
   - **Resumen (Abstract)**: Pega el abstract completo
3. Haz clic en **"Clasificar"**
4. Revisa los resultados con niveles de confianza

**Ejemplo de entrada:**
```
Título: "Cardiac Arrhythmias in Elderly Patients"
Resumen: "This study examines the prevalence of cardiac arrhythmias in elderly patients over 65 years old. We analyzed ECG data from 1,200 patients and found that atrial fibrillation was the most common arrhythmia..."
```

#### 📊 **Evaluar un Archivo CSV**
1. Selecciona "Evaluar un archivo CSV"
2. Prepara tu archivo CSV con las siguientes columnas (separado por `;`):
   ```csv
   title;abstract;group
   "Cardiac Function Analysis";"Study of heart function...";cardiovascular
   "Brain Tumor Detection";"Analysis of brain imaging...";neurological,oncological
   ```
3. Sube el archivo usando el botón "Choose a file"
4. Espera el procesamiento automático
5. Revisa las métricas de rendimiento
6. Descarga los resultados con la nueva columna `group_predicted`

### 3. **Interpretar Resultados**

#### Para Texto Individual:
- **Sí (Confianza: X%)**: El artículo pertenece a esa categoría
- **No (Confianza: X%)**: El artículo no pertenece a esa categoría
- **Confianza**: Nivel de certeza del modelo (mayor es mejor)

#### Para Archivo CSV:
- **F1-Score Ponderado**: Métrica principal de rendimiento (0-1, mayor es mejor)
- **Matrices de Confusión**: Visualización de aciertos y errores por categoría
- **group_predicted**: Nueva columna con las predicciones del modelo

## 🛠️ Estructura del Proyecto

```
.
├── inicio.py                          # Aplicación principal de Streamlit
├── requirements.txt                   # Dependencias Python
├── README.md                         # Esta documentación
│
├── modelo-baseline/
│   ├── main.py                       # Pipeline del modelo baseline
│   ├── src/                         # Código fuente baseline
│   ├── data/                        # Datos de entrenamiento
│   └── results/                     # Modelos entrenados
│       └── models/
│           └── baseline_model.joblib # Modelo baseline serializado
│
└── Biomedical-Clasifier/
    └── notebooks/                   # Notebooks del modelo SciBERT
        ├── preprocesamiento.ipynb
        ├── entrenamiento.ipynb
        └── testing.ipynb
```

## 🔧 Formato de Datos Requerido

Para usar la funcionalidad de CSV, tu archivo debe tener:

**Estructura mínima:**
```csv
title;abstract;group
"Título del artículo 1";"Resumen del artículo 1";"cardiovascular"
"Título del artículo 2";"Resumen del artículo 2";"neurological,oncological"
```

**Notas importantes:**
- Usar `;` como separador
- La columna `group` puede contener múltiples etiquetas separadas por comas
- Las etiquetas válidas son: `cardiovascular`, `neurological`, `hepatorenal`, `oncological`
- Usar comillas para textos largos que contengan comas

## 🎯 Casos de Uso

### 👨‍🔬 **Para Investigadores**
- Clasificar nuevos artículos para revisiones sistemáticas
- Organizar literatura por dominios médicos
- Validar la categorización manual de papers

### 📚 **Para Bibliotecarios Médicos**
- Automatizar la catalogación de artículos
- Crear colecciones temáticas especializadas
- Mejorar sistemas de búsqueda y recomendación

### 🏥 **Para Profesionales de la Salud**
- Encontrar literatura relevante por especialidad
- Filtrar investigaciones según área de interés
- Mantenerse actualizado en dominios específicos

## 📊 Stack Tecnológico

### Frontend
- **Streamlit** - Framework de aplicaciones web
- **Matplotlib/Seaborn** - Visualizaciones
- **Pandas** - Manipulación de datos

### Modelos de ML
- **Transformers (Hugging Face)** - SciBERT fine-tuneado
- **Scikit-learn** - Modelo baseline TF-IDF + SVM
- **PyTorch** - Framework de deep learning
- **Joblib** - Serialización de modelos

### Procesamiento
- **NumPy** - Operaciones numéricas
- **JSON** - Manejo de configuraciones
- **OS** - Operaciones del sistema

## 🤝 Contribuir

Si quieres contribuir al proyecto:

1. Fork el repositorio
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🆘 Soporte

¿Tienes preguntas o problemas?

1. **Primero**: Revisa esta documentación
2. **Aplicación Online**: Usa el [enlace directo](https://jjzm0521-bioclasificadorai-inicio-4px4ao.streamlit.app/) para evitar problemas de instalación
3. **Issues**: Abre un issue en el repositorio para problemas técnicos
4. **Contacto**: Reach out through the Hugging Face repository

---

## 🏆 Desarrollado para Tech Sphere 2025

Este proyecto fue desarrollado como solución al **AI + Data Challenge** de Tech Sphere 2025, combinando técnicas modernas de NLP con una interfaz de usuario intuitiva para democratizar el acceso a herramientas de clasificación de literatura biomédica.

**¡Explora la aplicación y descubre el poder de la IA aplicada a la investigación médica!** 🚀
