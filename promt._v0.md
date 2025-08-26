# Prompt para V0: Interfaz de Comparación de Clasificadores de Textos Médicos

Crea una interfaz web interactiva y moderna para comparar el rendimiento de dos modelos de clasificación de textos médicos. Los modelos clasifican textos en 4 categorías: **Neurológico**, **Cardiovascular**, **Oncológico**, y **Hepatoadrenal**.

## Contexto de los Modelos:
- **Modelo 1 (Robusto)**: Más consistente y balanceado en todas las categorías
- **Modelo 2 (Desviado)**: Menos consistente, con variaciones significativas entre categorías

## Datos a Mostrar:

### Métricas Globales:
```json
{
  "modelo_robusto": {
    "accuracy_global": 0.85,
    "precision_global": 0.83,
    "recall_global": 0.82,
    "f1_global": 0.825
  },
  "modelo_desviado": {
    "accuracy_global": 0.78,
    "precision_global": 0.75,
    "recall_global": 0.80,
    "f1_global": 0.775
  }
}
```

### Métricas por Categoría:
```json
{
  "neurologico": {
    "modelo_robusto": {"precision": 0.84, "recall": 0.82, "f1": 0.83},
    "modelo_desviado": {"precision": 0.92, "recall": 0.65, "f1": 0.76}
  },
  "cardiovascular": {
    "modelo_robusto": {"precision": 0.83, "recall": 0.85, "f1": 0.84},
    "modelo_desviado": {"precision": 0.68, "recall": 0.88, "f1": 0.77}
  },
  "oncologico": {
    "modelo_robusto": {"precision": 0.82, "recall": 0.81, "f1": 0.815},
    "modelo_desviado": {"precision": 0.85, "recall": 0.72, "f1": 0.78}
  },
  "hepatoadrenal": {
    "modelo_robusto": {"precision": 0.84, "recall": 0.80, "f1": 0.82},
    "modelo_desviado": {"precision": 0.55, "recall": 0.95, "f1": 0.69}
  }
}
```

### Matrices de Confusión:
[AQUÍ INSERTAR TUS MATRICES REALES - Usar formato JSON con estructura clara]

## Requisitos de la Interfaz:

### 1. Diseño Visual:
- **Diseño moderno y profesional** con tema médico/científico
- **Colores diferenciados** para cada modelo (ej: azul para robusto, naranja para desviado)
- **Layout responsive** que funcione en desktop y tablet
- **Tipografía clara** y jerarquía visual bien definida

### 2. Componentes Interactivos:
- **Toggle/Switch** para alternar entre vista de modelo individual y comparación
- **Tabs** para navegar entre: Métricas Globales, Por Categoría, Matrices de Confusión
- **Selector dropdown** para elegir métricas específicas a visualizar
- **Tooltips informativos** explicando cada métrica
- **Hover effects** en gráficos y elementos interactivos

### 3. Visualizaciones:
- **Gráficos de barras comparativos** para métricas por categoría
- **Radar/Spider charts** para comparación multidimensional
- **Heatmaps** para matrices de confusión con colores intuitivos
- **Indicadores KPI** con iconos para métricas globales
- **Gráficos de líneas** para mostrar tendencias si aplica

### 4. Funcionalidades:
- **Resaltar diferencias significativas** entre modelos
- **Filtros por categoría médica**
- **Ordenamiento** de métricas (ascendente/descendente)
- **Exportar/Copiar** datos seleccionados
- **Búsqueda rápida** por métrica o categoría

### 5. UX/UI Específico:
- **Indicadores visuales** de cuál modelo es mejor en cada métrica
- **Alertas/badges** para diferencias críticas entre modelos
- **Animaciones suaves** en transiciones entre vistas
- **Estados de carga** para cambios de datos
- **Leyendas claras** para todos los gráficos

### 6. Información Contextual:
- **Panel de resumen ejecutivo** con insights clave
- **Explicaciones breves** de qué significa cada métrica
- **Recomendaciones automáticas** basadas en los datos
- **Notas metodológicas** sobre la evaluación

## Estructura de Layout Sugerida:
1. **Header**: Título del proyecto y navegación principal
2. **Summary Cards**: KPIs globales de ambos modelos lado a lado  
3. **Control Panel**: Filtros, toggles y opciones de visualización
4. **Main Content**: Área principal con gráficos y tablas
5. **Details Panel**: Información contextual y explicaciones

## Estilo y Tema:
- Usar **gradientes suaves** y **sombras modernas**
- **Iconos médicos** relevantes (estetoscopio, corazón, cerebro, etc.)
- **Colores profesionales**: azules, verdes suaves, grises elegantes
- **Micro-interacciones** que mejoren la experiencia

Implementa esta interfaz usando React con componentes modernos, asegurándote de que sea **intuitiva, informativa y visualmente atractiva** para análisis de rendimiento de modelos de ML médicos.