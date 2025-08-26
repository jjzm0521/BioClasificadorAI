import streamlit as st

# 1. Configuración de la página
# Esto establece el título y el ícono que aparecen en la pestaña del navegador.
st.set_page_config(
    page_title="Comparación de Rendimiento",
    page_icon="📊",
    layout="wide"
)

# 2. Título y descripción en la página
st.title("📊 Comparación de Rendimiento de Modelos")
st.markdown("---")
st.markdown(
    """
    Para analizar las métricas de rendimiento detalladas de los modelos, 
    como el F1-Score, Precisión, Recall y las matrices de confusión de forma interactiva, 
    puedes acceder al siguiente dashboard.
    
    **Haz clic en el botón de abajo para abrir el dashboard en una nueva pestaña.**
    """
)

# 3. Botón que abre el enlace externo en una nueva pestaña
st.link_button(
    label="Abrir Dashboard de Comparación",                # Texto que se muestra en el botón
    url="https://v0-no-content-rho-sepia.vercel.app/",   # La URL a la que se dirigirá
    help="Abre el dashboard de métricas en una nueva pestaña del navegador", # Texto que aparece al pasar el cursor
    type="primary"                                       # Estilo del botón para que destaque
)

# 4. Mensaje informativo adicional (opcional)
st.info("ℹ️ El dashboard se abrirá en una nueva pestaña de tu navegador.", icon="👍")