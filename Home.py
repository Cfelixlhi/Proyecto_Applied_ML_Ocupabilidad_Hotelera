import streamlit as st

st.set_page_config(
    page_title="Predicción de Ocupabilidad Hotelera",
    page_icon="🏨",
    layout="wide"
)

# ======================================
# Título
# ======================================

st.title("🏨 Predicción de Ocupabilidad Hotelera en el Perú")

st.markdown("""
## Proyecto Final - Applied Machine Learning

Esta aplicación interactiva permite explorar el conjunto de datos de ocupabilidad hotelera del Perú, analizar el desempeño de diferentes modelos de Machine Learning y realizar predicciones utilizando el modelo seleccionado.
""")

st.divider()

# ======================================
# Métricas principales
# ======================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("📊 Registros", "1,800")
c2.metric("🗺 Departamentos", "25")
c3.metric("📅 Periodo", "2019–2024")
c4.metric("🤖 Modelo", "Random Forest")

st.divider()

# ======================================
# Objetivo
# ======================================

st.subheader("🎯 Objetivo")

st.write("""
Predecir la tasa de ocupabilidad hotelera mensual por departamento para apoyar
la planificación y la toma de decisiones en el sector turismo mediante técnicas
de Machine Learning.
""")

# ======================================
# Dataset
# ======================================

st.subheader("📂 Dataset")

st.markdown("""
- **Fuente:** MINCETUR
- **Periodo:** 2019–2024
- **Cobertura:** 25 departamentos del Perú
- **Variable objetivo:** Tasa de ocupabilidad hotelera futura
""")

# ======================================
# Modelo
# ======================================

st.subheader("🤖 Modelo seleccionado")

st.success("""
**Random Forest Regressor**

Fue seleccionado como modelo final por obtener el menor error absoluto medio (MAE)
y el mejor equilibrio entre precisión y capacidad de generalización durante la
evaluación de los algoritmos.
""")

# ======================================
# Navegación
# ======================================

st.subheader("🧭 ¿Qué encontrarás en esta aplicación?")

st.markdown("""
**📊 Análisis Exploratorio**
- Exploración interactiva del dataset.
- Indicadores hoteleros y turísticos.
- Evolución de la ocupabilidad.

**🤖 Modelos Predictivos**
- Comparación de algoritmos.
- Métricas de evaluación.
- Variables más importantes.

**🔮 Predicción**
- Estimación de la tasa de ocupabilidad para un departamento y periodo específicos.
- Comparación entre la predicción y el valor histórico.

**📌 Conclusiones**
- Principales hallazgos.
- Limitaciones del estudio.
- Recomendaciones y trabajo futuro.
""")

st.divider()

st.caption(
    "Proyecto desarrollado para el curso Applied Machine Learning | Universidad de Ingeniería y Tecnología (UTEC)"
)