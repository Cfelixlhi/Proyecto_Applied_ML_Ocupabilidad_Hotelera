import streamlit as st
import pandas as pd

# =====================================================
# Configuración
# =====================================================

st.set_page_config(
    page_title="Conclusiones",
    page_icon="🏁",
    layout="wide"
)

st.title("🏁 Conclusiones del Proyecto")

st.markdown("""
En esta sección se presenta un resumen de los principales resultados obtenidos
durante el desarrollo del proyecto de predicción de la tasa de ocupabilidad hotelera
utilizando técnicas de Machine Learning.
""")

# =====================================================
# Cargar datos
# =====================================================

@st.cache_data
def cargar_dataset():
    return pd.read_csv("data/dataset_final.csv")

@st.cache_data
def cargar_metricas():
    return pd.read_csv("data/metricas_modelos.csv")

@st.cache_data
def cargar_importancia():
    return pd.read_csv("data/importancia_variables.csv")


df = cargar_dataset()
metricas = cargar_metricas()
importancia = cargar_importancia()

# =====================================================
# Información general
# =====================================================

st.subheader("📊 Resumen del Proyecto")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Registros",
    f"{len(df):,}"
)

c2.metric(
    "Departamentos",
    df["DEPARTAMENTO"].nunique()
)

c3.metric(
    "Variables",
    len(df.columns)
)

c4.metric(
    "Periodo",
    f"{df['ANIO'].min()} - {df['ANIO'].max()}"
)

st.divider()

# =====================================================
# Mejor modelo
# =====================================================

mejor = metricas.sort_values("MAE").iloc[0]

st.subheader("🏆 Modelo Seleccionado")

st.success(
    f"""
### {mejor['Modelo']}

El modelo fue seleccionado por presentar el menor error absoluto promedio (MAE)
y un adecuado equilibrio entre precisión y capacidad de generalización.
"""
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", f"{mejor['MAE']:.3f}")
c2.metric("RMSE", f"{mejor['RMSE']:.3f}")
c3.metric("MAPE", f"{mejor['MAPE']:.2f}%")
c4.metric("R²", f"{mejor['R2']:.3f}")

st.divider()

# =====================================================
# Variables importantes
# =====================================================

st.subheader("⭐ Variables más importantes")

top10 = importancia.head(10).copy()

top10["Importancia"] = top10["Importancia"].round(4)

st.dataframe(
    top10,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# Principales hallazgos
# =====================================================

st.subheader("🔍 Principales Hallazgos")

st.info(f"""
- Se evaluaron **{len(metricas)} modelos** de Machine Learning para la predicción de la tasa de ocupabilidad hotelera.

- El modelo **{mejor['Modelo']}** obtuvo el mejor desempeño al registrar el menor MAE (**{mejor['MAE']:.3f}**) y un coeficiente de determinación de **{mejor['R2']:.3f}**.

- La variable con mayor importancia fue **{top10.iloc[0]['Variable']}**, seguida por variables relacionadas con la estacionalidad y el comportamiento histórico de la ocupabilidad.

- La incorporación de variables derivadas como rezagos (lags), media móvil y variables temporales permitió mejorar la capacidad predictiva del modelo.

- Los resultados muestran que la ocupabilidad hotelera presenta un comportamiento estacional y depende significativamente de la información histórica reciente.
""")

st.divider()

# =====================================================
# Aplicación práctica
# =====================================================

st.subheader("💡 Aplicación Práctica")

st.success("""
El modelo desarrollado puede servir como herramienta de apoyo para:

- Estimar la tasa de ocupabilidad hotelera por departamento.
- Apoyar la planificación de infraestructura hotelera.
- Analizar tendencias de la actividad turística.
- Facilitar la toma de decisiones en entidades públicas y privadas relacionadas con el turismo.
""")

st.divider()

# =====================================================
# Trabajo futuro
# =====================================================

st.subheader("🚀 Trabajo Futuro")

st.warning("""
- Incorporar nuevas variables económicas, climáticas y sociales.

- Actualizar periódicamente el modelo con información más reciente.

- Evaluar algoritmos más avanzados de Machine Learning y Deep Learning.

- Implementar pronósticos para periodos futuros mediante modelos especializados de series temporales.
""")

st.divider()

# =====================================================
# Conclusión final
# =====================================================

st.subheader("✅ Conclusión General")

st.success(f"""
El desarrollo de este proyecto permitió demostrar que las técnicas de Machine Learning
constituyen una alternativa efectiva para modelar la tasa de ocupabilidad hotelera en el Perú.

Entre los modelos evaluados, **{mejor['Modelo']}** obtuvo el mejor desempeño al alcanzar
un **MAE de {mejor['MAE']:.3f}**, un **RMSE de {mejor['RMSE']:.3f}**, un
**MAPE de {mejor['MAPE']:.2f}%** y un **R² de {mejor['R2']:.3f}**,
convirtiéndose en el modelo seleccionado para apoyar la estimación de la ocupabilidad hotelera
y contribuir a una mejor planificación del sector turístico.
""")