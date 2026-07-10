import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Configuración
# --------------------------------------------------

st.set_page_config(page_title="Análisis Exploratorio", layout="wide")

st.title("📊 Análisis Exploratorio de los Datos")

st.markdown("""
En esta sección se presenta una exploración interactiva del dataset utilizado
para entrenar el modelo de predicción de la tasa de ocupabilidad hotelera.
""")

# --------------------------------------------------
# Cargar datos
# --------------------------------------------------

@st.cache_data
def cargar_datos():
    return pd.read_csv("data/dataset_final.csv")

df = cargar_datos()

# --------------------------------------------------
# KPIs
# --------------------------------------------------

st.subheader("📌 Información General")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Registros", f"{len(df):,}")
c2.metric("Variables", df.shape[1])
c3.metric("Departamentos", df["DEPARTAMENTO"].nunique())
c4.metric("Periodo", f"{df['ANIO'].min()} - {df['ANIO'].max()}")

st.divider()

# --------------------------------------------------
# Filtros
# --------------------------------------------------

st.subheader("🎛️ Filtros")

col1, col2 = st.columns(2)

departamento = col1.selectbox(
    "Departamento",
    sorted(df["DEPARTAMENTO"].unique())
)

anio = col2.selectbox(
    "Año",
    sorted(df["ANIO"].unique())
)

df_filtrado = df[
    (df["DEPARTAMENTO"] == departamento) &
    (df["ANIO"] == anio)
]

# --------------------------------------------------
# Vista previa
# --------------------------------------------------

st.subheader("📋 Datos Filtrados")

st.dataframe(df_filtrado, use_container_width=True)

st.divider()

# --------------------------------------------------
# Evolución temporal
# --------------------------------------------------

st.subheader("📈 Evolución de la Ocupabilidad Hotelera")

fig = px.line(
    df_filtrado,
    x="MES",
    y="TASA_OCUPABILIDAD_HOTELERA",
    markers=True,
    title=f"{departamento} - {anio}"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Variables hoteleras
# --------------------------------------------------

st.subheader("🏨 Infraestructura Hotelera")

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        df_filtrado,
        x="MES",
        y="NUMERO_ESTABLECIMIENTOS",
        title="Número de establecimientos"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.bar(
        df_filtrado,
        x="MES",
        y="NUMERO_HABITACIONES",
        title="Número de habitaciones"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Turismo
# --------------------------------------------------

st.subheader("✈️ Actividad Turística")

col1, col2 = st.columns(2)

with col1:

    fig = px.line(
        df_filtrado,
        x="MES",
        y="TOTAL_ARRIBOS",
        markers=True,
        title="Total de arribos"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.line(
        df_filtrado,
        x="MES",
        y="TOTAL_PERNOCT",
        markers=True,
        title="Total de pernoctaciones"
    )

    st.plotly_chart(fig, use_container_width=True)