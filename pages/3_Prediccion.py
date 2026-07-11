import streamlit as st
import pandas as pd

# ==========================================================
# Configuración
# ==========================================================

st.set_page_config(
    page_title="Predicción",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Predicción de Ocupabilidad Hotelera")

st.markdown("""
Seleccione un **departamento**, **año** y **mes** para visualizar la
predicción de la tasa de ocupabilidad hotelera generada por el modelo
Random Forest para el periodo **2025–2027**.
""")

# ==========================================================
# Cargar predicciones
# ==========================================================

@st.cache_data
def cargar_predicciones():
    return pd.read_csv("data/predicciones_2025_2030.csv")

try:
    df = cargar_predicciones()

except Exception as e:
    st.error("No fue posible cargar el archivo de predicciones.")
    st.exception(e)
    st.stop()

# ==========================================================
# Filtros
# ==========================================================

col1, col2, col3 = st.columns(3)

departamento = col1.selectbox(
    "Departamento",
    sorted(df["DEPARTAMENTO"].unique())
)

anio = col2.selectbox(
    "Año",
    sorted(df["ANIO"].unique())
)

mes = col3.selectbox(
    "Mes",
    sorted(df["MES"].unique())
)

# ==========================================================
# Buscar registro
# ==========================================================

registro = df[
    (df["DEPARTAMENTO"] == departamento) &
    (df["ANIO"] == anio) &
    (df["MES"] == mes)
]

# ==========================================================
# Mostrar predicción
# ==========================================================

if st.button("🔮 Realizar Predicción", use_container_width=True):

    if registro.empty:

        st.error("No existe una predicción para la combinación seleccionada.")

    else:

        fila = registro.iloc[0]

        pred = fila["TASA_OCUPABILIDAD_HOTELERA"]

        st.success("✅ Predicción generada correctamente")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Departamento",
            departamento
        )

        c2.metric(
            "Periodo",
            f"{mes}/{anio}"
        )

        c3.metric(
            "Ocupabilidad estimada",
            f"{pred:.2f}%"
        )

        st.divider()

        st.subheader("📌 Interpretación")

        st.info(f"""
El modelo **Random Forest** estima que la tasa de ocupabilidad hotelera
para el departamento de **{departamento}** durante el mes **{mes}**
del año **{anio}** será aproximadamente de **{pred:.2f}%**.

Esta predicción fue obtenida utilizando el modelo entrenado con
información histórica del periodo **2019–2024**, proyectando el
comportamiento esperado para los años **2025–2027**.
""")

        st.divider()

        st.subheader("📄 Información utilizada")

        columnas = [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "NUMERO_ESTABLECIMIENTOS",
            "NUMERO_HABITACIONES",
            "NUMERO_PLAZAS_CAMA",
            "PROMEDIO_PERMANENCIA",
            "TOTAL_ARRIBOS",
            "TOTAL_PERNOCT",
            "TOTAL_EMPLEO"
        ]

        columnas = [c for c in columnas if c in df.columns]

        st.dataframe(
            registro[columnas],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.caption(
            "Las predicciones corresponden a escenarios futuros (2025–2027) generados mediante el modelo Random Forest entrenado con datos históricos de MINCETUR."
        )