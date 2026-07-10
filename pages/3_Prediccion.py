import streamlit as st

st.write("Inicio correcto")

import joblib
st.write("Joblib importado correctamente")

import pandas as pd
st.write("Pandas importado correctamente")

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
Seleccione un **departamento**, **año** y **mes** para estimar la tasa de
ocupabilidad hotelera utilizando el modelo de Machine Learning entrenado.
""")

# ==========================================================
# Cargar datos
# ==========================================================

@st.cache_data
def cargar_dataset():
    return pd.read_csv("data/dataset_final.csv")


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ocupabilidad.pkl")


df = cargar_dataset()
modelo = cargar_modelo()

# ==========================================================
# Filtros
# ==========================================================

col1, col2, col3 = st.columns(3)

departamento = col1.selectbox(
    "Departamento",
    sorted(df["DEPARTAMENTO"].dropna().unique())
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
# Predicción
# ==========================================================

if st.button("🔮 Realizar Predicción", use_container_width=True):

    if registro.empty:

        st.error("No existe información para la combinación seleccionada.")

    else:

        fila = registro.iloc[[0]].copy()

        columnas_modelo = list(modelo.feature_names_in_)

        X = fila[columnas_modelo]

        pred = modelo.predict(X)[0]

        valor_real = fila["TASA_OCUPABILIDAD_HOTELERA"].iloc[0]

        error = abs(pred - valor_real)

        st.success("✅ Predicción realizada correctamente.")

        # ==================================================
        # Métricas
        # ==================================================

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicción del modelo",
            f"{pred:.2f}%"
        )

        c2.metric(
            "Valor real observado",
            f"{valor_real:.2f}%"
        )

        c3.metric(
            "Error absoluto",
            f"{error:.2f} pp"
        )

        # ==================================================
        # Evaluación automática
        # ==================================================

        if error <= 1:

            st.success(
                "🟢 Excelente precisión. La diferencia entre el valor real y la predicción es menor a 1 punto porcentual."
            )

        elif error <= 3:

            st.warning(
                "🟡 Buena precisión. Existe una diferencia moderada entre la predicción y el valor observado."
            )

        else:

            st.error(
                "🔴 La diferencia entre la predicción y el valor observado es elevada."
            )

        st.divider()

        # ==================================================
        # Interpretación
        # ==================================================

        st.subheader("📌 Interpretación")

        st.info(
            f"""
El modelo estima que la **tasa de ocupabilidad hotelera**
para el departamento de **{departamento}** durante el mes
**{mes}** del año **{anio}** será de **{pred:.2f}%**.

El valor real observado para ese mismo periodo fue de
**{valor_real:.2f}%**, obteniéndose un error absoluto de
**{error:.2f} puntos porcentuales**.

Esta diferencia evidencia que el modelo logra aproximar
adecuadamente el comportamiento observado de la ocupabilidad hotelera.
"""
        )

        # ==================================================
        # Registro utilizado
        # ==================================================

        st.subheader("📄 Registro utilizado para la predicción")

        columnas = [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "TASA_OCUPABILIDAD_HOTELERA",
            "NUMERO_ESTABLECIMIENTOS",
            "NUMERO_HABITACIONES",
            "NUMERO_PLAZAS_CAMA",
            "PROMEDIO_PERMANENCIA",
            "TOTAL_ARRIBOS",
            "TOTAL_PERNOCT",
            "TOTAL_EMPLEO"
        ]

        st.dataframe(
            fila[columnas],
            use_container_width=True,
            hide_index=True
        )