import streamlit as st
import pandas as pd
import joblib

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


try:
    df = cargar_dataset()
    modelo = cargar_modelo()

except Exception as e:
    st.error("❌ No fue posible cargar el modelo o el dataset.")
    st.exception(e)
    st.stop()

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
    (df["DEPARTAMENTO"] == departamento)
    & (df["ANIO"] == anio)
    & (df["MES"] == mes)
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

        st.success("✅ Predicción realizada correctamente")

        # ==================================================
        # Métricas
        # ==================================================

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicción",
            f"{pred:.2f}%"
        )

        c2.metric(
            "Valor real",
            f"{valor_real:.2f}%"
        )

        c3.metric(
            "Error absoluto",
            f"{error:.2f} pp"
        )

        st.divider()

        # ==================================================
        # Interpretación
        # ==================================================

        st.subheader("📌 Interpretación")

        if error <= 1:

            st.success(
                "La predicción presenta una excelente precisión, con una diferencia menor a un punto porcentual respecto al valor observado."
            )

        elif error <= 3:

            st.warning(
                "La predicción presenta una buena aproximación al valor real, aunque existe una diferencia moderada."
            )

        else:

            st.error(
                "La diferencia entre el valor predicho y el observado es considerable."
            )

        st.info(
            f"""
El modelo Random Forest estima una tasa de ocupabilidad hotelera de
**{pred:.2f}%** para el departamento de **{departamento}**
durante el mes **{mes}** del año **{anio}**.

El valor histórico registrado para ese periodo fue de
**{valor_real:.2f}%**, obteniéndose un error absoluto de
**{error:.2f} puntos porcentuales**.
"""
        )

        st.divider()

        # ==================================================
        # Registro utilizado
        # ==================================================

        st.subheader("📄 Registro utilizado")

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

        columnas_existentes = [c for c in columnas if c in fila.columns]

        st.dataframe(
            fila[columnas_existentes],
            use_container_width=True,
            hide_index=True
        )