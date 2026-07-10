import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Prueba", page_icon="🧪")

st.title("🧪 Prueba de carga")

st.success("✅ Imports correctos")

# ======================================
# Cargar dataset
# ======================================

try:
    df = pd.read_csv("data/dataset_final.csv")
    st.success("✅ Dataset cargado correctamente")
    st.write(df.head())

except Exception as e:
    st.error("❌ Error cargando dataset")
    st.exception(e)
    st.stop()

# ======================================
# Cargar modelo
# ======================================

try:
    modelo = joblib.load("modelo_ocupabilidad.pkl")
    st.success("✅ Modelo cargado correctamente")

    st.write("Tipo de modelo:")
    st.write(type(modelo))

except Exception as e:
    st.error("❌ Error cargando modelo")
    st.exception(e)
    st.stop()

st.success("🎉 Todo funciona correctamente")