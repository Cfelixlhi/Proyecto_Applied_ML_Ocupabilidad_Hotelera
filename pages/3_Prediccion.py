import streamlit as st

st.title("Prueba")

st.write("Antes del import")

try:
    import joblib
    st.success("✅ Joblib funciona")
except Exception as e:
    st.error(type(e))
    st.error(str(e))
    st.stop()

st.write("Fin")