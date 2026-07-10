import streamlit as st
import pandas as pd
import plotly.express as px

# ======================================
# Configuración
# ======================================

st.set_page_config(
    page_title="Modelos Predictivos",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Modelos Predictivos")

st.markdown("""
En esta sección se presentan los algoritmos de Machine Learning evaluados para la
predicción de la tasa de ocupabilidad hotelera. Asimismo, se muestra el modelo
seleccionado y el análisis de importancia de variables.
""")

# ======================================
# Cargar datos
# ======================================

metricas = pd.read_csv("data/metricas_modelos.csv")
importancias = pd.read_csv("data/importancia_variables.csv")

# ======================================
# Mejor modelo
# ======================================

mejor = metricas.sort_values("MAE").iloc[0]

# ======================================
# Tabla de modelos
# ======================================

st.subheader("📋 Comparación de Modelos")

st.dataframe(
    metricas,
    use_container_width=True,
    hide_index=True
)

# ======================================
# Modelo seleccionado
# ======================================

st.markdown("## 🏆 Modelo Seleccionado")

st.success(
    f"""
**{mejor['Modelo']}**

Fue seleccionado como modelo final debido a que obtuvo el menor **MAE**
y el mejor equilibrio entre precisión y capacidad de generalización.
"""
)

# ======================================
# KPIs
# ======================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "MAE",
    f"{mejor['MAE']:.3f}"
)

c2.metric(
    "RMSE",
    f"{mejor['RMSE']:.3f}"
)

c3.metric(
    "MAPE",
    f"{mejor['MAPE']:.2f}%"
)

c4.metric(
    "R²",
    f"{mejor['R2']:.3f}"
)

st.divider()

# ======================================
# Ranking
# ======================================

st.subheader("🥇 Ranking de Modelos")

ranking = (
    metricas
    .sort_values("MAE")
    .reset_index(drop=True)
)

ranking.index = ranking.index + 1

st.dataframe(
    ranking,
    use_container_width=True
)

st.divider()

# ======================================
# Comparación de métricas
# ======================================

st.subheader("📊 Comparación de Métricas")

col1, col2 = st.columns(2)

# --------------------------
# MAE
# --------------------------

fig_mae = px.bar(
    metricas.sort_values("MAE"),
    x="Modelo",
    y="MAE",
    color="MAE",
    text_auto=".3f",
    color_continuous_scale="Blues"
)

fig_mae.update_layout(
    xaxis_title="Modelo",
    yaxis_title="MAE"
)

col1.plotly_chart(fig_mae, use_container_width=True)

# --------------------------
# RMSE
# --------------------------

fig_rmse = px.bar(
    metricas.sort_values("RMSE"),
    x="Modelo",
    y="RMSE",
    color="RMSE",
    text_auto=".3f",
    color_continuous_scale="Blues"
)

fig_rmse.update_layout(
    xaxis_title="Modelo",
    yaxis_title="RMSE"
)

col2.plotly_chart(fig_rmse, use_container_width=True)

# --------------------------
# R²
# --------------------------

fig_r2 = px.bar(
    metricas.sort_values("R2", ascending=False),
    x="Modelo",
    y="R2",
    color="R2",
    text_auto=".3f",
    color_continuous_scale="Blues"
)

fig_r2.update_layout(
    xaxis_title="Modelo",
    yaxis_title="R²"
)

st.plotly_chart(fig_r2, use_container_width=True)

st.divider()

# ======================================
# Variables importantes
# ======================================

st.subheader("⭐ Variables más importantes")

top_variables = importancias.head(10)

fig_imp = px.bar(
    top_variables,
    x="Importancia",
    y="Variable",
    orientation="h",
    color="Importancia",
    color_continuous_scale="Blues"
)

fig_imp.update_layout(
    yaxis=dict(autorange="reversed"),
    xaxis_title="Importancia",
    yaxis_title="Variable"
)

st.plotly_chart(fig_imp, use_container_width=True)

# ======================================
# Interpretación
# ======================================

st.info("""
### Interpretación

Las variables con mayor influencia corresponden a la tasa de ocupabilidad
hotelera actual, la media móvil de tres meses y los rezagos históricos.
Esto evidencia que el comportamiento reciente de la serie temporal constituye
el principal predictor de la ocupabilidad hotelera futura.
""")

st.divider()

# ======================================
# Pipeline
# ======================================

st.subheader("⚙️ Flujo del Modelo")

st.markdown("""
```text
Dataset MINCETUR
        │
        ▼
Preprocesamiento
(Limpieza + Variables temporales + Lags)
        │
        ▼
Random Forest Regressor
        │
        ▼
Predicción de la ocupabilidad hotelera""")

st.divider()


# ======================================
# Conclusion
# ======================================
st.subheader("📌 Conclusión")

st.success(
f"""
Después de comparar todos los algoritmos evaluados, el modelo
{mejor['Modelo']} fue seleccionado como solución final debido a que
presentó el menor MAE y un desempeño consistente en las métricas de
evaluación.

Asimismo, el análisis de importancia de variables evidenció que la
ocupabilidad histórica y sus rezagos representan la principal fuente de
información para anticipar la ocupabilidad hotelera futura.
"""
)