import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("GENERADOR DE PREDICCIONES 2025-2030")
print("=" * 70)

# ==========================================================
# Cargar dataset
# ==========================================================

print("\nCargando dataset...")

df = pd.read_csv("data/dataset_final.csv")

print("Dataset cargado correctamente")
print(f"Registros : {len(df):,}")
print(f"Columnas  : {len(df.columns)}")

# ==========================================================
# Cargar modelo
# ==========================================================

print("\nCargando modelo...")

modelo = joblib.load("modelo_ocupabilidad.pkl")

print("Modelo cargado correctamente")

# ==========================================================
# Variables utilizadas por el modelo
# ==========================================================

variables_modelo = list(modelo.feature_names_in_)

print("\nVariables del modelo:")

for v in variables_modelo:
    print("-", v)

# ==========================================================
# Último registro disponible por departamento
# ==========================================================

print("\nBuscando último registro por departamento...")

ultimo_registro = (
    df
    .sort_values(["DEPARTAMENTO", "ANIO", "MES"])
    .groupby("DEPARTAMENTO")
    .tail(1)
    .reset_index(drop=True)
)

print(f"\nDepartamentos encontrados: {len(ultimo_registro)}")

print("\nPrimeros registros:")

print(
    ultimo_registro[
        [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "TASA_OCUPABILIDAD_HOTELERA"
        ]
    ].head()
)

# ==========================================================
# Variables que vamos a proyectar
# ==========================================================

variables_proyectar = [

    "NUMERO_ESTABLECIMIENTOS",
    "NUMERO_HABITACIONES",
    "NUMERO_PLAZAS_CAMA",
    "PROMEDIO_PERMANENCIA",
    "TOTAL_ARRIBOS",
    "TOTAL_PERNOCT",
    "TOTAL_EMPLEO"

]

print("\nVariables a proyectar:")

for v in variables_proyectar:
    print("-", v)

# ==========================================================
# PARTE 2
# Calcular tendencia histórica por departamento
# ==========================================================

print("\n" + "=" * 70)
print("CALCULANDO TENDENCIAS HISTÓRICAS")
print("=" * 70)

tendencias = {}

for departamento in df["DEPARTAMENTO"].unique():

    datos_dep = (
        df[df["DEPARTAMENTO"] == departamento]
        .sort_values(["ANIO", "MES"])
        .reset_index(drop=True)
    )

    tendencias[departamento] = {}

    for variable in variables_proyectar:

        # Promedio anual
        promedio_anual = (
            datos_dep
            .groupby("ANIO")[variable]
            .mean()
            .reset_index()
        )

        # Si existe más de un año calculamos incremento promedio
        if len(promedio_anual) > 1:

            incrementos = (
                promedio_anual[variable]
                .diff()
                .dropna()
            )

            tendencia = incrementos.mean()

        else:

            tendencia = 0

        tendencias[departamento][variable] = tendencia

print("\nTendencias calculadas correctamente.")

# ==========================================================
# Mostrar ejemplo
# ==========================================================

ejemplo = list(tendencias.keys())[0]

print(f"\nEjemplo de tendencias para {ejemplo}:\n")

for variable in variables_proyectar:

    print(
        f"{variable:30s}"
        f"{tendencias[ejemplo][variable]:10.2f}"
    )

# ==========================================================
# PARTE 3
# Generar predicciones futuras (2025-2030)
# ==========================================================

print("\n" + "=" * 70)
print("GENERANDO PREDICCIONES FUTURAS")
print("=" * 70)

predicciones = []

for _, ultimo in ultimo_registro.iterrows():

    departamento = ultimo["DEPARTAMENTO"]

    # Se trabaja sobre una copia del último registro conocido
    registro_base = ultimo.copy()

    for anio in range(2025, 2031):

        for mes in range(1, 13):

            nuevo = registro_base.copy()

            # =====================================================
            # Variables temporales
            # =====================================================

            nuevo["ANIO"] = anio
            nuevo["MES"] = mes

            nuevo["MES_SIN"] = np.sin(2 * np.pi * mes / 12)
            nuevo["MES_COS"] = np.cos(2 * np.pi * mes / 12)

            nuevo["ES_COVID_2020"] = 0

            nuevo["TENDENCIA_MESES"] = (
                (anio - 2019) * 12 + mes
            )

            # =====================================================
            # Aplicar tendencia histórica
            # =====================================================

            años_transcurridos = anio - 2024

            for variable in variables_proyectar:

                incremento = tendencias[departamento][variable]

                nuevo[variable] = (
                    ultimo[variable]
                    + incremento * años_transcurridos
                )

            # Evitar negativos

            for variable in variables_proyectar:

                if nuevo[variable] < 0:

                    nuevo[variable] = 0

            # =====================================================
            # Variables rezagadas
            # =====================================================

            nuevo["TASA_LAG_1"] = registro_base["TASA_OCUPABILIDAD_HOTELERA"]
            nuevo["TASA_LAG_2"] = registro_base["TASA_LAG_1"]
            nuevo["TASA_LAG_3"] = registro_base["TASA_LAG_2"]

            nuevo["ARRIBOS_LAG_1"] = registro_base["TOTAL_ARRIBOS"]
            nuevo["PERNOCT_LAG_1"] = registro_base["TOTAL_PERNOCT"]
            nuevo["EMPLEO_LAG_1"] = registro_base["TOTAL_EMPLEO"]

            nuevo["TASA_MEDIA_MOVIL_3"] = (
                nuevo["TASA_LAG_1"]
                + nuevo["TASA_LAG_2"]
                + nuevo["TASA_LAG_3"]
            ) / 3

            # =====================================================
            # Predicción
            # =====================================================

            X = nuevo[variables_modelo].to_frame().T

            pred = modelo.predict(X)[0]

            nuevo["TASA_OCUPABILIDAD_HOTELERA"] = pred

            predicciones.append(nuevo)

            # =====================================================
            # Actualizar historial
            # =====================================================

            registro_base = nuevo.copy()

print("\nPredicciones generadas correctamente.")

predicciones = pd.DataFrame(predicciones)

print(f"Total de registros: {len(predicciones):,}")

print("\nPrimeras predicciones:")

print(
    predicciones[
        [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "TASA_OCUPABILIDAD_HOTELERA"
        ]
    ].head(20)
)

# ==========================================================
# PARTE 4
# Guardar archivo final
# ==========================================================

print("\n" + "=" * 70)
print("GUARDANDO PREDICCIONES")
print("=" * 70)

# Ordenar registros
predicciones = predicciones.sort_values(
    ["DEPARTAMENTO", "ANIO", "MES"]
).reset_index(drop=True)

# Guardar CSV
predicciones.to_csv(
    "data/predicciones_2025_2030.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado correctamente.")

print(f"Total de registros: {len(predicciones):,}")

print("\nPrimeras filas:")

print(
    predicciones[
        [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "TASA_OCUPABILIDAD_HOTELERA"
        ]
    ].head(15)
)

print("\nÚltimas filas:")

print(
    predicciones[
        [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "TASA_OCUPABILIDAD_HOTELERA"
        ]
    ].tail(15)
)

print("\nArchivo guardado en:")

print("data/predicciones_2025_2030.csv")

print("\n" + "=" * 70)
print("PROCESO FINALIZADO")
print("=" * 70)