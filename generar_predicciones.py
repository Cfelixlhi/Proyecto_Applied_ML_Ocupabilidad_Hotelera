# ==========================================================
# GENERADOR DE PREDICCIONES 2025-2030
# Proyecto Applied Machine Learning
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("=" * 70)
print("GENERADOR DE PREDICCIONES FUTURAS")
print("=" * 70)

# ==========================================================
# Rutas
# ==========================================================

BASE_DIR = Path(__file__).parent

DATASET_PATH = BASE_DIR / "data" / "dataset_final.csv"
MODELO_PATH = BASE_DIR / "modelo_ocupabilidad.pkl"
SALIDA_PATH = BASE_DIR / "data" / "predicciones_2025_2030.csv"

# ==========================================================
# Cargar dataset
# ==========================================================

print("\nCargando dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset cargado correctamente")
print(f"Registros: {len(df):,}")
print(f"Columnas : {len(df.columns)}")

# ==========================================================
# Cargar modelo
# ==========================================================

print("\nCargando modelo...")

modelo = joblib.load(MODELO_PATH)

print("Modelo cargado correctamente")

columnas_modelo = list(modelo.feature_names_in_)

print("\nVariables utilizadas por el modelo:\n")

for c in columnas_modelo:
    print("-", c)

# ==========================================================
# Obtener último registro por departamento
# ==========================================================

df = df.sort_values(
    ["DEPARTAMENTO", "ANIO", "MES"]
)

ultimos = (
    df
    .groupby("DEPARTAMENTO")
    .tail(1)
    .reset_index(drop=True)
)

print(f"\nDepartamentos encontrados: {len(ultimos)}")

registros_futuros = []

# ==========================================================
# Función para crear un registro futuro
# ==========================================================

def crear_registro_futuro(registro, anio, mes):

    nuevo = registro.copy()

    # ------------------------------------------------------
    # Variables temporales
    # ------------------------------------------------------

    nuevo["ANIO"] = anio
    nuevo["MES"] = mes

    nuevo["MES_SIN"] = np.sin(2 * np.pi * mes / 12)
    nuevo["MES_COS"] = np.cos(2 * np.pi * mes / 12)

    nuevo["ES_COVID_2020"] = 0
    nuevo["TENDENCIA_MESES"] += 1

    # ------------------------------------------------------
    # Años transcurridos desde el último dato (2024)
    # ------------------------------------------------------

    incremento = anio - 2024

    # ------------------------------------------------------
    # Crecimiento proyectado
    # ------------------------------------------------------

    nuevo["NUMERO_ESTABLECIMIENTOS"] *= (1.008 ** incremento)

    nuevo["NUMERO_HABITACIONES"] *= (1.010 ** incremento)

    nuevo["NUMERO_PLAZAS_CAMA"] *= (1.010 ** incremento)

    nuevo["TOTAL_ARRIBOS"] *= (1.030 ** incremento)

    nuevo["TOTAL_PERNOCT"] *= (1.028 ** incremento)

    nuevo["TOTAL_EMPLEO"] *= (1.015 ** incremento)

    nuevo["PROMEDIO_PERMANENCIA"] *= (1.002 ** incremento)

    # ------------------------------------------------------
    # También actualizar los lags de variables auxiliares
    # ------------------------------------------------------

    nuevo["ARRIBOS_LAG_1"] = nuevo["TOTAL_ARRIBOS"]

    nuevo["PERNOCT_LAG_1"] = nuevo["TOTAL_PERNOCT"]

    nuevo["EMPLEO_LAG_1"] = nuevo["TOTAL_EMPLEO"]

    return nuevo


# ==========================================================
# Función para actualizar los lags de ocupabilidad
# ==========================================================

def actualizar_lags(registro, pred):

    registro["TASA_LAG_3"] = registro["TASA_LAG_2"]
    registro["TASA_LAG_2"] = registro["TASA_LAG_1"]
    registro["TASA_LAG_1"] = pred

    registro["TASA_MEDIA_MOVIL_3"] = (
        registro["TASA_LAG_1"] +
        registro["TASA_LAG_2"] +
        registro["TASA_LAG_3"]
    ) / 3

    return registro


print("\nFunciones creadas correctamente.")

# ==========================================================
# PARTE 3
# Generar predicciones futuras
# ==========================================================

print("\nGenerando predicciones...\n")

for _, fila in ultimos.iterrows():

    estado = fila.copy()

    for anio in range(2025, 2031):

        for mes in range(1, 13):

            # Crear registro futuro
            futuro = crear_registro_futuro(
                estado,
                anio,
                mes
            )

            # Variables que espera el Pipeline
            X = futuro[columnas_modelo].to_frame().T

            # Predicción
            pred = modelo.predict(X)[0]

            # Guardar predicción
            futuro["TASA_OCUPABILIDAD_HOTELERA"] = pred

            registros_futuros.append(futuro.copy())

            # Actualizar el estado para el siguiente mes
            estado = futuro.copy()

            estado = actualizar_lags(
                estado,
                pred
            )

print("Predicciones generadas correctamente.")

print(f"Total de registros: {len(registros_futuros)}")

# ==========================================================
# Crear DataFrame
# ==========================================================

predicciones = pd.DataFrame(registros_futuros)

print("\nPrimeras predicciones:\n")

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
# Guardar predicciones
# ==========================================================

print("\n" + "=" * 70)
print("GUARDANDO PREDICCIONES")
print("=" * 70)

predicciones = predicciones.sort_values(
    ["DEPARTAMENTO", "ANIO", "MES"]
).reset_index(drop=True)

predicciones.to_csv(
    SALIDA_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado correctamente.")

print(f"Ruta: {SALIDA_PATH}")

print(f"Total de registros: {len(predicciones):,}")

print("\nAños generados:")

print(
    sorted(
        predicciones["ANIO"].unique()
    )
)

print("\nDepartamentos:")

print(
    predicciones["DEPARTAMENTO"].nunique()
)

print("\nPrimeras filas:\n")

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

print("\nÚltimas filas:\n")

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

print("\nProceso finalizado correctamente.")