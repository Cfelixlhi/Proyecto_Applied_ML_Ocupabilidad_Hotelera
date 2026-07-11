# ==========================================================
# GENERAR PREDICCIONES 2025-2030
# Proyecto Applied Machine Learning
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("=" * 60)
print("GENERADOR DE PREDICCIONES 2025-2030")
print("=" * 60)

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

# ==========================================================
# Información del modelo
# ==========================================================

print("\nVariables utilizadas por el modelo:\n")

columnas_modelo = list(modelo.feature_names_in_)

for c in columnas_modelo:
    print("-", c)

# ==========================================================
# Obtener último registro de cada departamento
# ==========================================================

print("\nBuscando último registro disponible por departamento...")

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

print("\nPrimeros registros:")

print(
    ultimos[
        [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "TASA_OCUPABILIDAD_HOTELERA"
        ]
    ].head()
)

# ==========================================================
# Variables base
# ==========================================================

print("\nPreparando variables base...")

registros_futuros = []

# ==========================================================
# Función para crear un registro futuro
# ==========================================================

def crear_registro_futuro(registro, anio, mes):

    nuevo = registro.copy()

    # Fecha
    nuevo["ANIO"] = anio
    nuevo["MES"] = mes

    # Variables cíclicas
    nuevo["MES_SIN"] = np.sin(2 * np.pi * mes / 12)
    nuevo["MES_COS"] = np.cos(2 * np.pi * mes / 12)

    # Tendencia temporal
    nuevo["TENDENCIA_MESES"] += 1

    # Después de 2020 ya no existe COVID
    nuevo["ES_COVID_2020"] = 0

    return nuevo

# ==========================================================
# Función para actualizar los lags
# ==========================================================

def actualizar_lags(registro, prediccion):

    registro["TASA_LAG_3"] = registro["TASA_LAG_2"]
    registro["TASA_LAG_2"] = registro["TASA_LAG_1"]
    registro["TASA_LAG_1"] = prediccion

    registro["TASA_MEDIA_MOVIL_3"] = (
        registro["TASA_LAG_1"] +
        registro["TASA_LAG_2"] +
        registro["TASA_LAG_3"]
    ) / 3

    return registro

# ==========================================================
# Verificación
# ==========================================================

ejemplo = ultimos.iloc[0].copy()

nuevo = crear_registro_futuro(
    ejemplo,
    2025,
    1
)

print("\nRegistro de prueba")

print(
    nuevo[
        [
            "DEPARTAMENTO",
            "ANIO",
            "MES",
            "MES_SIN",
            "MES_COS",
            "TENDENCIA_MESES"
        ]
    ]
)

# ==========================================================
# GENERAR PREDICCIONES FUTURAS
# ==========================================================

print("\nGenerando predicciones...\n")

for _, fila in ultimos.iterrows():

    estado = fila.copy()

    for anio in range(2025, 2031):

        for mes in range(1, 13):

            futuro = crear_registro_futuro(
                estado,
                anio,
                mes
            )

            # Variables que espera el modelo
            X = futuro[columnas_modelo].to_frame().T

            # Predicción
            pred = modelo.predict(X)[0]

            # Guardar la predicción
            futuro["TASA_OCUPABILIDAD_HOTELERA"] = pred

            registros_futuros.append(futuro.copy())

            # Actualizar estado para el siguiente mes
            estado = futuro.copy()

            estado = actualizar_lags(
                estado,
                pred
            )

print("Predicciones generadas correctamente.")
print(f"Total de registros: {len(registros_futuros)}")

# ==========================================================
# CREAR DATAFRAME FINAL
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
# GUARDAR CSV
# ==========================================================

predicciones.to_csv(
    "data/predicciones_2025_2030.csv",
    index=False
)

print("\n==========================================")
print("Archivo guardado correctamente.")
print("Ruta:")
print("data/predicciones_2025_2030.csv")
print("==========================================")