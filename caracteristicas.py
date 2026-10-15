"""Construcción de características a partir de las variables meteorológicas
crudas. Se usa tanto al entrenar como al predecir en vivo, para garantizar
que ambos lados generen exactamente las mismas columnas en el mismo orden.
"""
import numpy as np
import pandas as pd

# pressure_msl se excluye a propósito: es surface_pressure reducida a nivel
# del mar con una fórmula que depende de la temperatura, así que el cociente
# pressure_msl/surface_pressure reconstruye casi exactamente temperature_2m
# (correlación de -0.9997 medida en el histórico). Incluirla sería fuga de
# información, igual que dew_point_2m o apparent_temperature.
COLUMNAS_MODELO = [
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "precipitation",
    "wind_dir_sin",
    "wind_dir_cos",
    "hora_sin",
    "hora_cos",
    "doy_sin",
    "doy_cos",
]


def construir_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas (codificación cíclica de hora/día/dirección
    del viento) a partir de 'fecha_hora' y 'wind_direction_10m'."""
    df = df.copy()
    hora = df["fecha_hora"].dt.hour + df["fecha_hora"].dt.minute / 60
    doy = df["fecha_hora"].dt.dayofyear

    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    direccion = np.deg2rad(df["wind_direction_10m"].fillna(0))
    df["wind_dir_sin"] = np.sin(direccion)
    df["wind_dir_cos"] = np.cos(direccion)

    return df
