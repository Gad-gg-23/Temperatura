"""Configuración compartida para la práctica de predicción de temperatura (ESCOM)."""
from pathlib import Path

# Coordenadas aproximadas de ESCOM (IPN Zacatenco, Gustavo A. Madero, CDMX).
# La resolución de la malla de Open-Meteo es de varios km, así que pequeños
# ajustes aquí no cambian mucho los datos obtenidos.
LATITUD = 19.5038
LONGITUD = -99.1479
ZONA_HORARIA = "America/Mexico_City"

# Primer año completo a descargar para el histórico de entrenamiento.
ANIO_INICIO_HISTORICO = 2016

# Variables horarias a descargar. Se excluyen a propósito dew_point_2m y
# apparent_temperature porque son calculadas a partir de la temperatura:
# usarlas como entrada sería fuga de información (el modelo "haría trampa").
VARIABLES_HORARIAS = [
    "temperature_2m",  # variable objetivo
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "precipitation",
]

# Ventana horaria de la clase (hora local CDMX), usada para armar el
# subconjunto de evaluación "realista" (parecido al momento de la entrega).
# Ajustar si el horario real de la clase es distinto.
HORA_CLASE_INICIO = 7
HORA_CLASE_FIN = 9  # exclusivo

# Meses relevantes para la entrega (finales de octubre, clima ya más frío).
MESES_ENTREGA = [9, 10, 11]

DIR_BASE = Path(__file__).resolve().parent
DIR_DATOS = DIR_BASE / "datos"
ARCHIVO_HISTORICO = DIR_DATOS / "historico_escom.csv"
ARCHIVO_MODELO = DIR_BASE / "modelo_temperatura.joblib"
ARCHIVO_RESULTADOS = DIR_BASE / "resultados_temperatura.csv"
ARCHIVO_CALIBRACION = DIR_BASE / "calibracion.csv"

URL_HISTORICO = "https://archive-api.open-meteo.com/v1/archive"
URL_ACTUAL = "https://api.open-meteo.com/v1/forecast"
