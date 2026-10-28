"""Predicción en vivo de la temperatura actual en ESCOM.

Consulta las condiciones meteorológicas actuales (humedad, presión, viento,
nubosidad, radiación, precipitación) en la API de Open-Meteo y calcula DOS
candidatos:
    - "modelo": nuestra regresión multivariable (entrenar_modelo.py)
    - "api":    el pronóstico crudo de Open-Meteo (temperature_2m)

Ninguno de los dos coincide exactamente con el termómetro real de ESCOM (son
estimaciones de una malla meteorológica, no mediciones en el campus), así que
a cada uno se le aplica su propia corrección de sesgo aprendida de las
pruebas de campo (calibracion.csv), y la **predicción final es el promedio
de ambos candidatos ya corregidos**.

¿Por qué promediar en vez de "elegir un ganador"? Con los primeros 11
registros de campo se probó explícitamente: elegir un solo candidato daba
MAE≈1.25°C; promediar ambos candidatos corregidos bajó el error a MAE≈0.94°C.
Si los dos candidatos se equivocan por razones distintas (más o menos
independientes entre sí), promediarlos cancela parte de ese ruido.

Uso:
    python predecir.py                   # solo predice
    python predecir.py --registrar 18.4  # predice y guarda el resultado junto
                                          # con la lectura real del termómetro,
                                          # para recalibrar con pruebas futuras
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import requests

import config
from caracteristicas import construir_caracteristicas

sys.stdout.reconfigure(encoding="utf-8")

COLUMNAS_CALIBRACION = ["temp_predicha_modelo", "temp_pronostico_api", "temp_termometro"]


def obtener_condiciones_actuales() -> dict:
    variables = [v for v in config.VARIABLES_HORARIAS if v != "temperature_2m"] + ["temperature_2m"]
    params = {
        "latitude": config.LATITUD,
        "longitude": config.LONGITUD,
        "current": ",".join(variables),
        "timezone": config.ZONA_HORARIA,
    }
    respuesta = requests.get(config.URL_ACTUAL, params=params, timeout=30)
    respuesta.raise_for_status()
    return respuesta.json()["current"]


def _cargar_calibracion() -> pd.DataFrame:
    try:
        calib = pd.read_csv(config.ARCHIVO_CALIBRACION)
    except FileNotFoundError:
        return pd.DataFrame(columns=COLUMNAS_CALIBRACION)
    return calib.dropna(subset=COLUMNAS_CALIBRACION)


def _offset(calib: pd.DataFrame, columna_pred: str) -> float:
    return float((calib["temp_termometro"] - calib[columna_pred]).mean())


def _error_blend_loo(calib: pd.DataFrame) -> float:
    """Error leave-one-out de la predicción final (promedio de los dos
    candidatos, cada uno corregido con el offset aprendido de los OTROS
    registros). Es el número que mejor simula qué tan bien funcionaría la
    mezcla en un registro nuevo, nunca visto."""
    n = len(calib)
    if n < 2:
        offset_m = _offset(calib, "temp_predicha_modelo")
        offset_a = _offset(calib, "temp_pronostico_api")
        pred = (1 - config.PESO_API) * (calib["temp_predicha_modelo"] + offset_m) + config.PESO_API * (calib["temp_pronostico_api"] + offset_a)
        return float((calib["temp_termometro"] - pred).abs().mean())

    errores = []
    for i in range(n):
        previos = calib.drop(calib.index[i])
        offset_m = _offset(previos, "temp_predicha_modelo")
        offset_a = _offset(previos, "temp_pronostico_api")
        pred_m = calib["temp_predicha_modelo"].iloc[i] + offset_m
        pred_a = calib["temp_pronostico_api"].iloc[i] + offset_a
        pred = (1 - config.PESO_API) * pred_m + config.PESO_API * pred_a
        errores.append(abs(calib["temp_termometro"].iloc[i] - pred))
    return float(np.mean(errores))


def calcular_calibracion(fecha_hora_actual=None) -> dict:
    calib = _cargar_calibracion()
    if calib.empty:
        return {"offset_modelo": 0.0, "offset_api": 0.0, "error_estimado": None, "n_muestras": 0, "mismo_dia": None}

    # Persistencia del sesgo: si ya hay lecturas anteriores del mismo día, el
    # sesgo de la más reciente representa mejor las condiciones de hoy que el
    # promedio de días con otro clima. Solo se usan lecturas ANTERIORES al
    # momento que se predice.
    if fecha_hora_actual is not None:
        fechas = pd.to_datetime(calib["fecha_hora"])
        hoy = calib[(fechas.dt.date == fecha_hora_actual.date()) & (fechas < fecha_hora_actual)]
        if not hoy.empty:
            ultima = hoy.iloc[[-1]]
            return {
                "offset_modelo": _offset(ultima, "temp_predicha_modelo"),
                "offset_api": _offset(ultima, "temp_pronostico_api"),
                "error_estimado": None,
                "n_muestras": len(calib),
                "mismo_dia": pd.to_datetime(ultima["fecha_hora"].iloc[0]),
            }

    return {
        "offset_modelo": _offset(calib, "temp_predicha_modelo"),
        "offset_api": _offset(calib, "temp_pronostico_api"),
        "error_estimado": _error_blend_loo(calib),
        "n_muestras": len(calib),
        "mismo_dia": None,
    }


def registrar_prueba(fecha_hora, pred_modelo: float, temp_api, temp_termometro: float):
    config.ARCHIVO_CALIBRACION.parent.mkdir(parents=True, exist_ok=True)
    fila = pd.DataFrame([{
        "fecha_hora": fecha_hora,
        "temp_predicha_modelo": pred_modelo,
        "temp_pronostico_api": temp_api,
        "temp_termometro": temp_termometro,
    }])
    encabezado = not config.ARCHIVO_CALIBRACION.exists()
    fila.to_csv(config.ARCHIVO_CALIBRACION, mode="a", header=encabezado, index=False)


def predecir_actual(registrar: float | None = None) -> float:
    paquete = joblib.load(config.ARCHIVO_MODELO)
    modelo, columnas = paquete["modelo"], paquete["columnas"]

    actual = obtener_condiciones_actuales()
    fila = pd.DataFrame([actual])
    fila["fecha_hora"] = pd.to_datetime(actual["time"])
    fila = construir_caracteristicas(fila)

    pred_cruda_modelo = float(modelo.predict(fila[columnas])[0])
    temp_api = actual.get("temperature_2m")

    # Se usa la hora real del reloj: la API publica cada 15 min y puede ir
    # atrasada respecto al momento en que se toma la lectura del termómetro.
    calibracion = calcular_calibracion(pd.Timestamp.now())
    pred_modelo_corregida = pred_cruda_modelo + calibracion["offset_modelo"]
    pred_api_corregida = temp_api + calibracion["offset_api"]
    pred_final = (1 - config.PESO_API) * pred_modelo_corregida + config.PESO_API * pred_api_corregida

    print(f"Hora local (CDMX): {fila['fecha_hora'].iloc[0]}")
    print(
        f"Variables actuales: humedad={actual.get('relative_humidity_2m')}%, "
        f"presión superficie={actual.get('surface_pressure')} hPa, viento={actual.get('wind_speed_10m')} km/h, "
        f"nubosidad={actual.get('cloud_cover')}%"
    )
    print(f"Candidato 1 - modelo de regresión ({paquete['nombre']}): {pred_cruda_modelo:.2f} °C")
    print(f"Candidato 2 - pronóstico crudo de Open-Meteo: {temp_api} °C")

    if calibracion["n_muestras"] == 0:
        print("\nTodavía no hay pruebas de campo registradas (calibracion.csv vacío):")
        print("se usa el promedio simple de ambos candidatos, SIN corrección de sesgo.")
    elif calibracion["mismo_dia"] is not None:
        print(f"\nCorrección por persistencia: sesgo de la lectura más reciente de hoy ({calibracion['mismo_dia']:%H:%M}):")
        print(f"  API corregida ({calibracion['offset_api']:+.2f} °C): {pred_api_corregida:.2f} °C")
    else:
        print(f"\nCalibración con {calibracion['n_muestras']} prueba(s) de campo:")
        print(f"  Modelo corregido ({calibracion['offset_modelo']:+.2f} °C): {pred_modelo_corregida:.2f} °C")
        print(f"  Error leave-one-out estimado de la mezcla: {calibracion['error_estimado']:.2f} °C")

    print(f"\nPREDICCIÓN FINAL (mezcla ponderada de ambos candidatos corregidos): {pred_final:.2f} °C")

    if registrar is not None:
        registrar_prueba(fila["fecha_hora"].iloc[0], pred_cruda_modelo, temp_api, registrar)
        print(f"\nPrueba registrada en {config.ARCHIVO_CALIBRACION} (termómetro real = {registrar} °C)")

    return pred_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registrar", type=float, default=None, help="Lectura real del termómetro en ESCOM")
    args = parser.parse_args()
    predecir_actual(args.registrar)
