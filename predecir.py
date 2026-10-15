"""Predicción en vivo de la temperatura actual en ESCOM.

Consulta las condiciones meteorológicas actuales (humedad, presión, viento,
nubosidad, radiación, precipitación) en la API de Open-Meteo y calcula DOS
candidatos:
    - "modelo": nuestra regresión multivariable (entrenar_modelo.py)
    - "api":    el pronóstico crudo de Open-Meteo (temperature_2m)

Ninguno de los dos coincide exactamente con el termómetro real de ESCOM (son
estimaciones de una malla meteorológica, no mediciones en el campus), así que
a ambos se les aplica una corrección de sesgo aprendida de las pruebas de
campo (calibracion.csv). Con pocas pruebas se usa el sesgo promedio; en
cuanto hay suficientes, se elige automáticamente el candidato que, tras
corregir su sesgo, se haya equivocado menos (validación leave-one-out).

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

sys.stdout.reconfigure(encoding="utf-8")
from caracteristicas import construir_caracteristicas

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


def _evaluar_candidato(calib: pd.DataFrame, columna_pred: str) -> tuple[float, float]:
    """Devuelve (offset, error_estimado) para una columna de predicción,
    validando con leave-one-out cuando hay suficientes muestras."""
    residuales = calib["temp_termometro"] - calib[columna_pred]
    offset_global = float(residuales.mean())

    n = len(calib)
    if n < 2:
        return offset_global, float(residuales.abs().mean())

    errores_loo = []
    for i in range(n):
        offset_i = residuales.drop(residuales.index[i]).mean()
        pred_corregida = calib[columna_pred].iloc[i] + offset_i
        errores_loo.append(abs(calib["temp_termometro"].iloc[i] - pred_corregida))
    return offset_global, float(np.mean(errores_loo))


def calcular_calibracion() -> dict:
    calib = _cargar_calibracion()
    if calib.empty:
        return {"fuente": "modelo", "offset": 0.0, "error_estimado": None, "n_muestras": 0}

    offset_modelo, error_modelo = _evaluar_candidato(calib, "temp_predicha_modelo")
    offset_api, error_api = _evaluar_candidato(calib, "temp_pronostico_api")

    if error_api < error_modelo:
        return {"fuente": "api", "offset": offset_api, "error_estimado": error_api, "n_muestras": len(calib)}
    return {"fuente": "modelo", "offset": offset_modelo, "error_estimado": error_modelo, "n_muestras": len(calib)}


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

    calibracion = calcular_calibracion()
    base = pred_cruda_modelo if calibracion["fuente"] == "modelo" else temp_api
    pred_final = base + calibracion["offset"]

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
        print("se usa el candidato del modelo SIN corrección de sesgo.")
    else:
        print(
            f"\nCalibración con {calibracion['n_muestras']} prueba(s) de campo: "
            f"se eligió el candidato '{calibracion['fuente']}' "
            f"(error leave-one-out estimado: {calibracion['error_estimado']:.2f} °C)"
        )
        print(f"Corrección de sesgo aplicada: {calibracion['offset']:+.2f} °C")

    print(f"\nPREDICCIÓN FINAL: {pred_final:.2f} °C")

    if registrar is not None:
        registrar_prueba(fila["fecha_hora"].iloc[0], pred_cruda_modelo, temp_api, registrar)
        print(f"\nPrueba registrada en {config.ARCHIVO_CALIBRACION} (termómetro real = {registrar} °C)")

    return pred_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registrar", type=float, default=None, help="Lectura real del termómetro en ESCOM")
    args = parser.parse_args()
    predecir_actual(args.registrar)
