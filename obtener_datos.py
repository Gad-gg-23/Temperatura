"""Descarga el histórico horario de variables meteorológicas para ESCOM
usando la API gratuita de Open-Meteo (no requiere API key) y lo guarda en CSV.

Uso:
    python obtener_datos.py
"""
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

import config

sys.stdout.reconfigure(encoding="utf-8")


def descargar_anio(anio: int) -> pd.DataFrame:
    inicio = f"{anio}-01-01"
    hoy = date.today()
    fin_dt = date(anio, 12, 31)
    if fin_dt >= hoy:
        fin_dt = hoy - timedelta(days=1)  # el día de hoy puede estar incompleto
    fin = fin_dt.strftime("%Y-%m-%d")
    if fin < inicio:
        return pd.DataFrame()

    params = {
        "latitude": config.LATITUD,
        "longitude": config.LONGITUD,
        "start_date": inicio,
        "end_date": fin,
        "hourly": ",".join(config.VARIABLES_HORARIAS),
        "timezone": config.ZONA_HORARIA,
    }
    respuesta = requests.get(config.URL_HISTORICO, params=params, timeout=60)
    respuesta.raise_for_status()
    datos = respuesta.json()["hourly"]
    df = pd.DataFrame(datos)
    df["fecha_hora"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


def main():
    config.DIR_DATOS.mkdir(parents=True, exist_ok=True)
    anio_actual = date.today().year

    partes = []
    for anio in range(config.ANIO_INICIO_HISTORICO, anio_actual + 1):
        print(f"Descargando {anio}...")
        df_anio = descargar_anio(anio)
        if df_anio.empty:
            continue
        partes.append(df_anio)
        time.sleep(1)  # cortesía con la API gratuita

    df = pd.concat(partes, ignore_index=True)
    df = df.dropna(subset=["temperature_2m"])
    df.to_csv(config.ARCHIVO_HISTORICO, index=False)
    print(f"\nGuardados {len(df)} registros horarios en {config.ARCHIVO_HISTORICO}")
    print(f"Rango de fechas: {df['fecha_hora'].min()} a {df['fecha_hora'].max()}")


if __name__ == "__main__":
    main()
