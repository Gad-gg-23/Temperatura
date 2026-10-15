"""Entrena y compara varios regresores para predecir la temperatura actual de
ESCOM (nowcasting) a partir de otras variables meteorológicas simultáneas
(humedad, presión, viento, nubosidad, radiación, precipitación).

Validación temporal (no aleatoria): se entrena con los años más antiguos y se
evalúa con el año más reciente completo, para no filtrar información del
"futuro" hacia el pasado como pasaría con un train_test_split aleatorio sobre
una serie de tiempo.

Uso:
    python entrenar_modelo.py
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from caracteristicas import COLUMNAS_MODELO, construir_caracteristicas

sys.stdout.reconfigure(encoding="utf-8")


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(config.ARCHIVO_HISTORICO, parse_dates=["fecha_hora"])
    df = construir_caracteristicas(df)
    df = df.dropna(subset=COLUMNAS_MODELO + ["temperature_2m"])
    return df


def dividir_train_test(df: pd.DataFrame):
    anio_max = df["fecha_hora"].dt.year.max()
    es_test = df["fecha_hora"].dt.year == anio_max
    return df[~es_test].reset_index(drop=True), df[es_test].reset_index(drop=True)


def subconjunto_entrega(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra al subconjunto que más se parece a las condiciones de entrega:
    meses de la entrega y horario de clase."""
    hora = df["fecha_hora"].dt.hour
    mes = df["fecha_hora"].dt.month
    en_horario = hora.between(config.HORA_CLASE_INICIO, config.HORA_CLASE_FIN - 1)
    return df[mes.isin(config.MESES_ENTREGA) & en_horario]


def calcular_metricas(y_true, y_pred) -> dict:
    error_abs = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "%<=0.3C": (error_abs <= 0.3).mean() * 100,
        "%<=0.5C": (error_abs <= 0.5).mean() * 100,
        "%<=1.0C": (error_abs <= 1.0).mean() * 100,
    }


def construir_modelos() -> dict:
    return {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=42),
        ),
    }


def main():
    df = cargar_datos()
    train, test = dividir_train_test(df)
    test_entrega = subconjunto_entrega(test)

    print(f"Entrenamiento: {len(train)} registros ({train['fecha_hora'].min()} a {train['fecha_hora'].max()})")
    print(f"Prueba (año más reciente completo): {len(test)} registros")
    print(
        f"Prueba enfocada en entrega (meses {config.MESES_ENTREGA}, "
        f"hora {config.HORA_CLASE_INICIO}-{config.HORA_CLASE_FIN}h): {len(test_entrega)} registros\n"
    )

    X_train, y_train = train[COLUMNAS_MODELO], train["temperature_2m"]
    X_test, y_test = test[COLUMNAS_MODELO], test["temperature_2m"]
    X_test_entrega = test_entrega[COLUMNAS_MODELO]
    y_test_entrega = test_entrega["temperature_2m"]

    filas = []
    for nombre, modelo in construir_modelos().items():
        modelo.fit(X_train, y_train)

        pred_test = modelo.predict(X_test)
        pred_entrega = modelo.predict(X_test_entrega)

        fila_general = {"Modelo": nombre, "Conjunto": "Año completo", **calcular_metricas(y_test, pred_test)}
        fila_entrega = {
            "Modelo": nombre,
            "Conjunto": "Entrega (meses/hora)",
            **calcular_metricas(y_test_entrega, pred_entrega),
        }
        filas.extend([fila_general, fila_entrega])

        print(f"=== {nombre} ===")
        print(pd.DataFrame([fila_general, fila_entrega]).set_index("Conjunto").to_string())
        print()

    resultados = pd.DataFrame(filas)
    resultados.to_csv(config.ARCHIVO_RESULTADOS, index=False)
    print(f"Resultados guardados en {config.ARCHIVO_RESULTADOS}")

    resultados_entrega = resultados[resultados["Conjunto"] == "Entrega (meses/hora)"]
    mejor_nombre = resultados_entrega.sort_values("MAE").iloc[0]["Modelo"]
    print(f"\nMejor modelo para el escenario de entrega (menor MAE): {mejor_nombre}")

    # Reentrenar el mejor modelo con TODOS los datos disponibles (train + test)
    # antes de guardarlo, para aprovechar el histórico completo en producción.
    mejor_modelo = construir_modelos()[mejor_nombre]
    X_todo, y_todo = df[COLUMNAS_MODELO], df["temperature_2m"]
    mejor_modelo.fit(X_todo, y_todo)

    joblib.dump(
        {"modelo": mejor_modelo, "nombre": mejor_nombre, "columnas": COLUMNAS_MODELO},
        config.ARCHIVO_MODELO,
    )
    print(f"Modelo final ({mejor_nombre}) guardado en {config.ARCHIVO_MODELO}")


if __name__ == "__main__":
    main()
