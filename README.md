# Práctica: Predicción de la temperatura en ESCOM (margen ≤ 0.3°C)

Práctica de regresión aplicada: predecir la temperatura **del momento en que se entrega el trabajo** (nowcasting, no pronóstico a futuro) con un error máximo de 0.3°C contra un termómetro real en ESCOM. Es la práctica más exigente del curso porque 0.3°C es un margen extremadamente ajustado incluso para un servicio meteorológico profesional.

## Idea general

1. **Nowcasting, no forecasting**: el modelo predice la temperatura *actual* a partir de otras variables meteorológicas *simultáneas* (humedad, presión, viento, nubosidad, radiación solar), no la de un momento futuro. Es mucho más alcanzable que pronosticar con antelación.
2. **El termómetro de ESCOM es la verdad de referencia**, pero no existe un histórico de sus lecturas. El histórico de entrenamiento viene de la API gratuita Open-Meteo (interpolación/malla meteorológica), que no necesariamente coincide con el microclima exacto del campus.
3. Por eso, además del modelo de regresión, hay una **capa de calibración de campo**: en los días previos a la entrega se registran pares (predicción, lectura real del termómetro) y el script aprende/corrige el sesgo con esos datos. Sin esta calibración, llegar a 0.3°C de forma consistente no es realista.

## Archivos

| Archivo | Descripción |
|---|---|
| `config.py` | Coordenadas de ESCOM, variables a usar, ventana horaria de clase, rutas. |
| `caracteristicas.py` | Construye las columnas del modelo (codificación cíclica de hora/día/viento) a partir de los datos crudos. Se usa igual al entrenar y al predecir, para que no se desincronicen. |
| `obtener_datos.py` | Descarga histórico horario (2016–hoy) desde Open-Meteo y lo guarda en `datos/historico_escom.csv`. |
| `entrenar_modelo.py` | Entrena y compara 5 regresores, evalúa con validación temporal y guarda el mejor en `modelo_temperatura.joblib`. |
| `predecir.py` | Predicción en vivo: consulta condiciones actuales, calcula la predicción y aplica la calibración de campo. |
| `Temperatura.py` | Punto de entrada único (`datos` / `entrenar` / `predecir`). |
| `calibracion.csv` | Bitácora de pruebas de campo (predicción vs. termómetro real). Se llena con `--registrar`. |

## Fuente de datos

API gratuita [Open-Meteo](https://open-meteo.com/) (sin API key), coordenadas aproximadas de IPN Zacatenco/ESCOM (19.5038, -99.1479):

- **Histórico**: `archive-api.open-meteo.com` (reanálisis horario, 2016 en adelante).
- **Condiciones actuales**: `api.open-meteo.com/v1/forecast` (bloque `current`).

## Variables usadas y una fuga de datos detectada

Se usan como predictoras: humedad relativa, presión superficial, nubosidad, velocidad y dirección del viento (codificada cíclicamente), ráfagas, radiación solar, precipitación, hora del día y día del año (ambas cíclicas, `sin`/`cos`).

**Se excluyen a propósito** `dew_point_2m` y `apparent_temperature` porque son variables calculadas a partir de la temperatura (usarlas sería fuga de información).

Durante el desarrollo se detectó una fuga menos obvia: **`pressure_msl`** (presión reducida a nivel del mar) se calcula a partir de `surface_pressure` con una fórmula que depende de la temperatura. Como la elevación de ESCOM es fija, el cociente `pressure_msl / surface_pressure` reconstruye la temperatura casi exactamente (correlación medida: **-0.9997**). Al incluirla, un simple `LinearRegression` parecía acertar con MAE ≈ 0.05°C — en realidad solo invertía esa fórmula, no predecía nada real. Se quitó `pressure_msl` de las columnas del modelo (ver comentario en `caracteristicas.py`).

También se quitó **`precipitation`**: con las 94 mil filas del histórico, su correlación con la temperatura es casi nula (0.018 en todo el día, 0.072 en el horario de clase) y su importancia en el modelo (nativa de RandomForest y permutation importance) es prácticamente cero. En esta época del año casi no llueve en las mañanas, así que la columna es casi puro ceros. Al reentrenar sin ella, KNN y RandomForest mejoraron ligeramente y MLP (el modelo elegido) empeoró 0.03°C — una diferencia dentro del ruido normal del subconjunto de 62 muestras de "entrega", así que se mantuvo fuera por la justificación estadística del dataset grande.

`cloud_cover` y la dirección del viento (`wind_dir_sin`/`wind_dir_cos`) se revisaron también y se decidió **mantenerlas**: nubosidad tiene algo de señal en el horario de clase (corr. 0.10) y es físicamente relevante (cielos despejados favorecen más enfriamiento en la madrugada); la dirección del viento debe mantenerse como par seno/coseno completo o no tiene sentido (por separado no representa bien el ángulo).

## Modelos y validación

`LinearRegression`, `Ridge`, `KNeighborsRegressor`, `RandomForestRegressor`, `MLPRegressor`. Validación **temporal** (no aleatoria): se entrena con 2016–2025 y se evalúa con el año más reciente completo, para no filtrar información del futuro como pasaría con un `train_test_split` al azar sobre una serie de tiempo. Además del conjunto de prueba completo, se evalúa un subconjunto "parecido a la entrega" (meses de septiembre a noviembre, hora de clase) definido en `config.py` (`MESES_ENTREGA`, `HORA_CLASE_INICIO/FIN` — ajustar si el horario real es distinto).

Métricas: MAE, RMSE, R², MAPE y, lo más relevante para esta práctica, el **porcentaje de predicciones dentro de 0.3°C / 0.5°C / 1.0°C**.

## Resultado actual (sin calibración de campo todavía)

En el subconjunto "entrega" (62 registros, sept–nov, 7–9h) el mejor modelo (MLP) da **MAE ≈ 0.54°C**, lejos de 0.3°C. Es esperado: predecir temperatura solo a partir de humedad/viento/presión/radiación tiene un límite físico, y la muestra de evaluación es pequeña (un único año). **No se puede llegar de forma confiable a 0.3°C sin el paso de calibración de campo.**

## Calibración de campo (el paso que falta para llegar a 0.3°C)

En los días previos a la entrega, a la hora de la clase, correr:

```bash
python Temperatura.py predecir --registrar <lectura_del_termometro>
```

Esto guarda en `calibracion.csv` tanto la predicción del modelo como el pronóstico crudo de Open-Meteo junto con la lectura real. Con esos datos, `predecir.py`:

1. Calcula el sesgo promedio de cada candidato (modelo vs. API cruda) respecto al termómetro real.
2. Corrige ambos candidatos con su propio sesgo.
3. **La predicción final es el promedio de los dos candidatos ya corregidos** (no se elige un solo "ganador").

*Historial de la estrategia:* al principio el script elegía, vía validación leave-one-out, cuál de los dos candidatos se equivocaba menos. Con 11 registros de campo (6–20 oct) se comparó contra simplemente promediar ambos candidatos corregidos: elegir un solo candidato daba MAE≈1.25°C; promediar bajó el error a MAE≈0.94°C. Al promediar, el ruido de cada candidato (que se equivoca por razones distintas) se cancela parcialmente. Se probaron también ventanas deslizantes (usar solo los últimos 3-7 registros en vez de todo el historial) y no mejoraron el resultado.

Mientras más pruebas se acumulen (idealmente en distintos días/condiciones, a la misma hora de la clase), más confiable será la corrección. Esto es exactamente lo planeado: "hacer pruebas para ir ajustando".

## Cómo ejecutar

```bash
python -m pip install -r requirements.txt
python Temperatura.py datos                    # una vez (o para refrescar el histórico)
python Temperatura.py entrenar                  # entrena y guarda el mejor modelo
python Temperatura.py predecir                  # predicción en vivo
python Temperatura.py predecir --registrar 18.4 # predicción + registro de prueba de campo
```

Requiere internet (API de Open-Meteo).

## Notas y limitaciones

- Las coordenadas de ESCOM son aproximadas; la resolución de malla de Open-Meteo (varios km) hace que esto importe poco comparado con el propio error de modelado.
- `HORA_CLASE_INICIO/FIN` y `MESES_ENTREGA` en `config.py` deben ajustarse al horario real de la clase y a la fecha de entrega confirmada.
- Este README se irá actualizando conforme se acumulen pruebas de campo reales y se vea si el margen de 0.3°C es alcanzable o si hay que ajustar el enfoque.
