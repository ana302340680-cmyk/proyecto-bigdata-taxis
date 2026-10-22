# proyecto-bigdata-taxis
# Proyecto Big Data: análisis de viajes de taxi en Nueva York con Apache Spark

## Problema
Analizar la demanda de taxis amarillos de Nueva York para identificar:
1. En qué horas y días se concentra la demanda.
2. Cómo se relacionan distancia, duración y tarifa.
3. Qué factores influyen en el porcentaje de propina.

## Dataset
NYC TLC Yellow Taxi Trip Records (enero 2023, formato parquet):
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Se trabajó con una muestra del 25 % de los viajes.

## Procesamiento en batch (PySpark)
- **Carga:** descarga del archivo parquet original.
- **Limpieza:** eliminación de nulos y de valores fuera de rango (distancias, tarifas y duraciones imposibles).
- **Transformación:** creación de hora, día de la semana, velocidad, porcentaje de propina y tipo de pago.
- **EDA:** estadísticas descriptivas, agregaciones por hora, día, zona y tipo de pago, y correlaciones, con DataFrames y un ejemplo con RDDs.
- **Visualización:** 7 gráficas con matplotlib.

## Cómo ejecutarlo
Requisitos: Java 17 y uv.

    uv run --python 3.11 --with pyspark --with "pandas<3" --with matplotlib --with pyarrow taxi_batch_spark.py

## Autora
Tu nombre aquí
