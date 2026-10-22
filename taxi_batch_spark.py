import os
import urllib.request                                                                  
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MESES = ["2023-01"]          # para la versión final: ["2023-01", "2023-02", "2023-03"]
FECHA_INICIO = "2023-01-01"
FECHA_FIN = "2023-02-01"     # para 3 meses: "2023-04-01"
DATA_DIR = "data"
OUT_DIR = "resultados"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
COLUMNAS = {
    "tpep_pickup_datetime": TimestampType(),
    "tpep_dropoff_datetime": TimestampType(),
    "passenger_count": IntegerType(),
    "trip_distance": DoubleType(),
    "PULocationID": IntegerType(),
    "DOLocationID": IntegerType(),
    "payment_type": IntegerType(),
    "fare_amount": DoubleType(),
    "tip_amount": DoubleType(),
    "total_amount": DoubleType(),
}
def descargar(mes):
    nombre = f"yellow_tripdata_{mes}.parquet"
    destino = os.path.join(DATA_DIR, nombre)
    if not os.path.exists(destino):
        print(f"Descargando {nombre} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{nombre}", destino)
    return destino
def cargar(spark):
    dfs = []
    for mes in MESES:
        ruta = os.path.abspath(descargar(mes))
        df = spark.read.parquet(ruta)
        df = df.select([F.col(c).cast(t).alias(c) for c, t in COLUMNAS.items()])
        dfs.append(df)
    resultado = dfs[0]
    for d in dfs[1:]:
        resultado = resultado.unionByName(d)
    return resultado
# ---------------- 2. Limpieza ----------------
def limpiar(df):
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                           "trip_distance", "fare_amount", "total_amount"])
    df = df.dropDuplicates()
    df = df.withColumn(
        "duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime")
         - F.unix_timestamp("tpep_pickup_datetime")) / 60.0)
    df = df.filter(
        (F.col("trip_distance") > 0) & (F.col("trip_distance") < 100)
        & (F.col("fare_amount") > 0) & (F.col("fare_amount") < 500)
        & (F.col("tip_amount") >= 0)
        & (F.col("duration_min") > 1) & (F.col("duration_min") < 180)
        & (F.coalesce(F.col("passenger_count"), F.lit(1)).between(1, 6)))
    df = df.filter(F.col("tpep_pickup_datetime") >= FECHA_INICIO) \
           .filter(F.col("tpep_pickup_datetime") < FECHA_FIN)
    df = df.fillna({"passenger_count": 1})
    return df


# ---------------- 3. Transformación ----------------
def transformar(df):
    return (
        df.withColumn("hora", F.hour("tpep_pickup_datetime"))
          .withColumn("dia_semana", F.dayofweek("tpep_pickup_datetime"))
          .withColumn("nombre_dia", F.date_format("tpep_pickup_datetime", "E"))
          .withColumn("mes", F.month("tpep_pickup_datetime"))
          .withColumn("velocidad_mph",
                      F.col("trip_distance") / (F.col("duration_min") / 60.0))
          .withColumn("tip_pct",
                      F.when(F.col("fare_amount") > 0,
                             F.col("tip_amount") / F.col("fare_amount") * 100))
          .withColumn("tipo_pago",
                      F.when(F.col("payment_type") == 1, "Tarjeta")
                       .when(F.col("payment_type") == 2, "Efectivo")
                       .otherwise("Otro"))
          .filter(F.col("velocidad_mph") < 80)
    )


# ---------------- 4. EDA ----------------
def eda(df, total_inicial):
    df.cache()
    total = df.count()
    print("\n" + "=" * 70)
    print(f"Registros originales   : {total_inicial:,}")
    print(f"Registros tras limpieza: {total:,} ({total / total_inicial:.1%})")
    print("=" * 70)

    print("\n-- Esquema --")
    df.printSchema()

    print("\n-- Estadísticas descriptivas --")
    df.select("trip_distance", "duration_min", "fare_amount",
              "tip_amount", "velocidad_mph", "tip_pct").summary().show()

    print("\n-- Cuantiles aproximados --")
    for c in ["trip_distance", "duration_min", "fare_amount"]:
        q = df.approxQuantile(c, [0.25, 0.5, 0.75, 0.95], 0.01)
        print(f"{c:15s} Q1={q[0]:.2f} Mediana={q[1]:.2f} "
              f"Q3={q[2]:.2f} P95={q[3]:.2f}")

    por_hora = (df.groupBy("hora")
                  .agg(F.count("*").alias("viajes"),
                       F.round(F.avg("fare_amount"), 2).alias("tarifa_prom"))
                  .orderBy("hora"))
    print("\n-- Viajes por hora --")
    por_hora.show(24)

    por_dia = (df.groupBy("dia_semana", "nombre_dia")
                 .agg(F.count("*").alias("viajes"))
                 .orderBy("dia_semana"))
    print("\n-- Viajes por día --")
    por_dia.show()

    propina = (df.groupBy("tipo_pago")
                 .agg(F.count("*").alias("viajes"),
                      F.round(F.avg("tip_pct"), 2).alias("tip_pct_prom"))
                 .orderBy(F.desc("viajes")))
    print("\n-- Propina por tipo de pago --")
    propina.show()

    top_zonas = (df.groupBy("PULocationID")
                   .agg(F.count("*").alias("viajes"))
                   .orderBy(F.desc("viajes")).limit(10))
    print("\n-- Top 10 zonas de recogida --")
    top_zonas.show()

    heat = df.groupBy("dia_semana", "hora").agg(F.count("*").alias("viajes"))

    muestra = df.select("trip_distance", "fare_amount", "duration_min") \
                .sample(False, 0.01, seed=42)
    print("\n-- RDD: viajes por número de pasajeros --")
    try:
        rdd = (df.select("passenger_count").rdd
                 .map(lambda r: (r[0], 1))
                 .reduceByKey(lambda a, b: a + b)
                 .sortByKey())
        for k, v in rdd.collect():
            print(f"  {k} pasajero(s): {v:,}")
    except Exception as e:
        print("RDD omitido en este equipo:", str(e)[:100])

    print("\n-- RDD: viajes por número de pasajeros --")
    rdd = (df.select("passenger_count").rdd
             .map(lambda r: (r[0], 1))
             .reduceByKey(lambda a, b: a + b)
             .sortByKey())
    for k, v in rdd.collect():
        print(f"  {k} pasajero(s): {v:,}")

    return {
        "por_hora": por_hora.toPandas(),
        "por_dia": por_dia.toPandas(),
        "propina": propina.toPandas(),
        "top_zonas": top_zonas.toPandas(),
        "heat": heat.toPandas(),
        "muestra": muestra.toPandas(),
        "hist_dist": df.filter(F.col("trip_distance") <= 20)
                       .select("trip_distance").sample(False, 0.05, seed=1)
                       .toPandas(),
    }


# ---------------- 5. Visualización ----------------
def graficar(r):
    ax = r["por_hora"].plot(x="hora", y="viajes", kind="bar",
                            legend=False, figsize=(10, 4), color="#2a7de1")
    ax.set_title("Demanda de viajes por hora del día")
    ax.set_xlabel("Hora"); ax.set_ylabel("Número de viajes")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_viajes_por_hora.png", dpi=150)
    plt.close()

    ax = r["por_dia"].plot(x="nombre_dia", y="viajes", kind="bar",
                           legend=False, figsize=(8, 4), color="#e1892a")
    ax.set_title("Viajes por día de la semana")
    ax.set_xlabel(""); ax.set_ylabel("Número de viajes")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/02_viajes_por_dia.png", dpi=150)
    plt.close()

    pivot = r["heat"].pivot(index="dia_semana", columns="hora", values="viajes")
    plt.figure(figsize=(11, 4))
    plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="Viajes")
    plt.yticks(range(7), ["Dom", "Lun", "Mar", "Mié", "Jue", "Vie", "Sáb"])
    plt.xticks(range(24), range(24))
    plt.title("Mapa de calor: demanda por día y hora")
    plt.xlabel("Hora"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_mapa_calor.png", dpi=150); plt.close()

    m = r["muestra"]
    plt.figure(figsize=(7, 5))
    plt.scatter(m["trip_distance"], m["fare_amount"], s=3, alpha=0.3)
    plt.title("Distancia vs. tarifa (muestra)")
    plt.xlabel("Distancia (millas)"); plt.ylabel("Tarifa (USD)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_distancia_vs_tarifa.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(r["hist_dist"]["trip_distance"], bins=40, color="#3aa655")
    plt.title("Distribución de la distancia de los viajes")
    plt.xlabel("Millas"); plt.ylabel("Frecuencia")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_hist_distancia.png", dpi=150)
    plt.close()

    ax = r["propina"].plot(x="tipo_pago", y="tip_pct_prom", kind="bar",
                           legend=False, figsize=(6, 4), color="#8e44ad")
    ax.set_title("Propina promedio (%) según tipo de pago")
    ax.set_xlabel(""); ax.set_ylabel("% de propina")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/06_propina_tipo_pago.png", dpi=150)
    plt.close()

    ax = r["top_zonas"].plot(x="PULocationID", y="viajes", kind="barh",
                             legend=False, figsize=(7, 4), color="#c0392b")
    ax.set_title("Top 10 zonas de recogida (ID)")
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/07_top_zonas.png", dpi=150)
    plt.close()

    print(f"\nGráficas guardadas en ./{OUT_DIR}/")


# ---------------- Main ----------------
def main():
    spark = (SparkSession.builder
             .appName("NYC-Taxi-Batch-EDA")
             .master("local[2]")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    crudo = cargar(spark)
    crudo = crudo.sample(False, 0.25, seed=42)
    total_inicial = crudo.count()

    limpio = transformar(limpiar(crudo))
    resultados = eda(limpio, total_inicial)
    graficar(resultados)

    try:
        limpio.write.mode("overwrite").partitionBy("mes") \
              .parquet(f"{OUT_DIR}/taxi_limpio.parquet")
        print("Dataset limpio guardado en parquet.")
    except Exception as e:
        print("No se pudo guardar el parquet (común en Windows sin winutils).")
        print("Las gráficas y el análisis ya se generaron. Detalle:", str(e)[:200])

    spark.stop()


if __name__ == "__main__":
    main() 