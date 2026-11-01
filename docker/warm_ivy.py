"""Start and stop a Delta-enabled Spark session once so the Delta jars are cached in the image."""

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

builder = (
    SparkSession.builder.master("local[1]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.ui.enabled", "false")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.range(1).count()
spark.stop()
print("delta jars cached")
