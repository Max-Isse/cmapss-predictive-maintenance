"""SparkSession with Delta Lake, configured from `spark` in configs/config.yaml.

Runs in the project's Linux container (docker/Dockerfile.dev) or on Linux CI; native Windows needs Hadoop's
winutils, which this project does not use (decisions.md D-002). On Databricks the session already exists and
`get_spark` simply returns it.
"""

from __future__ import annotations

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from cmapss_pm.config import Config


def get_spark(cfg: Config, app_name: str = "cmapss-pm") -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    sc = cfg.section("spark")
    builder = (
        SparkSession.builder.master(str(sc["master"]))
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", str(sc["shuffle_partitions"]))
        .config("spark.driver.memory", str(sc["driver_memory"]))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
