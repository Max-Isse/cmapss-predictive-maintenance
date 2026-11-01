"""The silver data contract, expressed as a pandera (PySpark backend) schema built from configs/config.yaml.

Row-level rules (quality/silver.py) decide what goes to quarantine; this contract is the independent gate on the
finished silver table: types, nullability, key domains and the same sensor ranges. A silver table that violates
it fails the run (D-009).
"""

from __future__ import annotations

from typing import Any

import pandera.pyspark  # noqa: F401  (importing it registers the DataFrame.pandera accessor)
import pyspark.sql.types as T
from pandera.api.checks import Check
from pandera.api.pyspark.components import Column
from pandera.api.pyspark.container import DataFrameSchema
from pyspark.sql import DataFrame

from cmapss_pm.config import Config


def silver_cycles_schema(cfg: Config) -> DataFrameSchema:
    vc = cfg.section("validation")
    columns: dict[str, Column] = {
        "dataset": Column(T.StringType(), Check.isin(list(cfg.section("source")["datasets"])), nullable=False),
        "split": Column(T.StringType(), Check.isin(["train", "test", "replay"]), nullable=False),
        "unit": Column(T.IntegerType(), Check.greater_than(0), nullable=False),
        "cycle": Column(
            T.IntegerType(), [Check.greater_than(0), Check.less_than_or_equal_to(int(vc["max_cycle"]))], nullable=False
        ),
    }
    for c, (lo, hi) in vc["ranges"].items():
        columns[c] = Column(T.DoubleType(), Check.in_range(float(lo), float(hi)), nullable=False)
    return DataFrameSchema(columns, name="silver.cycles", strict=False)


def silver_rul_schema(cfg: Config) -> DataFrameSchema:
    return DataFrameSchema(
        {
            "dataset": Column(T.StringType(), Check.isin(list(cfg.section("source")["datasets"])), nullable=False),
            "unit": Column(T.IntegerType(), Check.greater_than(0), nullable=False),
            "rul": Column(T.IntegerType(), Check.greater_than_or_equal_to(0), nullable=False),
        },
        name="silver.rul",
        strict=False,
    )


def validate(schema: DataFrameSchema, df: DataFrame) -> list[dict[str, Any]]:
    """Run the contract; return one row per failed check (empty list = contract satisfied)."""
    out = schema.validate(df)
    errors: dict[str, Any] = dict(out.pandera.errors)  # type: ignore[attr-defined,call-overload]
    rows: list[dict[str, Any]] = []
    for level, by_type in errors.items():
        for error_type, items in by_type.items():
            for item in items:
                rows.append(
                    {
                        "schema": schema.name,
                        "level": level,
                        "type": error_type,
                        "column": item.get("column"),
                        "check": item.get("check"),
                        "error": item.get("error"),
                    }
                )
    return rows
