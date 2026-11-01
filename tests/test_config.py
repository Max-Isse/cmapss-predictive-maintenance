"""Config loading: the Databricks overlay merges over the base config."""

from __future__ import annotations

from cmapss_pm.config import load_config


def test_databricks_overlay_extends_the_base_config() -> None:
    base = load_config()
    dbx = load_config(base.root / "configs" / "databricks.yaml")
    mc = dbx.section("models")
    assert mc["mlflow"]["registered_model"] == "ml_dev.cmapss.cmapss_maintenance"  # overridden
    assert mc["mlflow"]["champion_alias"] == base.section("models")["mlflow"]["champion_alias"]  # inherited, nested
    assert mc["cv_folds"] == base.section("models")["cv_folds"] and dbx.seed == base.seed
    assert dbx.raw["paths"]["landing_dir"].startswith("/Volumes/") and "extends" not in dbx.raw
