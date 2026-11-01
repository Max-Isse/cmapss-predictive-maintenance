"""Evidence for the phase 3 feature choices: reports/features.md, tables and figures (generated; do not edit).

Every number comes from the training engines in silver/gold:
- near-constant sensors: distinct values and spread within each operating condition;
- operating conditions: the k-means centroids, rows per dataset, distance of the furthest point;
- per-condition normalisation: correlation with RUL before and after, per dataset;
- RUL cap: a health index (mean normalised value of sensors that rise with wear) relative to each engine's own
  early life, by true RUL; least-squares fit of a hinge (flat, then linear) over candidate caps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cmapss_pm.config import Config
from cmapss_pm.features.build import assign_condition, feature_columns
from cmapss_pm.features.gold import GoldResult, gold_path
from cmapss_pm.features.spec import FeatureSpec
from cmapss_pm.quality.silver import silver_path

ONSET_DELTA = 0.1  # health-index rise (in within-condition standard deviations) taken as visible wear
CAP_GRID = range(60, 205, 5)


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    # object dtype keeps integer columns as integers (a mixed frame would otherwise print 20631 as 20631.000)
    return str(df.astype(object).to_markdown(index=False, floatfmt=floatfmt)) if len(df) else "_none_"


def sensor_table(spark: SparkSession, cfg: Config, spec: FeatureSpec) -> pd.DataFrame:
    train = spark.read.format("delta").load(silver_path(cfg, "cycles")).where(F.col("split") == "train")
    labelled = assign_condition(train, spec)
    sensors = cfg.sensor_columns
    per = (
        labelled.groupBy("op_condition")
        .agg(
            *[F.countDistinct(s).alias(f"{s}__n") for s in sensors],
            *[F.stddev_samp(s).alias(f"{s}__sd") for s in sensors],
        )
        .toPandas()
    )
    overall = train.agg(*[F.stddev_samp(s).alias(s) for s in sensors]).toPandas().iloc[0]
    rows = []
    for s in sensors:
        within = per[f"{s}__sd"].fillna(0.0)
        rows.append(
            {
                "sensor": s,
                "distinct_within_condition_min": int(per[f"{s}__n"].min()),
                "distinct_within_condition_max": int(per[f"{s}__n"].max()),
                "within_condition_sd_max": float(within.max()),
                "overall_sd": float(overall[s] or 0.0),
                "decision": "drop" if s in spec.dropped_sensors else "keep",
            }
        )
    return pd.DataFrame(rows)


def condition_table(spark: SparkSession, cfg: Config, spec: FeatureSpec) -> pd.DataFrame:
    gold = spark.read.format("delta").load(gold_path(cfg, "features"))
    counts = (
        gold.where(F.col("split") == "train")
        .groupBy("op_condition")
        .pivot("dataset")
        .count()
        .join(
            gold.groupBy("op_condition").agg(F.max("op_condition_dist").alias("max_distance_all_splits")),
            "op_condition",
        )
        .orderBy("op_condition")
        .toPandas()
        .fillna(0)
    )
    datasets = [c for c in counts.columns if c.startswith("FD")]
    counts[datasets] = counts[datasets].astype(int)
    cents = pd.DataFrame(spec.centroids, columns=spec.settings).assign(op_condition=range(len(spec.centroids)))
    return cents.merge(counts, on="op_condition", how="left")[["op_condition", *spec.settings, *counts.columns[1:]]]


def correlation_table(spark: SparkSession, cfg: Config, spec: FeatureSpec) -> pd.DataFrame:
    silver = spark.read.format("delta").load(silver_path(cfg, "cycles"))
    gold = spark.read.format("delta").load(gold_path(cfg, "features"))
    joined = gold.where(F.col("split") == "train").join(
        silver.select("dataset", "split", "unit", "cycle", *spec.kept_sensors), ["dataset", "split", "unit", "cycle"]
    )
    # pandas, not Spark's corr: a sensor constant within a dataset (sensor_10 in FD001) divides by zero under ANSI
    pdf = joined.select("dataset", "rul", *spec.kept_sensors, *[f"z_{s}" for s in spec.kept_sensors]).toPandas()
    rows = []
    for s in spec.kept_sensors:
        row: dict[str, Any] = {"sensor": s}
        for d, g in pdf.groupby("dataset"):
            raw, z = g[s], g[f"z_{s}"]
            row[f"{d} raw"] = raw.corr(g["rul"]) if raw.std() > 0 else np.nan
            row[f"{d} normalised"] = z.corr(g["rul"]) if z.std() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def rul_cap_profile(
    spark: SparkSession, cfg: Config, spec: FeatureSpec
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fc = cfg.section("features")
    hi_sensors = [s for s in fc["health_index_sensors"] if s in spec.kept_sensors]
    gold = spark.read.format("delta").load(gold_path(cfg, "features")).where(F.col("split") == "train")
    pdf = gold.select("dataset", "unit", "cycle", "rul", *[f"z_{s}" for s in hi_sensors]).toPandas()
    pdf["hi"] = pdf[[f"z_{s}" for s in hi_sensors]].mean(axis=1)
    base = pdf[pdf.cycle <= int(fc["baseline_cycles"])].groupby(["dataset", "unit"])["hi"].mean().rename("base")
    pdf = pdf.join(base, on=["dataset", "unit"])
    pdf["delta_hi"] = pdf["hi"] - pdf["base"]
    pdf["rul_bin"] = (pdf.rul // 10) * 10
    curve = pdf[pdf.rul < 250].groupby(["rul_bin", "dataset"])["delta_hi"].mean().unstack().reset_index()

    def best_cap(d: pd.DataFrame) -> tuple[int, dict[int, float]]:
        r, y = d.rul.to_numpy(float), d.delta_hi.to_numpy()
        sse = {}
        for k in CAP_GRID:
            x = np.c_[np.ones_like(r), np.maximum(0.0, k - r)]
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            sse[k] = float(((y - x @ beta) ** 2).sum())
        k_best = min(sse, key=lambda k: sse[k])
        return k_best, {k: v / sse[k_best] for k, v in sse.items()}

    fits = {}
    rel = {}
    for name, d in [*pdf.groupby("dataset"), ("all", pdf)]:
        k_best, rel_sse = best_cap(d)
        fits[str(name)] = k_best
        rel[str(name)] = rel_sse
    hinge = pd.DataFrame(rel).rename_axis("cap").reset_index()
    pooled = pdf[pdf.rul < 250].groupby("rul_bin")["delta_hi"].mean()
    above = pooled[pooled >= ONSET_DELTA]
    onset = int(above.index.max() + 10) if len(above) else None  # upper edge of the last bin above the threshold
    summary = {
        "health_index_sensors": hi_sensors,
        "hinge_best_cap": fits,
        "onset_rul": onset,
        "configured_cap": spec.rul_cap,
        "cap_sse_vs_best_pooled": rel["all"].get(spec.rul_cap),
        "life_cycles_quantiles": pdf.groupby(["dataset", "unit"]).cycle.max().quantile([0.0, 0.05, 0.5]).to_dict(),
    }
    return curve, hinge, summary


def _figures(
    spark: SparkSession, cfg: Config, spec: FeatureSpec, curve: pd.DataFrame, summary: dict[str, Any]
) -> list[Path]:
    out = cfg.path("figures_dir")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    sensor = "sensor_11" if "sensor_11" in spec.kept_sensors else spec.kept_sensors[0]
    silver = spark.read.format("delta").load(silver_path(cfg, "cycles"))
    gold = spark.read.format("delta").load(gold_path(cfg, "features"))
    engine = gold.where("split = 'train' AND dataset = 'FD002'").select("unit").orderBy("unit").first()
    if engine is not None:
        e = (
            gold.where(f"split = 'train' AND dataset = 'FD002' AND unit = {engine['unit']}")
            .join(silver.select("dataset", "split", "unit", "cycle", sensor), ["dataset", "split", "unit", "cycle"])
            .select("cycle", "op_condition", sensor, f"z_{sensor}", f"z_{sensor}_mean_{max(spec.windows)}")
            .orderBy("cycle")
            .toPandas()
        )
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        sc = ax[0].scatter(e.cycle, e[sensor], c=e.op_condition, cmap="tab10", s=8)
        ax[0].set_ylabel(f"{sensor} (raw)")
        ax[0].set_title(f"FD002 training engine {engine['unit']}: raw value is dominated by the operating condition")
        ax[0].legend(*sc.legend_elements(), title="condition", fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax[1].plot(e.cycle, e[f"z_{sensor}"], lw=0.7, color="0.6", label="normalised per condition")
        ax[1].plot(
            e.cycle, e[f"z_{sensor}_mean_{max(spec.windows)}"], lw=2, label=f"rolling mean, {max(spec.windows)} cycles"
        )
        ax[1].set_ylabel(f"z_{sensor}")
        ax[1].set_xlabel("cycle")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        p = out / "features_condition_normalisation.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for d in [c for c in curve.columns if c != "rul_bin"]:
        ax.plot(curve.rul_bin + 5, curve[d], marker="o", ms=3, label=d)
    ax.axvline(spec.rul_cap, color="k", ls="--", lw=1, label=f"configured cap = {spec.rul_cap}")
    if summary["onset_rul"] is not None:
        ax.axvline(
            summary["onset_rul"],
            color="0.5",
            ls=":",
            lw=1,
            label=f"rise >= {ONSET_DELTA} from RUL {summary['onset_rul']}",
        )
    ax.axvline(
        summary["hinge_best_cap"]["all"],
        color="0.5",
        ls="-.",
        lw=1,
        label=f"hinge best fit {summary['hinge_best_cap']['all']}",
    )
    ax.invert_xaxis()
    ax.set_xlabel("true RUL (cycles, 10-cycle bins)")
    ax.set_ylabel("health index minus engine's early-life baseline")
    ax.set_title("Degradation is flat at high RUL and rises over the last ~100-150 cycles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "features_rul_cap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def write_feature_report(spark: SparkSession, cfg: Config, res: GoldResult) -> Path:
    spec = res.spec
    tables = cfg.path("tables_dir")
    sensors = sensor_table(spark, cfg, spec)
    conditions = condition_table(spark, cfg, spec)
    corr = correlation_table(spark, cfg, spec)
    curve, hinge, cap = rul_cap_profile(spark, cfg, spec)
    figs = _figures(spark, cfg, spec, curve, cap)
    for name, df in [
        ("features_sensors", sensors),
        ("features_conditions", conditions),
        ("features_correlation_with_rul", corr),
        ("features_rul_cap_curve", curve),
        ("features_rul_cap_hinge_sse", hinge),
    ]:
        df.to_csv(tables / f"{name}.csv", index=False)
    labels = pd.DataFrame(res.labels)
    pit = res.point_in_time
    pit_ok = (
        pit["rows_compared"] == pit["rows_expected"]
        and pit["max_abs_diff"] <= 1e-9
        and pit["null_pattern_mismatches"] == 0
    )
    cols = feature_columns(spec)
    n_s, n_w = len(spec.kept_sensors), len(spec.windows)
    hinge_show = hinge[hinge.cap.isin([80, 90, 100, 110, 120, 125, 130, 140, 150, 175, 200])]
    fd = [c for c in corr.columns if c.startswith(("FD002", "FD004"))]
    corr_show = corr[["sensor", *fd]]
    reports = cfg.path("reports_dir")
    fig_list = ", ".join(f"`{p.relative_to(reports).as_posix()}`" for p in figs)
    hb = cap["hinge_best_cap"]
    hinge_best = f"{hb['all']} pooled (" + ", ".join(f"{d} {v}" for d, v in hb.items() if d != "all") + ")"
    lives = ", ".join(f"{q:.0%}: {v:.0f}" for q, v in cap["life_cycles_quantiles"].items())
    text = f"""<!-- GENERATED by `cmapss gold` - do not edit by hand. -->
# Gold features and labels: evidence

Run {res.run_id}; feature spec version `{spec.version}` (fitted on {spec.fitted_on["rows"]:,} training cycles of
{sum(spec.fitted_on["engines"].values())} training engines; test engines never enter the fit). Code:
`src/cmapss_pm/features/`. Decisions: D-013 to D-017 in `docs/decisions.md`.

| item | value |
|---|---|
| silver cycles in | {res.counts["silver_cycles"]:,} |
| gold rows out | {res.counts["gold_features"]:,} |
| feature columns | {len(cols)} = cycle + {n_s} normalised sensors + {n_s} x {n_w} windows {spec.windows} x 3 |
| rolling statistics | mean, std (DataFrame API), slope (Spark SQL `regr_slope`) |
| kept sensors | {n_s}: {", ".join(spec.kept_sensors)} |
| dropped sensors | {", ".join(spec.dropped_sensors) or "none"} |
| operating conditions | {len(spec.centroids)} |
| RUL cap | {spec.rul_cap} cycles |
| failure horizon N | {spec.horizon} cycles (= maintenance lead time, config `business`) |

## 1. Near-constant sensors

A sensor is dropped when it has at most {cfg.section("features")["constant_max_distinct"]} distinct values within
every operating condition: then it carries the condition, not the engine's health. Counting within conditions
matters: across the whole fleet every sensor varies, because the six conditions shift all of them.

{_md(sensors)}

## 2. Operating conditions

k-means (k = {len(spec.centroids)}) on the three settings, standardised; centroids in original units. Training rows
per condition and dataset; the furthest point of any split is {conditions["max_distance_all_splits"].max():.3f}
standardised units from its centroid, so the six conditions are discrete points with small jitter. FD001 and FD003
run only at sea level; FD002 and FD004 switch condition from cycle to cycle.

{_md(conditions)}

## 3. Why per-condition normalisation (FD002, FD004)

Correlation with true RUL on training cycles, raw value versus value normalised by its condition's training mean
and standard deviation. Raw, the condition swamps the wear signal in FD002/FD004; normalised, the same sensors
correlate about as strongly as in the single-condition datasets. Sensors 07, 12, 15, 20 and 21 weaken or change
sign in FD004 (and FD003): those datasets have two fault modes (HPC and fan degradation), which move these sensors
in different directions, so the models must not assume one direction of wear per sensor. All four datasets:
`reports/tables/features_correlation_with_rul.csv`.

{_md(corr_show, ".2f")}

![raw vs normalised](figures/features_condition_normalisation.png)

## 4. RUL cap (piecewise-linear target)

Health index = mean normalised value of {", ".join(cap["health_index_sensors"])} (they rise with wear in all four
datasets), minus each engine's own mean over its first {cfg.section("features")["baseline_cycles"]} cycles. Mean
by true-RUL bin:

{_md(curve, ".2f")}

- The rise first exceeds {ONSET_DELTA} (within-condition standard deviations) at RUL {cap["onset_rul"]} (pooled).
  Above that the index is flat within noise: the true RUL of a healthy engine cannot be read from the sensors.
- Least-squares hinge fit (flat until the cap, then linear): best cap {hinge_best}. The hinge is biased
  low because wear accelerates (convex curve), so a straight line fits the steep end best.
- Training engine lives (cycles) by quantile: {lives}.

**Choice: cap = {spec.rul_cap}.** It lies between the hinge optimum and the visible onset, costs
{(cap["cap_sse_vs_best_pooled"] or float("nan")) - 1:.1%} more squared error than the best hinge on the pooled
data, sits below the 5% quantile of training engine life (so almost every engine has a flat-then-falling target),
and is a value widely used in C-MAPSS studies (125-130), which keeps RMSE comparable with them. Relative SSE
by cap (1.0 = best):

{_md(hinge_show, ".3f")}

![RUL cap evidence](figures/features_rul_cap.png)

## 5. Labels

`rul` = cycles left (training: last cycle minus cycle; test: the NASA RUL at the last cycle plus cycles to the
end). `rul_capped` = min(rul, {spec.rul_cap}). `fail_within_n` = 1 if rul <= {spec.horizon}. `is_last_cycle` marks
the test-set evaluation point.

{_md(labels)}

## 6. Point-in-time correctness on the full data

Each engine's history cut at a random cycle t; features recomputed from the cut history only, compared with
gold at every cycle <= t. **{"PASSED" if pit_ok else "FAILED"}**: {pit["engines"]:,} engines,
{pit["rows_compared"]:,} of {pit["rows_expected"]:,} rows matched, {pit["features_compared"]} features, maximum
absolute difference {pit["max_abs_diff"]:.3g}, {pit["null_pattern_mismatches"]} null-pattern mismatches.

Figures: {fig_list}.
"""
    path = cfg.path("reports_dir") / "features.md"
    path.write_text(text, encoding="utf-8")
    return path
