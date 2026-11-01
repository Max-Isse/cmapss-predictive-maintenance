"""FastAPI service: /health, /model-info, /predict, /predict/batch.

A prediction request names an engine (the API looks up its latest gold feature row, or a given cycle) or posts a
feature row directly. The response carries RUL with its 90% interval, P(fail within N), and the maintenance
decision under the served threshold. Every request is logged as one JSON line with its request ID (taken from
the X-Request-ID header or generated), and every prediction is appended to the prediction log.

Run: `uvicorn --factory cmapss_pm.serving.app:create_app --host 0.0.0.0 --port 8000` (docker/Dockerfile.api).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator

from cmapss_pm.config import Config, load_config
from cmapss_pm.serving.store import (
    GoldFeatures,
    LoadedModel,
    PredictionLog,
    gold_features_path,
    load_champion_model,
    prediction_log_path,
)

REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
log = logging.getLogger("cmapss.api")


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message, request_id and any `extra` fields."""

    RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        out.update({k: v for k, v in vars(record).items() if k not in self.RESERVED})
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging() -> None:
    if any(isinstance(h.formatter, JsonFormatter) for h in log.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


# ------------------------------------------------------------------ request / response schemas


class EngineRef(BaseModel):
    dataset: str = Field(pattern=r"^FD00[1-4]$")
    split: str = Field(default="replay", pattern=r"^(train|test|replay)$")
    unit: int = Field(gt=0)
    cycle: int | None = Field(default=None, gt=0, description="omit for the engine's latest cycle")


class PredictRequest(BaseModel):
    engine: EngineRef | None = None
    features: dict[str, float | None] | None = Field(default=None, description="a full gold feature row")

    @model_validator(mode="after")
    def exactly_one(self) -> PredictRequest:
        if (self.engine is None) == (self.features is None):
            raise ValueError("give exactly one of `engine` or `features`")
        return self


class BatchRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1)


class Decision(BaseModel):
    book_shop_visit: bool
    threshold: float
    threshold_source: str
    lead_time_cycles: int | None


class Prediction(BaseModel):
    engine: EngineRef | None
    rul_pred: float
    rul_lower: float
    rul_upper: float
    p_fail_within_n: float
    horizon_cycles: int
    decision: Decision


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    prediction: Prediction


class BatchItem(BaseModel):
    prediction: Prediction | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    items: list[BatchItem]


# ------------------------------------------------------------------ app


class NotFound(Exception):
    pass


def create_app(cfg: Config | None = None, model: LoadedModel | None = None) -> FastAPI:
    """`model` can be injected (tests); otherwise the champion is loaded from MLflow at startup."""
    cfg = cfg or load_config()
    configure_logging()
    sc = cfg.section("serving")
    log_predictions = os.environ.get("CMAPSS_LOG_PREDICTIONS", str(sc["log_predictions"])).lower() == "true"
    state: dict[str, Any] = {"model": model}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["model"] is None:
            try:
                loaded_model = load_champion_model(cfg)
                state["model"] = loaded_model
                log.info("model loaded", extra={"model_version": loaded_model.version})
            except Exception:  # keep serving /health (503) so the orchestrator sees why
                log.exception("model could not be loaded")
        if state["model"] is not None:
            state["gold"] = GoldFeatures(gold_features_path(cfg), state["model"].bundle.features)
        state["log"] = PredictionLog(prediction_log_path(cfg))
        yield

    app = FastAPI(title="C-MAPSS maintenance API", version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = REQUEST_ID.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            return response
        finally:
            REQUEST_ID.reset(token)

    def loaded() -> LoadedModel:
        if state.get("model") is None:
            raise HTTPException(503, "model not loaded")
        return state["model"]  # type: ignore[no-any-return]

    def feature_row(req: PredictRequest, m: LoadedModel) -> tuple[pd.DataFrame, EngineRef | None, str]:
        if req.engine is not None:
            e = req.engine
            row = state["gold"].lookup(e.dataset, e.split, e.unit, e.cycle)
            if row.empty:
                raise NotFound(
                    f"no gold features for {e.dataset}/{e.split}/unit {e.unit}"
                    + (f"/cycle {e.cycle}" if e.cycle else "")
                )
            ref = EngineRef(dataset=e.dataset, split=e.split, unit=e.unit, cycle=int(row["cycle"].iloc[0]))
            return row, ref, "api-lookup"
        assert req.features is not None
        missing = [f for f in m.bundle.features if f not in req.features]
        if missing:
            raise HTTPException(422, f"missing features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        values = {f: (np.nan if req.features[f] is None else req.features[f]) for f in m.bundle.features}
        return pd.DataFrame([values]), None, "api-features"

    def score(rows: list[tuple[pd.DataFrame, EngineRef | None, str]], m: LoadedModel) -> list[Prediction]:
        started = time.perf_counter()
        frame = pd.concat([r[0] for r in rows], ignore_index=True)
        out = m.bundle.predict_frame(frame)
        latency = (time.perf_counter() - started) * 1000 / len(rows)
        preds, logged = [], []
        for i, (row, ref, source) in enumerate(rows):
            p = float(out["p_fail_within_n"].iloc[i])
            pred = Prediction(
                engine=ref,
                rul_pred=float(out["rul_pred"].iloc[i]),
                rul_lower=float(out["rul_lower"].iloc[i]),
                rul_upper=float(out["rul_upper"].iloc[i]),
                p_fail_within_n=p,
                horizon_cycles=m.bundle.horizon,
                decision=Decision(
                    book_shop_visit=p >= m.threshold,
                    threshold=m.threshold,
                    threshold_source=m.threshold_source,
                    lead_time_cycles=m.lead_time,
                ),
            )
            preds.append(pred)
            logged.append(
                {
                    "request_id": REQUEST_ID.get(),
                    "logged_at": datetime.now(UTC),
                    "source": source,
                    "model_name": m.name,
                    "model_version": m.version,
                    "threshold": m.threshold,
                    "dataset": ref.dataset if ref else None,
                    "split": ref.split if ref else None,
                    "unit": ref.unit if ref else None,
                    "cycle": ref.cycle if ref else None,
                    "rul_pred": pred.rul_pred,
                    "rul_lower": pred.rul_lower,
                    "rul_upper": pred.rul_upper,
                    "p_fail_within_n": p,
                    "book_shop_visit": pred.decision.book_shop_visit,
                    "latency_ms": latency,
                    "features_json": None if ref else row.iloc[0].to_json(),
                }
            )
        if log_predictions:
            state["log"].append(logged)
        return preds

    @app.get("/health")
    def health(response: Response) -> dict[str, Any]:
        m = state.get("model")
        if m is None:
            response.status_code = 503
            return {"status": "unavailable", "model_loaded": False}
        return {"status": "ok", "model_loaded": True, "model_version": m.version}

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        m = loaded()
        b = m.bundle
        gold_version = None
        try:
            gold_version = state["gold"].version()
        except Exception:  # gold not mounted: scoring posted features still works
            log.warning("gold features not readable")
        return {
            "model_name": m.name,
            "alias": m.alias,
            "version": m.version,
            "run_id": m.run_id,
            "loaded_at": m.loaded_at,
            "trained_on": {k: b.meta.get(k) for k in ("gold_delta_version", "feature_spec_version", "train_engines")},
            "gold_features_version_now": gold_version,
            "n_features": len(b.features),
            "rul_cap": b.rul_cap,
            "horizon_cycles": b.horizon,
            "interval_nominal_coverage": 1 - b.alpha,
            "threshold": m.threshold,
            "threshold_source": m.threshold_source,
            "lead_time_cycles": m.lead_time,
            "metrics": m.metrics,
            "data_note": "NASA C-MAPSS simulation; costs behind the threshold are assumptions",
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        m = loaded()
        try:
            row = feature_row(req, m)
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        return PredictResponse(
            request_id=REQUEST_ID.get(), model_name=m.name, model_version=m.version, prediction=score([row], m)[0]
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest) -> BatchResponse:
        m = loaded()
        if len(req.items) > int(sc["max_batch"]):
            raise HTTPException(413, f"at most {sc['max_batch']} items per batch")
        items: list[BatchItem] = [BatchItem() for _ in req.items]
        rows, positions = [], []
        for i, item in enumerate(req.items):
            try:
                rows.append(feature_row(item, m))
                positions.append(i)
            except NotFound as exc:
                items[i] = BatchItem(error=str(exc))
        if rows:
            for i, pred in zip(positions, score(rows, m), strict=True):
                items[i] = BatchItem(prediction=pred)
        return BatchResponse(request_id=REQUEST_ID.get(), model_name=m.name, model_version=m.version, items=items)

    return app
