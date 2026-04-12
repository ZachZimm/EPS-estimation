from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _list_run_dirs() -> list[Path]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in ARTIFACTS_DIR.iterdir()
            if path.is_dir() and (path / "metrics.json").exists()
        ],
        key=lambda path: path.name,
    )


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = _json_load(run_dir / "metrics.json")
    history = _read_csv(run_dir / "training_history.csv")
    predictions = _read_csv(run_dir / "test_predictions.csv")

    if "actual" in predictions.columns and "prediction" in predictions.columns:
        predictions = predictions.copy()
        predictions["abs_error"] = (predictions["prediction"] - predictions["actual"]).abs()
    else:
        predictions["abs_error"] = pd.NA

    ticker_count = int(predictions["ticker"].nunique()) if "ticker" in predictions.columns else 0
    prediction_count = int(len(predictions))
    median_abs_error = (
        _safe_float(predictions["abs_error"].median()) if "abs_error" in predictions.columns else None
    )
    p90_abs_error = (
        _safe_float(predictions["abs_error"].quantile(0.9))
        if "abs_error" in predictions.columns and not predictions.empty
        else None
    )

    dataset_config_path = run_dir.with_name(run_dir.name.replace("_training", "_dataset")) / "config.json"
    dataset_config = _json_load(dataset_config_path) if dataset_config_path.exists() else None

    test_baseline = metrics.get("test_baseline", {})
    improvement_vs_baseline = None
    if _safe_float(metrics.get("test_mae")) and _safe_float(test_baseline.get("mae")):
        improvement_vs_baseline = float(test_baseline["mae"]) - float(metrics["test_mae"])

    return {
        "id": run_dir.name,
        "name": run_dir.name,
        "path": str(run_dir),
        "model_type": metrics.get("model_type"),
        "optimizer": metrics.get("optimizer"),
        "best_epoch": _safe_int(metrics.get("best_epoch")),
        "epochs_completed": _safe_int(metrics.get("epochs_completed")),
        "stopped_early": bool(metrics.get("stopped_early", False)),
        "val_mae": _safe_float(metrics.get("val_mae")),
        "test_mae": _safe_float(metrics.get("test_mae")),
        "test_rmse": _safe_float(metrics.get("test_rmse")),
        "baseline_test_mae": _safe_float(test_baseline.get("mae")),
        "baseline_test_rmse": _safe_float(test_baseline.get("rmse")),
        "mae_improvement_vs_baseline": improvement_vs_baseline,
        "prediction_count": prediction_count,
        "ticker_count": ticker_count,
        "median_abs_error": median_abs_error,
        "p90_abs_error": p90_abs_error,
        "history_points": int(len(history)),
        "dataset_config": dataset_config,
    }


@lru_cache(maxsize=1)
def _run_index() -> dict[str, dict[str, Any]]:
    return {run_dir.name: _summarize_run(run_dir) for run_dir in _list_run_dirs()}


def _get_run_dir(run_id: str) -> Path:
    run_dir = ARTIFACTS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
    return run_dir


def _load_predictions(run_id: str) -> pd.DataFrame:
    predictions_path = _get_run_dir(run_id) / "test_predictions.csv"
    if not predictions_path.exists():
        raise HTTPException(status_code=404, detail=f"Predictions for run {run_id!r} were not found.")
    frame = _read_csv(predictions_path)
    frame["abs_error"] = (frame["prediction"] - frame["actual"]).abs()
    frame["signed_error"] = frame["prediction"] - frame["actual"]
    return frame


def _load_history(run_id: str) -> pd.DataFrame:
    history_path = _get_run_dir(run_id) / "training_history.csv"
    if not history_path.exists():
        raise HTTPException(status_code=404, detail=f"Training history for run {run_id!r} was not found.")
    return _read_csv(history_path)


app = FastAPI(title="EPS Run Viewer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    runs = list(_run_index().values())
    runs.sort(
        key=lambda run: (
            run["test_mae"] is None,
            run["test_mae"] if run["test_mae"] is not None else float("inf"),
        )
    )
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    run_dir = _get_run_dir(run_id)
    metrics = _json_load(run_dir / "metrics.json")
    history = _load_history(run_id)
    predictions = _load_predictions(run_id)

    by_ticker = (
        predictions.groupby("ticker", dropna=False)
        .agg(
            samples=("ticker", "size"),
            mae=("abs_error", "mean"),
            rmse=("signed_error", lambda values: float((values.pow(2).mean()) ** 0.5)),
            mean_prediction=("prediction", "mean"),
            mean_actual=("actual", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    return {
        "summary": _run_index().get(run_id, _summarize_run(run_dir)),
        "metrics": metrics,
        "history": history.to_dict(orient="records"),
        "ticker_summary": by_ticker.to_dict(orient="records"),
        "prediction_columns": list(predictions.columns),
    }


@app.get("/api/runs/{run_id}/predictions")
def run_predictions(
    run_id: str,
    ticker: str | None = None,
    sort_by: str = "abs_error",
    descending: bool = True,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    predictions = _load_predictions(run_id)

    if ticker:
        predictions = predictions[predictions["ticker"].astype(str).str.upper() == ticker.upper()]

    if sort_by in predictions.columns:
        predictions = predictions.sort_values(sort_by, ascending=not descending)

    limited = predictions.head(limit).copy()
    for column in [
        "target_as_of_date",
        "target_published_date",
        "last_observed_market_date",
    ]:
        if column in limited.columns:
            limited[column] = limited[column].astype(str)

    return {
        "rows": limited.to_dict(orient="records"),
        "row_count": int(len(limited)),
        "total_count": int(len(predictions)),
    }


@app.get("/api/runs/{run_id}/history")
def run_history(run_id: str) -> dict[str, Any]:
    history = _load_history(run_id)
    return {"rows": history.to_dict(orient="records")}


@app.get("/api/compare")
def compare_runs(run_ids: str = Query(..., description="Comma-separated run ids")) -> dict[str, Any]:
    requested_ids = [value.strip() for value in run_ids.split(",") if value.strip()]
    if not requested_ids:
        raise HTTPException(status_code=400, detail="At least one run id is required.")

    rows = []
    for run_id in requested_ids:
        run = _run_index().get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        rows.append(run)
    return {"runs": rows}
