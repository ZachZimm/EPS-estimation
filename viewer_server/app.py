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


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _read_csv(path)


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


def _resolve_baseline_metrics(metrics: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    test_baselines = metrics.get("test_baselines") or {}
    selected_name = metrics.get("selected_val_baseline")
    preferred_names: list[str] = []
    if selected_name:
        preferred_names.append(str(selected_name))
    preferred_names.extend(["baseline_persistence", "baseline_trailing_mean", "baseline_seasonal_naive"])
    for name in preferred_names:
        baseline_metrics = test_baselines.get(name)
        if isinstance(baseline_metrics, dict):
            return _safe_float(baseline_metrics.get("mae")), _safe_float(baseline_metrics.get("rmse")), name
    return None, None, None


def _load_sector_rows(run_dir: Path) -> list[dict[str, Any]]:
    sector_metrics_path = run_dir / "sector_metrics.csv"
    if sector_metrics_path.exists():
        rows = _read_csv(sector_metrics_path).to_dict(orient="records")
        rows.sort(key=lambda row: (_safe_float(row.get("test_mae")) is None, _safe_float(row.get("test_mae")) or float("inf")))
        return rows
    sectors_root = run_dir / "sectors"
    rows: list[dict[str, Any]] = []
    if sectors_root.exists():
        for sector_dir in sorted(path for path in sectors_root.iterdir() if path.is_dir() and (path / "metrics.json").exists()):
            metrics = _json_load(sector_dir / "metrics.json")
            rows.append(
                {
                    "sector_bucket": metrics.get("sector_bucket", sector_dir.name),
                    "test_mae": _safe_float(metrics.get("test_mae")),
                    "test_rmse": _safe_float(metrics.get("test_rmse")),
                    "num_train": _safe_int(metrics.get("num_train")),
                    "num_val": _safe_int(metrics.get("num_val")),
                    "num_test": _safe_int(metrics.get("num_test")),
                    "run_id": sector_dir.relative_to(ARTIFACTS_DIR).as_posix(),
                }
            )
    rows.sort(key=lambda row: (_safe_float(row.get("test_mae")) is None, _safe_float(row.get("test_mae")) or float("inf")))
    return rows


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = _json_load(run_dir / "metrics.json")
    history = _read_csv_optional(run_dir / "training_history.csv")
    predictions = _read_csv_optional(run_dir / "test_predictions.csv")

    if not predictions.empty and {"actual", "prediction"}.issubset(predictions.columns):
        predictions = predictions.copy()
        predictions["abs_error"] = (pd.to_numeric(predictions["prediction"], errors="coerce") - pd.to_numeric(predictions["actual"], errors="coerce")).abs()
    elif not predictions.empty:
        predictions = predictions.copy()
        predictions["abs_error"] = pd.NA

    ticker_count = int(predictions["ticker"].nunique()) if "ticker" in predictions.columns else _safe_int(metrics.get("num_tickers")) or 0
    prediction_count = int(len(predictions)) if not predictions.empty else _safe_int(metrics.get("test_count")) or 0
    median_abs_error = _safe_float(predictions["abs_error"].median()) if "abs_error" in predictions.columns and not predictions.empty else None
    p90_abs_error = _safe_float(predictions["abs_error"].quantile(0.9)) if "abs_error" in predictions.columns and not predictions.empty else None

    dataset_dir = run_dir.with_name(run_dir.name.replace("_training", "_dataset"))
    dataset_config_path = dataset_dir / "config.json"
    dataset_config = _json_load(dataset_config_path) if dataset_config_path.exists() else None

    baseline_test_mae, baseline_test_rmse, baseline_name = _resolve_baseline_metrics(metrics)
    improvement_vs_baseline = None
    test_mae = _safe_float(metrics.get("test_mae"))
    if test_mae is not None and baseline_test_mae is not None:
        improvement_vs_baseline = baseline_test_mae - test_mae

    sector_rows = _load_sector_rows(run_dir)

    return {
        "id": run_dir.name,
        "name": run_dir.name,
        "path": str(run_dir),
        "mode": metrics.get("mode", "single"),
        "model_type": metrics.get("model_type"),
        "optimizer": metrics.get("optimizer"),
        "best_epoch": _safe_int(metrics.get("best_epoch")),
        "epochs_completed": _safe_int(metrics.get("epochs_completed")),
        "stopped_early": bool(metrics.get("stopped_early", False)),
        "val_mae": _safe_float(metrics.get("val_mae")),
        "test_mae": test_mae,
        "test_rmse": _safe_float(metrics.get("test_rmse")),
        "baseline_name": baseline_name,
        "baseline_test_mae": baseline_test_mae,
        "baseline_test_rmse": baseline_test_rmse,
        "mae_improvement_vs_baseline": improvement_vs_baseline,
        "prediction_count": prediction_count,
        "ticker_count": ticker_count,
        "median_abs_error": median_abs_error,
        "p90_abs_error": p90_abs_error,
        "history_points": int(len(history)),
        "dataset_config": dataset_config,
        "sector_count": len(sector_rows),
        "has_sector_breakdown": bool(sector_rows),
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
    if {"prediction", "actual"}.issubset(frame.columns):
        frame["abs_error"] = (pd.to_numeric(frame["prediction"], errors="coerce") - pd.to_numeric(frame["actual"], errors="coerce")).abs()
        frame["signed_error"] = pd.to_numeric(frame["prediction"], errors="coerce") - pd.to_numeric(frame["actual"], errors="coerce")
    else:
        frame["abs_error"] = pd.NA
        frame["signed_error"] = pd.NA
    return frame


def _load_history(run_id: str) -> pd.DataFrame:
    history_path = _get_run_dir(run_id) / "training_history.csv"
    return _read_csv_optional(history_path)


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
    _run_index.cache_clear()
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

    if not predictions.empty and "ticker" in predictions.columns:
        by_ticker = (
            predictions.groupby("ticker", dropna=False)
            .agg(
                samples=("ticker", "size"),
                mae=("abs_error", "mean"),
                rmse=("signed_error", lambda values: float((pd.to_numeric(values, errors="coerce").pow(2).mean()) ** 0.5)),
                mean_prediction=("prediction", "mean"),
                mean_actual=("actual", "mean"),
            )
            .reset_index()
            .sort_values("mae", ascending=False)
        )
    else:
        by_ticker = pd.DataFrame(columns=["ticker", "samples", "mae", "rmse", "mean_prediction", "mean_actual"])

    return {
        "summary": _run_index().get(run_id, _summarize_run(run_dir)),
        "metrics": metrics,
        "history": history.to_dict(orient="records"),
        "ticker_summary": by_ticker.to_dict(orient="records"),
        "prediction_columns": list(predictions.columns),
        "sector_summary": _load_sector_rows(run_dir),
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

    if ticker and "ticker" in predictions.columns:
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

    _run_index.cache_clear()
    rows = []
    for run_id in requested_ids:
        run = _run_index().get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        rows.append(run)
    return {"runs": rows}
