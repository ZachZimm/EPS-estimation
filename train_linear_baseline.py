from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, QuantileRegressor

from dataset_builder import BASELINE_COLUMNS, PrototypeConfig
from train_transformer import (
    DatasetBundle,
    TargetPreprocessor,
    _compute_prediction_metrics,
    _sanitize_slug,
    apply_tail_hardening,
    evaluate_baselines,
    fit_feature_preprocessor,
    fit_target_preprocessor,
    inverse_transform_prediction,
    load_dataset,
    select_best_baseline,
    set_seed,
)


def _sequence_summary_features(sequences: np.ndarray, mode: str) -> np.ndarray:
    if mode == "flat_full":
        return sequences.reshape(sequences.shape[0], -1).astype(np.float32)
    if mode != "summary":
        raise ValueError(f"Unsupported linear_feature_mode: {mode}")
    last = sequences[:, -1, :]
    mean = sequences.mean(axis=1)
    std = sequences.std(axis=1)
    minimum = sequences.min(axis=1)
    maximum = sequences.max(axis=1)
    return np.concatenate([last, mean, std, minimum, maximum], axis=1).astype(np.float32)


def _prepare_split_arrays(
    bundle: DatasetBundle,
    split: str,
    include_ticker_fixed_effects: bool,
    feature_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = fit_feature_preprocessor(bundle)
    metadata = bundle.metadata[bundle.metadata["split"] == split].reset_index(drop=True).copy()
    source_idx = metadata["sample_id"].to_numpy(dtype=np.int64)

    raw_sequences = bundle.sequences[source_idx]
    raw_static = bundle.static[source_idx]
    clipped_sequences = np.clip(
        raw_sequences,
        preprocessor.sequence_lower[None, None, :],
        preprocessor.sequence_upper[None, None, :],
    )
    clipped_static = np.clip(
        raw_static,
        preprocessor.static_lower[None, :],
        preprocessor.static_upper[None, :],
    )
    sequences = (clipped_sequences - preprocessor.sequence_mean[None, None, :]) / preprocessor.sequence_std[None, None, :]
    static = (clipped_static - preprocessor.static_mean[None, :]) / preprocessor.static_std[None, :]
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sequence_features = _sequence_summary_features(sequences, feature_mode)
    ticker_ids = metadata["ticker_id"].to_numpy(dtype=np.int64)
    parts = [sequence_features, static]
    if include_ticker_fixed_effects:
        num_tickers = int(bundle.metadata["ticker_id"].max()) + 1
        ticker_one_hot = np.eye(num_tickers, dtype=np.float32)[ticker_ids]
        parts.append(ticker_one_hot)
    design = np.concatenate(parts, axis=1).astype(np.float32)
    baselines = pd.to_numeric(
        metadata.get("baseline_persistence", metadata.get("last_prior_eps", 0.0)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    targets = bundle.targets[source_idx].astype(np.float32)
    return metadata, design, targets, baselines, ticker_ids


def _transform_targets_numpy(
    targets: np.ndarray,
    baselines: np.ndarray,
    ticker_ids: np.ndarray,
    target_preprocessor: TargetPreprocessor,
) -> np.ndarray:
    target_t = torch.tensor(targets, dtype=torch.float32)
    baseline_t = torch.tensor(baselines, dtype=torch.float32)
    ticker_t = torch.tensor(ticker_ids, dtype=torch.long)
    from train_transformer import transform_target

    return transform_target(target_t, baseline_t, ticker_t, target_preprocessor).cpu().numpy().astype(np.float32)


def _inverse_predictions_numpy(
    predictions: np.ndarray,
    baselines: np.ndarray,
    ticker_ids: np.ndarray,
    target_preprocessor: TargetPreprocessor,
) -> np.ndarray:
    pred_t = torch.tensor(predictions, dtype=torch.float32)
    baseline_t = torch.tensor(baselines, dtype=torch.float32)
    ticker_t = torch.tensor(ticker_ids, dtype=torch.long)
    restored = inverse_transform_prediction(pred_t, baseline_t, ticker_t, target_preprocessor)
    return restored.cpu().numpy().astype(np.float32)


def _inverse_quantiles_numpy(
    predictions: np.ndarray,
    baselines: np.ndarray,
    ticker_ids: np.ndarray,
    target_preprocessor: TargetPreprocessor,
) -> np.ndarray:
    pred_t = torch.tensor(predictions, dtype=torch.float32)
    baseline_t = torch.tensor(baselines, dtype=torch.float32).unsqueeze(-1)
    ticker_t = torch.tensor(ticker_ids, dtype=torch.long)
    restored = inverse_transform_prediction(pred_t, baseline_t, ticker_t, target_preprocessor)
    return restored.cpu().numpy().astype(np.float32)


def _fit_point_model(x_train: np.ndarray, y_train: np.ndarray, config: PrototypeConfig) -> LinearRegression:
    model = LinearRegression(fit_intercept=config.linear_fit_intercept)
    model.fit(x_train, y_train)
    return model


def _fit_quantile_models(x_train: np.ndarray, y_train: np.ndarray, config: PrototypeConfig) -> dict[str, QuantileRegressor]:
    max_rows = int(getattr(config, "linear_quantile_max_train_rows", 0) or 0)
    if max_rows > 0 and len(x_train) > max_rows:
        rng = np.random.default_rng(config.seed)
        keep = rng.choice(len(x_train), size=max_rows, replace=False)
        keep.sort()
        x_fit = x_train[keep]
        y_fit = y_train[keep]
    else:
        x_fit = x_train
        y_fit = y_train
    models: dict[str, QuantileRegressor] = {}
    for quantile in config.quantiles:
        key = f"q{int(round(float(quantile) * 100)):02d}"
        model = QuantileRegressor(
            quantile=float(quantile),
            alpha=float(config.linear_quantile_alpha),
            fit_intercept=config.linear_fit_intercept,
            solver="highs",
        )
        model.fit(x_fit, y_fit)
        models[key] = model
    return models


def _evaluate_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def _evaluate_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(actual[mask] - pred[mask]))))


def train_single_linear_model(bundle: DatasetBundle, config: PrototypeConfig, output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_preprocessor = fit_target_preprocessor(
        bundle,
        config.target_normalization,
        config.target_mode,
        config.residual_clip_lower_q,
        config.residual_clip_upper_q,
    )
    train_meta, x_train, y_train_raw, baseline_train, ticker_train = _prepare_split_arrays(
        bundle, "train", config.linear_include_ticker_fixed_effects, config.linear_feature_mode
    )
    _, x_val, y_val_raw, baseline_val, ticker_val = _prepare_split_arrays(
        bundle, "val", config.linear_include_ticker_fixed_effects, config.linear_feature_mode
    )
    test_meta, x_test, y_test_raw, baseline_test, ticker_test = _prepare_split_arrays(
        bundle, "test", config.linear_include_ticker_fixed_effects, config.linear_feature_mode
    )
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise RuntimeError("Train/val/test splits must all be non-empty.")

    y_train = _transform_targets_numpy(y_train_raw, baseline_train, ticker_train, target_preprocessor)

    if config.prediction_objective == "quantile":
        models = _fit_quantile_models(x_train, y_train, config)
        ordered_keys = sorted(models.keys())
        val_quantiles_norm = np.column_stack([models[key].predict(x_val) for key in ordered_keys]).astype(np.float32)
        test_quantiles_norm = np.column_stack([models[key].predict(x_test) for key in ordered_keys]).astype(np.float32)
        val_quantiles = np.sort(_inverse_quantiles_numpy(val_quantiles_norm, baseline_val, ticker_val, target_preprocessor), axis=1)
        test_quantiles = np.sort(_inverse_quantiles_numpy(test_quantiles_norm, baseline_test, ticker_test, target_preprocessor), axis=1)
        quantile_tensor = torch.tensor(test_quantiles, dtype=torch.float32)
        baseline_tensor = torch.tensor(baseline_test, dtype=torch.float32)
        ticker_tensor = torch.tensor(ticker_test, dtype=torch.long)
        test_quantiles = apply_tail_hardening(
            quantile_tensor,
            baseline_tensor,
            ticker_tensor,
            target_preprocessor,
            config.cold_start_baseline_min_train_samples,
            config.residual_clip_mode,
        ).cpu().numpy()
        test_quantiles = np.sort(test_quantiles, axis=1)
        quantiles = [float(q) for q in config.quantiles]
        median_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
        val_pred = val_quantiles[:, median_idx]
        test_pred = test_quantiles[:, median_idx]
    else:
        model = _fit_point_model(x_train, y_train, config)
        val_pred_norm = model.predict(x_val).astype(np.float32)
        test_pred_norm = model.predict(x_test).astype(np.float32)
        val_pred = _inverse_predictions_numpy(val_pred_norm, baseline_val, ticker_val, target_preprocessor)
        test_pred = _inverse_predictions_numpy(test_pred_norm, baseline_test, ticker_test, target_preprocessor)
        test_quantiles = None

    test_pred = apply_tail_hardening(
        torch.tensor(test_pred, dtype=torch.float32),
        torch.tensor(baseline_test, dtype=torch.float32),
        torch.tensor(ticker_test, dtype=torch.long),
        target_preprocessor,
        config.cold_start_baseline_min_train_samples,
        config.residual_clip_mode,
    ).cpu().numpy()

    alpha = float(config.prediction_blend_alpha)
    valid_baseline = np.isfinite(baseline_test)
    if alpha != 1.0:
        blended = test_pred.copy()
        blended[valid_baseline] = alpha * test_pred[valid_baseline] + (1.0 - alpha) * baseline_test[valid_baseline]
        test_pred = blended
        if test_quantiles is not None:
            adjusted = test_quantiles.copy()
            adjusted[valid_baseline, :] = alpha * adjusted[valid_baseline, :] + (1.0 - alpha) * baseline_test[valid_baseline, None]
            test_quantiles = adjusted

    test_metadata = test_meta.copy()
    test_metadata["prediction"] = test_pred
    test_metadata["actual"] = y_test_raw
    if test_quantiles is not None:
        ordered_keys = [f"q{int(round(float(q) * 100)):02d}" for q in config.quantiles]
        for idx, key in enumerate(ordered_keys):
            test_metadata[f"prediction_{key}"] = test_quantiles[:, idx]

    selected_baseline = select_best_baseline(bundle.metadata, split="val")
    selected_baseline_name = selected_baseline["name"]
    if selected_baseline_name and selected_baseline_name in test_metadata.columns:
        test_metadata["baseline_selected"] = pd.to_numeric(test_metadata[selected_baseline_name], errors="coerce")
    else:
        test_metadata["baseline_selected"] = np.nan
    test_metadata.to_csv(output_dir / "test_predictions.csv", index=False)

    test_metrics = _compute_prediction_metrics(test_metadata)
    val_mae = _evaluate_mae(y_val_raw, val_pred)
    metrics = {
        "best_epoch": 1,
        "stopped_early": False,
        "epochs_completed": 1,
        "model_type": "true_linear_regression" if config.prediction_objective == "point" else "true_linear_quantile_regression",
        "optimizer": None,
        "prediction_objective": config.prediction_objective,
        "quantiles": list(config.quantiles),
        "linear_include_ticker_fixed_effects": bool(config.linear_include_ticker_fixed_effects),
        "linear_fit_intercept": bool(config.linear_fit_intercept),
        "linear_quantile_alpha": float(config.linear_quantile_alpha),
        "linear_feature_mode": config.linear_feature_mode,
        "linear_quantile_max_train_rows": int(config.linear_quantile_max_train_rows),
        "val_mae": float(val_mae),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "interval_80_coverage": test_metrics.get("interval_80_coverage"),
        "interval_80_width": test_metrics.get("interval_80_width"),
        "val_baselines": evaluate_baselines(bundle.metadata, "val"),
        "test_baselines": evaluate_baselines(bundle.metadata, "test"),
        "selected_val_baseline": selected_baseline_name,
        "selected_val_baseline_mae": selected_baseline["mae"],
        "num_train": int(len(x_train)),
        "num_val": int(len(x_val)),
        "num_test": int(len(x_test)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([
        {
            "epoch": 1,
            "lr": np.nan,
            "train_loss": np.nan,
            "train_mae": np.nan,
            "train_rmse": np.nan,
            "val_loss": np.nan,
            "val_mae": val_mae,
            "val_rmse": _evaluate_rmse(y_val_raw, val_pred),
        }
    ]).to_csv(output_dir / "training_history.csv", index=False)
    return metrics, test_metadata


def train_per_sector_linear(bundle: DatasetBundle, config: PrototypeConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sector_metrics_rows: list[dict[str, Any]] = []
    all_test_predictions: list[pd.DataFrame] = []
    sector_dirs = output_dir / "sectors"
    for bucket in sorted(bundle.metadata["sector_bucket"].dropna().unique().tolist()):
        bucket_meta = bundle.metadata[bundle.metadata["sector_bucket"] == bucket].copy()
        if bucket_meta.empty:
            continue
        split_counts = bucket_meta["split"].value_counts().to_dict()
        if any(split_counts.get(split, 0) == 0 for split in ["train", "val", "test"]):
            continue
        bucket_bundle = bundle.subset(bucket_meta)
        bucket_dir = sector_dirs / _sanitize_slug(str(bucket))
        metrics, test_predictions = train_single_linear_model(bucket_bundle, config, bucket_dir)
        sector_metrics_rows.append(
            {
                "sector_bucket": bucket,
                "num_tickers": int(bucket_meta["ticker"].nunique()),
                "num_train": int(split_counts.get("train", 0)),
                "num_val": int(split_counts.get("val", 0)),
                "num_test": int(split_counts.get("test", 0)),
                "test_mae": metrics["test_mae"],
                "test_rmse": metrics["test_rmse"],
                "selected_val_baseline": metrics["selected_val_baseline"],
                "selected_val_baseline_mae": metrics["selected_val_baseline_mae"],
            }
        )
        all_test_predictions.append(test_predictions.assign(model_sector_bucket=bucket))

    if not all_test_predictions:
        raise RuntimeError("No sector buckets produced complete train/val/test splits.")

    aggregate_test = pd.concat(all_test_predictions, ignore_index=True)
    aggregate_test = aggregate_test.sort_values(["target_published_date", "ticker"]).reset_index(drop=True)
    aggregate_metrics = _compute_prediction_metrics(aggregate_test)
    aggregate_baselines: dict[str, dict[str, float]] = {}
    for column in BASELINE_COLUMNS:
        if column not in aggregate_test.columns:
            continue
        baseline_frame = aggregate_test[[column, "actual"]].rename(columns={column: "prediction"}).copy()
        aggregate_baselines[column] = _compute_prediction_metrics(baseline_frame)
    val_baseline_summary: dict[str, Any] = {}
    for bucket in sorted(bundle.metadata["sector_bucket"].dropna().unique().tolist()):
        bucket_meta = bundle.metadata[bundle.metadata["sector_bucket"] == bucket].copy()
        if bucket_meta.empty:
            continue
        val_baseline_summary[str(bucket)] = select_best_baseline(bucket_meta, split="val")

    aggregate_test.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(sector_metrics_rows).sort_values("test_mae").to_csv(output_dir / "sector_metrics.csv", index=False)
    metrics = {
        "mode": "per_sector",
        "prediction_objective": config.prediction_objective,
        "quantiles": list(config.quantiles),
        "model_type": "true_linear_baseline",
        "test_mae": float(aggregate_metrics["mae"]),
        "test_rmse": float(aggregate_metrics["rmse"]),
        "interval_80_coverage": aggregate_metrics.get("interval_80_coverage"),
        "interval_80_width": aggregate_metrics.get("interval_80_width"),
        "test_count": int(aggregate_metrics["count"]),
        "num_sector_models": int(len(sector_metrics_rows)),
        "test_baselines": aggregate_baselines,
        "val_baseline_selection_by_sector": val_baseline_summary,
        "linear_include_ticker_fixed_effects": bool(config.linear_include_ticker_fixed_effects),
        "linear_fit_intercept": bool(config.linear_fit_intercept),
        "linear_quantile_alpha": float(config.linear_quantile_alpha),
        "linear_feature_mode": config.linear_feature_mode,
        "linear_quantile_max_train_rows": int(config.linear_quantile_max_train_rows),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a true linear EPS baseline on the prepared dataset.")
    parser.add_argument("--config", default="configs/expanded_run_macro_true_linear_config.json", help="Path to config JSON.")
    parser.add_argument("--dataset-dir", default="artifacts/expanded_run_macro_true_linear_dataset", help="Directory containing dataset artifacts.")
    parser.add_argument("--output-dir", default="artifacts/expanded_run_macro_true_linear_training", help="Directory for metrics and predictions.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PrototypeConfig.from_path(args.config)
    set_seed(config.seed)
    bundle = load_dataset(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if config.sector_modeling_mode == "per_sector" and "sector_bucket" in bundle.metadata.columns:
        metrics = train_per_sector_linear(bundle, config, output_dir)
    else:
        metrics, _ = train_single_linear_model(bundle, config, output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
