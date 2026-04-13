from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset_builder import BASELINE_COLUMNS, PrototypeConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DatasetBundle:
    metadata: pd.DataFrame
    sequences: np.ndarray
    static: np.ndarray
    targets: np.ndarray
    normalization: dict[str, Any]

    def subset(self, metadata_subset: pd.DataFrame) -> "DatasetBundle":
        return DatasetBundle(
            metadata=metadata_subset.reset_index(drop=True),
            sequences=self.sequences,
            static=self.static,
            targets=self.targets,
            normalization=self.normalization,
        )


@dataclass
class FeaturePreprocessor:
    sequence_lower: np.ndarray
    sequence_upper: np.ndarray
    sequence_mean: np.ndarray
    sequence_std: np.ndarray
    static_lower: np.ndarray
    static_upper: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray


@dataclass
class TargetPreprocessor:
    mean: float
    std: float
    enabled: bool
    mode: str
    ticker_means: dict[int, float]
    ticker_stds: dict[int, float]
    ticker_train_counts: dict[int, int]
    global_delta_lower: float
    global_delta_upper: float
    ticker_delta_lower: dict[int, float]
    ticker_delta_upper: dict[int, float]


def load_dataset(dataset_dir: str | Path) -> DatasetBundle:
    dataset_path = Path(dataset_dir)
    metadata = pd.read_csv(dataset_path / "event_metadata.csv")
    if "ticker_id" not in metadata.columns:
        ticker_to_id = {ticker: idx for idx, ticker in enumerate(sorted(metadata["ticker"].unique()))}
        metadata["ticker_id"] = metadata["ticker"].map(ticker_to_id).astype(int)
    if "sector_bucket" not in metadata.columns:
        metadata["sector_bucket"] = metadata.get("sector", "GLOBAL")
    arrays = np.load(dataset_path / "dataset_arrays.npz")
    normalization = json.loads((dataset_path / "normalization.json").read_text())
    return DatasetBundle(
        metadata=metadata,
        sequences=arrays["sequences"].astype(np.float32),
        static=arrays["static"].astype(np.float32),
        targets=arrays["targets"].astype(np.float32),
        normalization=normalization,
    )


class EpsSequenceDataset(Dataset):
    def __init__(self, bundle: DatasetBundle, split: str, preprocessor: FeaturePreprocessor) -> None:
        self.metadata = bundle.metadata[bundle.metadata["split"] == split].reset_index(drop=True)
        source_idx = self.metadata["sample_id"].to_numpy(dtype=np.int64)

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

        self.sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        self.static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
        self.targets = bundle.targets[source_idx]
        self.ticker_ids = self.metadata["ticker_id"].to_numpy(dtype=np.int64)
        self.baselines = pd.to_numeric(
            self.metadata.get("baseline_persistence", self.metadata.get("last_prior_eps", 0.0)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": torch.from_numpy(self.sequences[index]),
            "static": torch.from_numpy(self.static[index]),
            "ticker_id": torch.tensor(self.ticker_ids[index], dtype=torch.long),
            "baseline": torch.tensor(self.baselines[index], dtype=torch.float32),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        hidden_dim: int,
        max_len: int,
        pooling: str,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(seq_dim, d_model)
        pos_len = max_len + 1 if pooling == "cls" else max_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, pos_len, d_model))
        self.pooling = pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(sequence)
        if self.pooling == "cls":
            cls = self.cls_token.expand(sequence.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embedding[:, : x.size(1), :]
        else:
            x = x + self.pos_embedding[:, : x.size(1), :]
        encoded = self.encoder(x)
        if self.pooling == "cls":
            return encoded[:, 0, :]
        return encoded.mean(dim=1)


class GruSequenceEncoder(nn.Module):
    def __init__(self, seq_dim: int, d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return hidden[-1]


class LstmSequenceEncoder(nn.Module):
    def __init__(self, seq_dim: int, d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        return hidden[-1]


class CnnSequenceEncoder(nn.Module):
    def __init__(self, seq_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(seq_dim, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=9, padding=4),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = sequence.transpose(1, 2)
        x = self.net(x)
        return x.mean(dim=-1)


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        num_tickers: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        hidden_dim: int,
        max_len: int,
        model_type: str,
        pooling: str,
        ticker_embedding_dim: int,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if model_type == "transformer":
            self.sequence_encoder = TransformerSequenceEncoder(seq_dim, d_model, num_heads, num_layers, dropout, hidden_dim, max_len, pooling)
            sequence_out_dim = d_model
        elif model_type == "gru":
            self.sequence_encoder = GruSequenceEncoder(seq_dim, d_model, num_layers, dropout)
            sequence_out_dim = d_model
        elif model_type == "lstm":
            self.sequence_encoder = LstmSequenceEncoder(seq_dim, d_model, num_layers, dropout)
            sequence_out_dim = d_model
        elif model_type == "cnn":
            self.sequence_encoder = CnnSequenceEncoder(seq_dim, d_model, dropout)
            sequence_out_dim = d_model
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ticker_embedding = nn.Embedding(num_tickers, ticker_embedding_dim)
        head_input_dim = sequence_out_dim + d_model + ticker_embedding_dim
        self.output_dim = output_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor, ticker_id: torch.Tensor) -> torch.Tensor:
        seq_emb = self.sequence_encoder(sequence)
        static_emb = self.static_proj(static)
        ticker_emb = self.ticker_embedding(ticker_id)
        out = self.head(torch.cat([seq_emb, static_emb, ticker_emb], dim=-1))
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out


def _train_source_idx(bundle: DatasetBundle) -> np.ndarray:
    return bundle.metadata.loc[bundle.metadata["split"] == "train", "sample_id"].to_numpy(dtype=np.int64)


def fit_feature_preprocessor(bundle: DatasetBundle) -> FeaturePreprocessor:
    train_idx = _train_source_idx(bundle)
    if len(train_idx) == 0:
        raise RuntimeError("No training samples available for feature preprocessing.")
    seq_train = bundle.sequences[train_idx]
    static_train = bundle.static[train_idx]
    sequence_lower = np.nanquantile(seq_train, 0.01, axis=(0, 1)).astype(np.float32)
    sequence_upper = np.nanquantile(seq_train, 0.99, axis=(0, 1)).astype(np.float32)
    clipped_seq_train = np.clip(seq_train, sequence_lower[None, None, :], sequence_upper[None, None, :])
    sequence_mean = np.nanmean(clipped_seq_train, axis=(0, 1)).astype(np.float32)
    sequence_std = np.nanstd(clipped_seq_train, axis=(0, 1)).astype(np.float32)
    sequence_mean = np.nan_to_num(sequence_mean, nan=0.0, posinf=0.0, neginf=0.0)
    sequence_std = np.nan_to_num(sequence_std, nan=1.0, posinf=1.0, neginf=1.0)
    sequence_std = np.where(sequence_std < 1e-6, 1.0, sequence_std).astype(np.float32)
    valid_static = np.isfinite(static_train).any(axis=0)
    static_train_safe = static_train.copy()
    static_train_safe[:, ~valid_static] = 0.0
    static_lower = np.nanquantile(static_train_safe, 0.01, axis=0).astype(np.float32)
    static_upper = np.nanquantile(static_train_safe, 0.99, axis=0).astype(np.float32)
    static_lower[~valid_static] = 0.0
    static_upper[~valid_static] = 0.0
    clipped_static_train = np.clip(static_train_safe, static_lower[None, :], static_upper[None, :])
    static_mean = np.nanmean(clipped_static_train, axis=0).astype(np.float32)
    static_std = np.nanstd(clipped_static_train, axis=0).astype(np.float32)
    static_mean = np.nan_to_num(static_mean, nan=0.0, posinf=0.0, neginf=0.0)
    static_std = np.nan_to_num(static_std, nan=1.0, posinf=1.0, neginf=1.0)
    static_mean[~valid_static] = 0.0
    static_std[~valid_static] = 1.0
    static_std = np.where(static_std < 1e-6, 1.0, static_std).astype(np.float32)
    return FeaturePreprocessor(
        sequence_lower=sequence_lower,
        sequence_upper=sequence_upper,
        sequence_mean=sequence_mean,
        sequence_std=sequence_std,
        static_lower=static_lower,
        static_upper=static_upper,
        static_mean=static_mean,
        static_std=static_std,
    )


def fit_target_preprocessor(
    bundle: DatasetBundle,
    enabled: bool,
    mode: str,
    residual_clip_lower_q: float,
    residual_clip_upper_q: float,
) -> TargetPreprocessor:
    train_idx = _train_source_idx(bundle)
    target_train = bundle.targets[train_idx]
    train_meta = bundle.metadata[bundle.metadata["split"] == "train"].copy()
    mean = float(np.mean(target_train))
    std = float(np.std(target_train))
    if std < 1e-6:
        std = 1.0
    ticker_means: dict[int, float] = {}
    ticker_stds: dict[int, float] = {}
    ticker_train_counts: dict[int, int] = {}
    ticker_delta_lower: dict[int, float] = {}
    ticker_delta_upper: dict[int, float] = {}
    ticker_targets = pd.DataFrame({"ticker_id": train_meta["ticker_id"].to_numpy(dtype=int), "target": target_train})
    baseline_train = pd.to_numeric(train_meta.get("baseline_persistence", train_meta.get("last_prior_eps")), errors="coerce").to_numpy(dtype=float)
    delta_train = target_train - baseline_train
    finite_delta = delta_train[np.isfinite(delta_train)]
    global_delta_lower = float(np.quantile(finite_delta, residual_clip_lower_q)) if len(finite_delta) else -np.inf
    global_delta_upper = float(np.quantile(finite_delta, residual_clip_upper_q)) if len(finite_delta) else np.inf
    for ticker_id, group in ticker_targets.groupby("ticker_id"):
        ticker_mean = float(group["target"].mean())
        ticker_std = float(group["target"].std())
        if not np.isfinite(ticker_std) or ticker_std < 1e-6:
            ticker_std = 1.0
        ticker_means[int(ticker_id)] = ticker_mean
        ticker_stds[int(ticker_id)] = ticker_std
        ticker_train_counts[int(ticker_id)] = int(len(group))
    delta_frame = pd.DataFrame({"ticker_id": train_meta["ticker_id"].to_numpy(dtype=int), "delta": delta_train}).dropna(subset=["delta"])
    for ticker_id, group in delta_frame.groupby("ticker_id"):
        if len(group) == 0:
            continue
        ticker_delta_lower[int(ticker_id)] = float(group["delta"].quantile(residual_clip_lower_q))
        ticker_delta_upper[int(ticker_id)] = float(group["delta"].quantile(residual_clip_upper_q))
    return TargetPreprocessor(
        mean=mean,
        std=std,
        enabled=enabled,
        mode=mode,
        ticker_means=ticker_means,
        ticker_stds=ticker_stds,
        ticker_train_counts=ticker_train_counts,
        global_delta_lower=global_delta_lower,
        global_delta_upper=global_delta_upper,
        ticker_delta_lower=ticker_delta_lower,
        ticker_delta_upper=ticker_delta_upper,
    )


def transform_target(target: torch.Tensor, baseline: torch.Tensor, ticker_id: torch.Tensor, target_preprocessor: TargetPreprocessor) -> torch.Tensor:
    mode = target_preprocessor.mode
    if mode == "raw":
        transformed = target
    elif mode == "delta_last":
        transformed = target - baseline
    elif mode == "signed_log":
        transformed = torch.sign(target) * torch.log1p(torch.abs(target))
    elif mode == "ticker_zscore":
        means = torch.tensor([target_preprocessor.ticker_means.get(int(x), target_preprocessor.mean) for x in ticker_id.cpu().tolist()], dtype=target.dtype, device=target.device)
        stds = torch.tensor([target_preprocessor.ticker_stds.get(int(x), target_preprocessor.std) for x in ticker_id.cpu().tolist()], dtype=target.dtype, device=target.device)
        transformed = (target - means) / stds
    else:
        raise ValueError(f"Unsupported target_mode: {mode}")
    if target_preprocessor.enabled:
        transformed = (transformed - target_preprocessor.mean) / target_preprocessor.std
    return transformed


def inverse_transform_prediction(prediction: torch.Tensor, baseline: torch.Tensor, ticker_id: torch.Tensor, target_preprocessor: TargetPreprocessor) -> torch.Tensor:
    restored = prediction
    if target_preprocessor.enabled:
        restored = restored * target_preprocessor.std + target_preprocessor.mean
    mode = target_preprocessor.mode
    if mode == "raw":
        return restored
    if mode == "delta_last":
        return baseline + restored
    if mode == "signed_log":
        return torch.sign(restored) * torch.expm1(torch.abs(restored))
    if mode == "ticker_zscore":
        means = torch.tensor([target_preprocessor.ticker_means.get(int(x), target_preprocessor.mean) for x in ticker_id.cpu().tolist()], dtype=prediction.dtype, device=prediction.device)
        stds = torch.tensor([target_preprocessor.ticker_stds.get(int(x), target_preprocessor.std) for x in ticker_id.cpu().tolist()], dtype=prediction.dtype, device=prediction.device)
        return means + restored * stds
    raise ValueError(f"Unsupported target_mode: {mode}")


def _quantile_tensor(config: PrototypeConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    quantiles = getattr(config, "quantiles", None) or [0.1, 0.5, 0.9]
    return torch.tensor([float(q) for q in quantiles], device=device, dtype=dtype)


def _point_prediction_from_output(output: torch.Tensor, config: PrototypeConfig) -> torch.Tensor:
    if getattr(config, "prediction_objective", "point") != "quantile":
        return output
    quantiles = [float(q) for q in (getattr(config, "quantiles", None) or [0.1, 0.5, 0.9])]
    if not quantiles:
        raise ValueError("Quantile objective requires at least one quantile.")
    median_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
    return output[:, median_idx]


def _sorted_quantile_output(output: torch.Tensor, config: PrototypeConfig) -> torch.Tensor:
    if getattr(config, "prediction_objective", "point") != "quantile":
        return output
    return torch.sort(output, dim=-1).values


def _quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    target_2d = target.unsqueeze(-1)
    errors = target_2d - prediction
    losses = torch.maximum((quantiles - 1.0) * errors, quantiles * errors)
    return losses.mean()


def apply_tail_hardening(
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    ticker_id: torch.Tensor,
    target_preprocessor: TargetPreprocessor,
    min_train_samples_for_model: int,
    residual_clip_mode: str,
) -> torch.Tensor:
    hardened = prediction.clone()
    baseline_expanded = baseline.unsqueeze(-1) if hardened.ndim > 1 else baseline
    if residual_clip_mode != "none":
        residual = hardened - baseline_expanded
        valid_baseline = torch.isfinite(baseline)
        valid_baseline_expanded = valid_baseline.unsqueeze(-1) if hardened.ndim > 1 else valid_baseline
        if residual_clip_mode == "global":
            clipped = torch.clamp(residual, min=target_preprocessor.global_delta_lower, max=target_preprocessor.global_delta_upper)
            residual = torch.where(valid_baseline_expanded, clipped, residual)
        elif residual_clip_mode == "per_ticker":
            lower = torch.tensor([target_preprocessor.ticker_delta_lower.get(int(x), target_preprocessor.global_delta_lower) for x in ticker_id.cpu().tolist()], dtype=prediction.dtype, device=prediction.device)
            upper = torch.tensor([target_preprocessor.ticker_delta_upper.get(int(x), target_preprocessor.global_delta_upper) for x in ticker_id.cpu().tolist()], dtype=prediction.dtype, device=prediction.device)
            if hardened.ndim > 1:
                lower = lower.unsqueeze(-1)
                upper = upper.unsqueeze(-1)
            clipped = torch.minimum(torch.maximum(residual, lower), upper)
            residual = torch.where(valid_baseline_expanded, clipped, residual)
        else:
            raise ValueError(f"Unsupported residual_clip_mode: {residual_clip_mode}")
        candidate = baseline_expanded + residual
        hardened = torch.where(valid_baseline_expanded, candidate, hardened)
    if min_train_samples_for_model > 0:
        train_counts = torch.tensor([target_preprocessor.ticker_train_counts.get(int(x), 0) for x in ticker_id.cpu().tolist()], dtype=prediction.dtype, device=prediction.device)
        use_baseline = train_counts < float(min_train_samples_for_model)
        valid_baseline = torch.isfinite(baseline)
        if hardened.ndim > 1:
            use_baseline = use_baseline.unsqueeze(-1)
            valid_baseline = valid_baseline.unsqueeze(-1)
            baseline_for_where = baseline.unsqueeze(-1)
        else:
            baseline_for_where = baseline
        hardened = torch.where((use_baseline.bool() & valid_baseline), baseline_for_where, hardened)
    return hardened


def evaluate_baseline_column(metadata: pd.DataFrame, split: str, column_name: str) -> dict[str, float]:
    if column_name not in metadata.columns:
        return {"mae": math.nan, "rmse": math.nan, "count": 0}
    frame = metadata[metadata["split"] == split].copy()
    frame["prediction"] = pd.to_numeric(frame[column_name], errors="coerce")
    frame["target"] = pd.to_numeric(frame["target_basic_eps"], errors="coerce")
    frame = frame.dropna(subset=["prediction", "target"])
    if frame.empty:
        return {"mae": math.nan, "rmse": math.nan, "count": 0}
    diff = frame["prediction"] - frame["target"]
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "count": int(len(frame)),
    }


def evaluate_baselines(metadata: pd.DataFrame, split: str) -> dict[str, dict[str, float]]:
    return {column: evaluate_baseline_column(metadata, split, column) for column in BASELINE_COLUMNS if column in metadata.columns}


def select_best_baseline(metadata: pd.DataFrame, split: str = "val") -> dict[str, Any]:
    baseline_metrics = evaluate_baselines(metadata, split)
    best_name = None
    best_mae = math.inf
    for baseline_name, metrics in baseline_metrics.items():
        mae = metrics.get("mae", math.nan)
        if np.isfinite(mae) and mae < best_mae:
            best_name = baseline_name
            best_mae = mae
    return {"name": best_name, "mae": None if best_name is None else float(best_mae), "metrics": baseline_metrics}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_preprocessor: TargetPreprocessor,
    config: PrototypeConfig,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.HuberLoss()
    losses: list[float] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    quantile_tensor: torch.Tensor | None = None
    for batch in loader:
        sequence = batch["sequence"].to(device)
        static = batch["static"].to(device)
        ticker_id = batch["ticker_id"].to(device)
        baseline = batch["baseline"].to(device)
        target = batch["target"].to(device)
        normalized_target = transform_target(target, baseline, ticker_id, target_preprocessor)
        with torch.set_grad_enabled(training):
            raw_output = model(sequence, static, ticker_id)
            if config.prediction_objective == "quantile":
                normalized_prediction = _sorted_quantile_output(raw_output, config)
                if quantile_tensor is None:
                    quantile_tensor = _quantile_tensor(config, normalized_prediction.device, normalized_prediction.dtype)
                loss = _quantile_loss(normalized_prediction, normalized_target, quantile_tensor)
                point_prediction_norm = _point_prediction_from_output(normalized_prediction, config)
            else:
                normalized_prediction = raw_output
                loss = criterion(normalized_prediction, normalized_target)
                point_prediction_norm = normalized_prediction
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        prediction = inverse_transform_prediction(point_prediction_norm.detach(), baseline, ticker_id, target_preprocessor)
        diff = (prediction - target).cpu().numpy()
        abs_errors.extend(np.abs(diff).tolist())
        sq_errors.extend(np.square(diff).tolist())
    return (
        float(np.mean(losses)) if losses else math.nan,
        float(np.mean(abs_errors)) if abs_errors else math.nan,
        float(np.sqrt(np.mean(sq_errors))) if sq_errors else math.nan,
    )


def predict_split(
    model: nn.Module,
    dataset: EpsSequenceDataset,
    device: torch.device,
    target_preprocessor: TargetPreprocessor,
    config: PrototypeConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    predictions: list[np.ndarray] = []
    quantile_predictions: dict[str, list[np.ndarray]] = {}
    quantiles = [float(q) for q in (getattr(config, "quantiles", None) or [0.1, 0.5, 0.9])]
    with torch.no_grad():
        for batch in loader:
            baseline = batch["baseline"].to(device)
            ticker_id = batch["ticker_id"].to(device)
            raw_output = model(batch["sequence"].to(device), batch["static"].to(device), ticker_id)
            if config.prediction_objective == "quantile":
                normalized_prediction = _sorted_quantile_output(raw_output, config)
                restored_quantiles = inverse_transform_prediction(normalized_prediction, baseline.unsqueeze(-1), ticker_id, target_preprocessor)
                point_prediction = _point_prediction_from_output(restored_quantiles, config)
                for idx, quantile in enumerate(quantiles):
                    key = f"q{int(round(quantile * 100)):02d}"
                    quantile_predictions.setdefault(key, []).append(restored_quantiles[:, idx].cpu().numpy())
            else:
                point_prediction = inverse_transform_prediction(raw_output, baseline, ticker_id, target_preprocessor)
            predictions.append(point_prediction.cpu().numpy())
    merged_quantiles = {
        key: np.concatenate(values) if values else np.array([], dtype=np.float32)
        for key, values in quantile_predictions.items()
    }
    point_array = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    return point_array, merged_quantiles


def _sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "bucket"


def _compute_prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    diff = pd.to_numeric(frame["prediction"], errors="coerce") - pd.to_numeric(frame["actual"], errors="coerce")
    diff = diff.dropna()
    metrics = {
        "mae": math.nan,
        "rmse": math.nan,
        "count": 0,
    }
    if not diff.empty:
        metrics.update(
            {
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "count": int(len(diff)),
            }
        )
    q10_col = "prediction_q10"
    q90_col = "prediction_q90"
    if q10_col in frame.columns and q90_col in frame.columns:
        interval = frame[[q10_col, q90_col, "actual"]].copy()
        interval[q10_col] = pd.to_numeric(interval[q10_col], errors="coerce")
        interval[q90_col] = pd.to_numeric(interval[q90_col], errors="coerce")
        interval["actual"] = pd.to_numeric(interval["actual"], errors="coerce")
        interval = interval.dropna()
        if not interval.empty:
            lower = np.minimum(interval[q10_col].to_numpy(dtype=float), interval[q90_col].to_numpy(dtype=float))
            upper = np.maximum(interval[q10_col].to_numpy(dtype=float), interval[q90_col].to_numpy(dtype=float))
            actual = interval["actual"].to_numpy(dtype=float)
            covered = (actual >= lower) & (actual <= upper)
            metrics["interval_80_coverage"] = float(np.mean(covered))
            metrics["interval_80_width"] = float(np.mean(upper - lower))
    return metrics


def train_single_model(bundle: DatasetBundle, config: PrototypeConfig, output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    preprocessor = fit_feature_preprocessor(bundle)
    target_preprocessor = fit_target_preprocessor(
        bundle,
        config.target_normalization,
        config.target_mode,
        config.residual_clip_lower_q,
        config.residual_clip_upper_q,
    )
    train_ds = EpsSequenceDataset(bundle, "train", preprocessor)
    val_ds = EpsSequenceDataset(bundle, "val", preprocessor)
    test_ds = EpsSequenceDataset(bundle, "test", preprocessor)
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Train/val/test splits must all be non-empty.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tickers = int(bundle.metadata["ticker_id"].max()) + 1
    model = SequenceRegressor(
        seq_dim=train_ds.sequences.shape[-1],
        static_dim=train_ds.static.shape[-1],
        num_tickers=num_tickers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        max_len=config.seq_len,
        model_type=config.model_type,
        pooling=config.pooling,
        ticker_embedding_dim=config.ticker_embedding_dim,
        output_dim=len(config.quantiles) if config.prediction_objective == "quantile" else 1,
    ).to(device)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.optimizer_momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = math.inf
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        train_loss, train_mae, train_rmse = run_epoch(model, train_loader, optimizer, device, target_preprocessor, config)
        val_loss, val_mae, val_rmse = run_epoch(model, val_loader, None, device, target_preprocessor, config)
        history.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            best_state = {"model_state_dict": model.state_dict(), "config": vars(config), "epoch": epoch}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                break
        scheduler.step(val_mae)
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state["model_state_dict"])

    test_predictions, test_quantiles = predict_split(model, test_ds, device, target_preprocessor, config)
    test_actuals = test_ds.targets
    test_baseline = pd.to_numeric(test_ds.metadata.get("baseline_persistence", test_ds.metadata.get("last_prior_eps")), errors="coerce").to_numpy(dtype=np.float32)
    test_ticker_id = test_ds.metadata["ticker_id"].to_numpy(dtype=np.int64)
    test_predictions_tensor = torch.tensor(test_predictions, dtype=torch.float32)
    test_baseline_tensor = torch.tensor(test_baseline, dtype=torch.float32)
    test_ticker_tensor = torch.tensor(test_ticker_id, dtype=torch.long)
    test_predictions = apply_tail_hardening(
        test_predictions_tensor,
        test_baseline_tensor,
        test_ticker_tensor,
        target_preprocessor,
        config.cold_start_baseline_min_train_samples,
        config.residual_clip_mode,
    ).cpu().numpy()
    if config.prediction_objective == "quantile" and test_quantiles:
        ordered_keys = sorted(test_quantiles.keys())
        quantile_tensor = torch.tensor(
            np.column_stack([test_quantiles[key] for key in ordered_keys]), dtype=torch.float32
        )
        quantile_tensor = apply_tail_hardening(
            quantile_tensor,
            test_baseline_tensor,
            test_ticker_tensor,
            target_preprocessor,
            config.cold_start_baseline_min_train_samples,
            config.residual_clip_mode,
        ).cpu().numpy()
        quantile_tensor = np.sort(quantile_tensor, axis=1)
        test_quantiles = {key: quantile_tensor[:, idx] for idx, key in enumerate(ordered_keys)}
    alpha = float(config.prediction_blend_alpha)
    if alpha != 1.0:
        valid_baseline = np.isfinite(test_baseline)
        blended = test_predictions.copy()
        blended[valid_baseline] = alpha * test_predictions[valid_baseline] + (1.0 - alpha) * test_baseline[valid_baseline]
        test_predictions = blended
        for key, values in list(test_quantiles.items()):
            adjusted = values.copy()
            adjusted[valid_baseline] = alpha * adjusted[valid_baseline] + (1.0 - alpha) * test_baseline[valid_baseline]
            test_quantiles[key] = adjusted

    test_metadata = test_ds.metadata.copy()
    test_metadata["prediction"] = test_predictions
    test_metadata["actual"] = test_actuals
    if config.prediction_objective == "quantile":
        ordered_keys = sorted(test_quantiles.keys())
        quantile_matrix = None
        if ordered_keys:
            quantile_matrix = np.column_stack([test_quantiles[key] for key in ordered_keys])
            quantile_matrix = np.sort(quantile_matrix, axis=1)
            for idx, key in enumerate(ordered_keys):
                test_metadata[f"prediction_{key}"] = quantile_matrix[:, idx]
    selected_baseline = select_best_baseline(bundle.metadata, split="val")
    selected_baseline_name = selected_baseline["name"]
    if selected_baseline_name and selected_baseline_name in test_metadata.columns:
        test_metadata["baseline_selected"] = pd.to_numeric(test_metadata[selected_baseline_name], errors="coerce")
    else:
        test_metadata["baseline_selected"] = np.nan
    test_metadata.to_csv(output_dir / "test_predictions.csv", index=False)

    test_metrics = _compute_prediction_metrics(test_metadata)
    metrics = {
        "best_epoch": int(best_state["epoch"]),
        "stopped_early": stopped_early,
        "epochs_completed": int(len(history)),
        "model_type": config.model_type,
        "optimizer": config.optimizer,
        "prediction_objective": config.prediction_objective,
        "quantiles": list(config.quantiles),
        "val_mae": float(best_val_mae),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "interval_80_coverage": test_metrics.get("interval_80_coverage"),
        "interval_80_width": test_metrics.get("interval_80_width"),
        "val_baselines": evaluate_baselines(bundle.metadata, "val"),
        "test_baselines": evaluate_baselines(bundle.metadata, "test"),
        "selected_val_baseline": selected_baseline_name,
        "selected_val_baseline_mae": selected_baseline["mae"],
        "num_train": int(len(train_ds)),
        "num_val": int(len(val_ds)),
        "num_test": int(len(test_ds)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    return metrics, test_metadata


def train_per_sector(bundle: DatasetBundle, config: PrototypeConfig, output_dir: Path) -> dict[str, Any]:
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
        metrics, test_predictions = train_single_model(bucket_bundle, config, bucket_dir)
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
        "num_sector_models": int(len(sector_metrics_rows)),
        "test_mae": float(aggregate_metrics["mae"]),
        "test_rmse": float(aggregate_metrics["rmse"]),
        "interval_80_coverage": aggregate_metrics.get("interval_80_coverage"),
        "interval_80_width": aggregate_metrics.get("interval_80_width"),
        "test_count": int(aggregate_metrics["count"]),
        "test_baselines": aggregate_baselines,
        "val_baseline_selection_by_sector": val_baseline_summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a transformer on the EPS prototype dataset.")
    parser.add_argument("--config", default="prototype_config.json", help="Path to config JSON.")
    parser.add_argument("--dataset-dir", default="artifacts/prototype_dataset", help="Directory containing dataset artifacts.")
    parser.add_argument("--output-dir", default="artifacts/prototype_training", help="Directory for checkpoints and metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PrototypeConfig.from_path(args.config)
    set_seed(config.seed)
    bundle = load_dataset(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if config.sector_modeling_mode == "per_sector" and "sector_bucket" in bundle.metadata.columns:
        metrics = train_per_sector(bundle, config, output_dir)
    else:
        metrics, _ = train_single_model(bundle, config, output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
