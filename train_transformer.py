from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset_builder import PrototypeConfig


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

        sequences = (
            clipped_sequences - preprocessor.sequence_mean[None, None, :]
        ) / preprocessor.sequence_std[None, None, :]
        static = (clipped_static - preprocessor.static_mean[None, :]) / preprocessor.static_std[None, :]

        self.sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        self.static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
        self.targets = bundle.targets[source_idx]
        self.ticker_ids = self.metadata["ticker_id"].to_numpy(dtype=np.int64)
        self.baselines = (
            pd.to_numeric(self.metadata["last_prior_eps"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

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
    ) -> None:
        super().__init__()
        if model_type == "transformer":
            self.sequence_encoder = TransformerSequenceEncoder(
                seq_dim=seq_dim,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                hidden_dim=hidden_dim,
                max_len=max_len,
                pooling=pooling,
            )
            sequence_out_dim = d_model
        elif model_type == "gru":
            self.sequence_encoder = GruSequenceEncoder(
                seq_dim=seq_dim,
                d_model=d_model,
                num_layers=num_layers,
                dropout=dropout,
            )
            sequence_out_dim = d_model
        elif model_type == "cnn":
            self.sequence_encoder = CnnSequenceEncoder(
                seq_dim=seq_dim,
                d_model=d_model,
                dropout=dropout,
            )
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
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        static: torch.Tensor,
        ticker_id: torch.Tensor,
    ) -> torch.Tensor:
        seq_emb = self.sequence_encoder(sequence)
        static_emb = self.static_proj(static)
        ticker_emb = self.ticker_embedding(ticker_id)
        output = self.head(torch.cat([seq_emb, static_emb, ticker_emb], dim=-1))
        return output.squeeze(-1)


def fit_feature_preprocessor(bundle: DatasetBundle) -> FeaturePreprocessor:
    train_idx = bundle.metadata.index[bundle.metadata["split"] == "train"].to_numpy(dtype=np.int64)
    if len(train_idx) == 0:
        raise RuntimeError("No training samples available for feature preprocessing.")

    seq_train = bundle.sequences[train_idx]
    static_train = bundle.static[train_idx]

    sequence_lower = np.nanquantile(seq_train, 0.01, axis=(0, 1)).astype(np.float32)
    sequence_upper = np.nanquantile(seq_train, 0.99, axis=(0, 1)).astype(np.float32)
    clipped_seq_train = np.clip(
        seq_train,
        sequence_lower[None, None, :],
        sequence_upper[None, None, :],
    )
    sequence_mean = np.nanmean(clipped_seq_train, axis=(0, 1)).astype(np.float32)
    sequence_std = np.nanstd(clipped_seq_train, axis=(0, 1)).astype(np.float32)
    sequence_std = np.where(sequence_std < 1e-6, 1.0, sequence_std).astype(np.float32)

    static_lower = np.nanquantile(static_train, 0.01, axis=0).astype(np.float32)
    static_upper = np.nanquantile(static_train, 0.99, axis=0).astype(np.float32)
    clipped_static_train = np.clip(static_train, static_lower[None, :], static_upper[None, :])
    static_mean = np.nanmean(clipped_static_train, axis=0).astype(np.float32)
    static_std = np.nanstd(clipped_static_train, axis=0).astype(np.float32)
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
    train_idx = bundle.metadata.index[bundle.metadata["split"] == "train"].to_numpy(dtype=np.int64)
    target_train = bundle.targets[train_idx]
    train_meta = bundle.metadata.iloc[train_idx].copy()
    mean = float(np.mean(target_train))
    std = float(np.std(target_train))
    if std < 1e-6:
        std = 1.0
    ticker_means: dict[int, float] = {}
    ticker_stds: dict[int, float] = {}
    ticker_train_counts: dict[int, int] = {}
    ticker_delta_lower: dict[int, float] = {}
    ticker_delta_upper: dict[int, float] = {}
    ticker_targets = pd.DataFrame(
        {"ticker_id": train_meta["ticker_id"].to_numpy(dtype=int), "target": target_train}
    )
    baseline_train = pd.to_numeric(train_meta["last_prior_eps"], errors="coerce").to_numpy(dtype=float)
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

    delta_frame = pd.DataFrame(
        {
            "ticker_id": train_meta["ticker_id"].to_numpy(dtype=int),
            "delta": delta_train,
        }
    ).dropna(subset=["delta"])
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


def transform_target(
    target: torch.Tensor,
    baseline: torch.Tensor,
    ticker_id: torch.Tensor,
    target_preprocessor: TargetPreprocessor,
) -> torch.Tensor:
    mode = target_preprocessor.mode
    if mode == "raw":
        transformed = target
    elif mode == "delta_last":
        transformed = target - baseline
    elif mode == "signed_log":
        transformed = torch.sign(target) * torch.log1p(torch.abs(target))
    elif mode == "ticker_zscore":
        means = torch.tensor(
            [
                target_preprocessor.ticker_means.get(int(x), target_preprocessor.mean)
                for x in ticker_id.cpu().tolist()
            ],
            dtype=target.dtype,
            device=target.device,
        )
        stds = torch.tensor(
            [
                target_preprocessor.ticker_stds.get(int(x), target_preprocessor.std)
                for x in ticker_id.cpu().tolist()
            ],
            dtype=target.dtype,
            device=target.device,
        )
        transformed = (target - means) / stds
    else:
        raise ValueError(f"Unsupported target_mode: {mode}")

    if target_preprocessor.enabled:
        transformed = (transformed - target_preprocessor.mean) / target_preprocessor.std
    return transformed


def inverse_transform_prediction(
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    ticker_id: torch.Tensor,
    target_preprocessor: TargetPreprocessor,
) -> torch.Tensor:
    restored = prediction
    if target_preprocessor.enabled:
        restored = restored * target_preprocessor.std + target_preprocessor.mean

    mode = target_preprocessor.mode
    if mode == "raw":
        return restored
    if mode == "delta_last":
        return baseline + restored
    if mode == "signed_log":
        return torch.sign(restored) * (torch.expm1(torch.abs(restored)))
    if mode == "ticker_zscore":
        means = torch.tensor(
            [
                target_preprocessor.ticker_means.get(int(x), target_preprocessor.mean)
                for x in ticker_id.cpu().tolist()
            ],
            dtype=prediction.dtype,
            device=prediction.device,
        )
        stds = torch.tensor(
            [
                target_preprocessor.ticker_stds.get(int(x), target_preprocessor.std)
                for x in ticker_id.cpu().tolist()
            ],
            dtype=prediction.dtype,
            device=prediction.device,
        )
        return means + restored * stds
    raise ValueError(f"Unsupported target_mode: {mode}")


def apply_tail_hardening(
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    ticker_id: torch.Tensor,
    target_preprocessor: TargetPreprocessor,
    min_train_samples_for_model: int,
    residual_clip_mode: str,
) -> torch.Tensor:
    hardened = prediction.clone()

    if residual_clip_mode != "none":
        residual = hardened - baseline
        valid_baseline = torch.isfinite(baseline)
        if residual_clip_mode == "global":
            clipped = torch.clamp(
                residual,
                min=target_preprocessor.global_delta_lower,
                max=target_preprocessor.global_delta_upper,
            )
            residual = torch.where(valid_baseline, clipped, residual)
        elif residual_clip_mode == "per_ticker":
            lower = torch.tensor(
                [
                    target_preprocessor.ticker_delta_lower.get(int(x), target_preprocessor.global_delta_lower)
                    for x in ticker_id.cpu().tolist()
                ],
                dtype=prediction.dtype,
                device=prediction.device,
            )
            upper = torch.tensor(
                [
                    target_preprocessor.ticker_delta_upper.get(int(x), target_preprocessor.global_delta_upper)
                    for x in ticker_id.cpu().tolist()
                ],
                dtype=prediction.dtype,
                device=prediction.device,
            )
            clipped = torch.minimum(torch.maximum(residual, lower), upper)
            residual = torch.where(valid_baseline, clipped, residual)
        else:
            raise ValueError(f"Unsupported residual_clip_mode: {residual_clip_mode}")
        candidate = baseline + residual
        hardened = torch.where(valid_baseline, candidate, hardened)

    if min_train_samples_for_model > 0:
        train_counts = torch.tensor(
            [
                target_preprocessor.ticker_train_counts.get(int(x), 0)
                for x in ticker_id.cpu().tolist()
            ],
            dtype=prediction.dtype,
            device=prediction.device,
        )
        use_baseline = train_counts < float(min_train_samples_for_model)
        valid_baseline = torch.isfinite(baseline)
        hardened = torch.where((use_baseline.bool() & valid_baseline), baseline, hardened)

    return hardened


def evaluate_baseline(metadata: pd.DataFrame, split: str) -> dict[str, float]:
    frame = metadata.copy()
    frame = frame[frame["split"] == split].copy()
    frame["prediction"] = pd.to_numeric(frame["last_prior_eps"], errors="coerce")
    frame["target"] = pd.to_numeric(frame["target_basic_eps"], errors="coerce")
    frame = frame.dropna(subset=["prediction", "target"])
    if frame.empty:
        return {"mae": math.nan, "rmse": math.nan, "count": 0}

    diff = frame["prediction"] - frame["target"]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    return {"mae": mae, "rmse": rmse, "count": int(len(frame))}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_preprocessor: TargetPreprocessor,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.HuberLoss()
    losses: list[float] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []

    for batch in loader:
        sequence = batch["sequence"].to(device)
        static = batch["static"].to(device)
        ticker_id = batch["ticker_id"].to(device)
        baseline = batch["baseline"].to(device)
        target = batch["target"].to(device)
        normalized_target = transform_target(target, baseline, ticker_id, target_preprocessor)

        with torch.set_grad_enabled(training):
            normalized_prediction = model(sequence, static, ticker_id)
            loss = criterion(normalized_prediction, normalized_target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        prediction = inverse_transform_prediction(
            normalized_prediction.detach(),
            baseline,
            ticker_id,
            target_preprocessor,
        )
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
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            baseline = batch["baseline"].to(device)
            ticker_id = batch["ticker_id"].to(device)
            pred = model(
                batch["sequence"].to(device),
                batch["static"].to(device),
                ticker_id,
            )
            pred = inverse_transform_prediction(pred, baseline, ticker_id, target_preprocessor)
            predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a transformer on the EPS prototype dataset.")
    parser.add_argument("--config", default="prototype_config.json", help="Path to config JSON.")
    parser.add_argument(
        "--dataset-dir",
        default="artifacts/prototype_dataset",
        help="Directory containing dataset artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/prototype_training",
        help="Directory for checkpoints and metrics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PrototypeConfig.from_path(args.config)
    set_seed(config.seed)

    bundle = load_dataset(args.dataset_dir)
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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = math.inf
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        train_loss, train_mae, train_rmse = run_epoch(
            model, train_loader, optimizer, device, target_preprocessor
        )
        val_loss, val_mae, val_rmse = run_epoch(
            model, val_loader, None, device, target_preprocessor
        )
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
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": vars(config),
                "epoch": epoch,
            }
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

    test_predictions = predict_split(model, test_ds, device, target_preprocessor)
    test_actuals = test_ds.targets
    test_baseline = pd.to_numeric(test_ds.metadata["last_prior_eps"], errors="coerce").to_numpy(
        dtype=np.float32
    )
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
    alpha = float(config.prediction_blend_alpha)
    if alpha != 1.0:
        valid_baseline = np.isfinite(test_baseline)
        blended = test_predictions.copy()
        blended[valid_baseline] = (
            alpha * test_predictions[valid_baseline]
            + (1.0 - alpha) * test_baseline[valid_baseline]
        )
        test_predictions = blended
    test_mae = float(np.mean(np.abs(test_predictions - test_actuals)))
    test_rmse = float(np.sqrt(np.mean(np.square(test_predictions - test_actuals))))

    test_metadata = test_ds.metadata.copy()
    test_metadata["prediction"] = test_predictions
    test_metadata["actual"] = test_actuals
    test_metadata.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics = {
        "best_epoch": int(best_state["epoch"]),
        "stopped_early": stopped_early,
        "epochs_completed": int(len(history)),
        "model_type": config.model_type,
        "optimizer": config.optimizer,
        "val_mae": float(best_val_mae),
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "val_baseline": evaluate_baseline(bundle.metadata, "val"),
        "test_baseline": evaluate_baseline(bundle.metadata, "test"),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
