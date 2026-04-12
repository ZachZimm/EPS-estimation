from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from database_helper import KairosDatabaseHelper
from external_data import ExternalSeriesCache


BASE_SEQUENCE_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volatility_60d",
    "close_to_sma20",
    "close_to_sma60",
    "close_to_sma200",
    "high_low_range",
    "open_close_gap",
    "volume_z20",
    "volume_z60",
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "rsi_14",
    "rsi_28",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bollinger_pos_20",
]

STATIC_FEATURE_COLUMNS = [
    "prev_eps_1",
    "prev_eps_2",
    "prev_eps_3",
    "prev_eps_4",
    "prev_eps_mean_4",
    "prev_eps_std_4",
    "prev_eps_growth_last",
    "prev_eps_yoy_change",
    "days_since_last_eps",
    "quarter_sin",
    "quarter_cos",
]


@dataclass
class PrototypeConfig:
    tickers: list[str]
    seq_len: int
    train_end: str
    val_end: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    d_model: int
    num_heads: int
    num_layers: int
    dropout: float
    hidden_dim: int
    seed: int
    early_stopping_patience: int = 5
    optimizer_momentum: float = 0.9
    optimizer: str = "adamw"
    model_type: str = "transformer"
    pooling: str = "mean"
    ticker_embedding_dim: int = 16
    target_normalization: bool = True
    target_mode: str = "raw"
    prediction_blend_alpha: float = 1.0
    cold_start_baseline_min_train_samples: int = 0
    residual_clip_mode: str = "none"
    residual_clip_lower_q: float = 0.01
    residual_clip_upper_q: float = 0.99
    macro_cache_dir: str = "artifacts/external_cache"
    fred_series: list[str] | None = None
    market_context_tickers: list[str] | None = None
    refresh_external_cache: bool = False

    @classmethod
    def from_path(cls, path: str | Path) -> "PrototypeConfig":
        payload = json.loads(Path(path).read_text())
        return cls(**payload)


def _compute_market_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    df = ohlc_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)
    df["return_60d"] = df["Close"].pct_change(60)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["volatility_60d"] = df["return_1d"].rolling(60).std()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma60"] = df["Close"].rolling(60).mean()
    df["sma200"] = df["Close"].rolling(200).mean()
    df["close_to_sma20"] = df["Close"] / df["sma20"] - 1.0
    df["close_to_sma60"] = df["Close"] / df["sma60"] - 1.0
    df["close_to_sma200"] = df["Close"] / df["sma200"] - 1.0
    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_gap"] = (df["Close"] - df["Open"]) / df["Open"]
    df["log_volume"] = np.log1p(df["Volume"])
    df["volume_z20"] = (
        (df["log_volume"] - df["log_volume"].rolling(20).mean())
        / df["log_volume"].rolling(20).std()
    )
    df["volume_z60"] = (
        (df["log_volume"] - df["log_volume"].rolling(60).mean())
        / df["log_volume"].rolling(60).std()
    )

    df["momentum_20"] = df["Close"] / df["Close"].shift(20) - 1.0
    df["momentum_60"] = df["Close"] / df["Close"].shift(60) - 1.0
    df["momentum_120"] = df["Close"] / df["Close"].shift(120) - 1.0

    close_diff = df["Close"].diff()
    gain_14 = close_diff.clip(lower=0).rolling(14).mean()
    loss_14 = (-close_diff.clip(upper=0)).rolling(14).mean()
    rs_14 = gain_14 / loss_14.replace(0, np.nan)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs_14))

    gain_28 = close_diff.clip(lower=0).rolling(28).mean()
    loss_28 = (-close_diff.clip(upper=0)).rolling(28).mean()
    rs_28 = gain_28 / loss_28.replace(0, np.nan)
    df["rsi_28"] = 100.0 - (100.0 / (1.0 + rs_28))

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    rolling_std_20 = df["Close"].rolling(20).std()
    bollinger_denom = (4.0 * rolling_std_20).replace(0, np.nan)
    df["bollinger_pos_20"] = (df["Close"] - (df["sma20"] - 2.0 * rolling_std_20)) / bollinger_denom
    return df


def _context_prefix(name: str) -> str:
    prefix = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    prefix = "_".join(part for part in prefix.split("_") if part)
    return f"ctx_{prefix}"


def _compute_context_market_features(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    df = frame.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    prefix = _context_prefix(name)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df[f"{prefix}_return_1d"] = df["Close"].pct_change()
    df[f"{prefix}_return_5d"] = df["Close"].pct_change(5)
    df[f"{prefix}_return_20d"] = df["Close"].pct_change(20)
    df[f"{prefix}_volatility_20d"] = df[f"{prefix}_return_1d"].rolling(20, min_periods=5).std()
    df[f"{prefix}_level_z60"] = (
        (df["Close"] - df["Close"].rolling(60, min_periods=10).mean())
        / df["Close"].rolling(60, min_periods=10).std()
    )
    df[f"{prefix}_rel_ema20"] = df["Close"] / df["Close"].ewm(span=20, adjust=False).mean() - 1.0
    marker_column = f"{prefix}_source_date"
    feature_columns = [
        f"{prefix}_return_1d",
        f"{prefix}_return_5d",
        f"{prefix}_return_20d",
        f"{prefix}_volatility_20d",
        f"{prefix}_level_z60",
        f"{prefix}_rel_ema20",
    ]
    output = df[["Date", *feature_columns]].copy()
    output[marker_column] = output["Date"]
    return output, feature_columns, marker_column


def _compute_fred_daily_features(series_id: str, releases: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = releases.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "realtime_start", "value"]).sort_values(
        ["realtime_start", "date"]
    )
    if frame.empty:
        return pd.DataFrame(columns=["Date"])

    value_by_release_day = (
        frame.sort_values(["realtime_start", "date"])
        .drop_duplicates(subset=["realtime_start", "date"], keep="last")
        .rename(columns={"realtime_start": "Date"})
        [["Date", "value"]]
    )
    daily = (
        value_by_release_day.groupby("Date", as_index=False)
        .last()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    prefix = _context_prefix(series_id)
    daily[f"{prefix}_value"] = daily["value"]
    daily[f"{prefix}_delta_1rel"] = daily["value"].diff(1)
    daily[f"{prefix}_delta_4rel"] = daily["value"].diff(4)
    daily[f"{prefix}_z60"] = (
        (daily["value"] - daily["value"].rolling(12, min_periods=3).mean())
        / daily["value"].rolling(12, min_periods=3).std()
    )
    feature_columns = [
        f"{prefix}_value",
        f"{prefix}_delta_1rel",
        f"{prefix}_delta_4rel",
        f"{prefix}_z60",
    ]
    marker_column = f"{prefix}_source_date"
    output = daily[["Date", *feature_columns]].copy()
    output[marker_column] = output["Date"]
    return output, feature_columns, marker_column


def _build_macro_feature_bundle(
    config: PrototypeConfig,
    env_file: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cache = ExternalSeriesCache(env_file=env_file, cache_dir=config.macro_cache_dir)
    frames: list[pd.DataFrame] = []
    feature_columns: list[str] = []
    marker_columns: list[str] = []

    for series_id in config.fred_series or []:
        daily, daily_features, marker_column = _compute_fred_daily_features(
            series_id,
            cache.get_fred_series(series_id, refresh=config.refresh_external_cache),
        )
        if daily.empty:
            continue
        frames.append(daily)
        feature_columns.extend(daily_features)
        marker_columns.append(marker_column)

    for symbol in config.market_context_tickers or []:
        daily, daily_features, marker_column = _compute_context_market_features(
            symbol,
            cache.get_market_series(symbol, refresh=config.refresh_external_cache),
        )
        if daily.empty:
            continue
        frames.append(daily)
        feature_columns.extend(daily_features)
        marker_columns.append(marker_column)

    if not frames:
        return pd.DataFrame(), [], []

    merged = frames[0].sort_values("Date").reset_index(drop=True)
    for frame in frames[1:]:
        merged = pd.merge_ordered(
            merged,
            frame.sort_values("Date"),
            on="Date",
            how="outer",
        )
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged[feature_columns] = merged[feature_columns].ffill()
    merged[marker_columns] = merged[marker_columns].ffill()
    return merged, feature_columns, marker_columns


def _compute_static_features(history: pd.DataFrame, target_row: pd.Series) -> dict[str, float]:
    previous_eps = history.sort_values("publishedDate")["BasicEPS"].tolist()
    last_four = previous_eps[-4:]
    padded = [np.nan] * (4 - len(last_four)) + last_four

    if len(previous_eps) >= 2 and pd.notna(previous_eps[-1]) and pd.notna(previous_eps[-2]):
        prev_growth = previous_eps[-1] - previous_eps[-2]
    else:
        prev_growth = np.nan

    if len(previous_eps) >= 4 and pd.notna(previous_eps[-1]) and pd.notna(previous_eps[-4]):
        prev_yoy_change = previous_eps[-1] - previous_eps[-4]
    else:
        prev_yoy_change = np.nan

    quarter = int(pd.Timestamp(target_row["asOfDate"]).quarter)
    quarter_angle = 2.0 * np.pi * (quarter - 1) / 4.0

    if history.empty:
        days_since_last_eps = np.nan
    else:
        last_published = pd.Timestamp(history["publishedDate"].max())
        days_since_last_eps = float(
            (pd.Timestamp(target_row["publishedDate"]) - last_published).days
        )

    return {
        "prev_eps_1": padded[-1],
        "prev_eps_2": padded[-2],
        "prev_eps_3": padded[-3],
        "prev_eps_4": padded[-4],
        "prev_eps_mean_4": float(np.nanmean(last_four)) if last_four else np.nan,
        "prev_eps_std_4": float(np.nanstd(last_four)) if last_four else np.nan,
        "prev_eps_growth_last": prev_growth,
        "prev_eps_yoy_change": prev_yoy_change,
        "days_since_last_eps": days_since_last_eps,
        "quarter_sin": float(np.sin(quarter_angle)),
        "quarter_cos": float(np.cos(quarter_angle)),
    }


def _split_name(published_date: pd.Timestamp, config: PrototypeConfig) -> str:
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)
    if published_date <= train_end:
        return "train"
    if published_date <= val_end:
        return "val"
    return "test"


def build_event_dataset(
    helper: KairosDatabaseHelper,
    config: PrototypeConfig,
    env_file: str | Path = ".env",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    macro_df, macro_feature_columns, macro_marker_columns = _build_macro_feature_bundle(config, env_file)
    sequence_feature_columns = list(BASE_SEQUENCE_FEATURE_COLUMNS)

    metadata_rows: list[dict[str, Any]] = []
    sequence_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    targets: list[float] = []

    for ticker in config.tickers:
        ohlc_df = helper.get_ohlc_dataframe(ticker)
        eps_df = helper.get_eps_dataframe(
            ticker,
            period_types=("3M",),
            eps_columns=("BasicEPS",),
        )
        if ohlc_df.empty or eps_df.empty:
            continue

        ohlc_df = _compute_market_features(ohlc_df)
        if not macro_df.empty:
            ohlc_df = pd.merge_ordered(
                ohlc_df.sort_values("Date"),
                macro_df,
                on="Date",
                how="left",
            )
            ohlc_df[macro_feature_columns] = ohlc_df[macro_feature_columns].ffill()
            ohlc_df[macro_marker_columns] = ohlc_df[macro_marker_columns].ffill()
            derived_macro_columns: list[str] = []
            for marker_column in macro_marker_columns:
                base_prefix = marker_column[: -len("_source_date")]
                available_column = f"{base_prefix}_available"
                staleness_column = f"{base_prefix}_staleness_days"
                ohlc_df[available_column] = ohlc_df[marker_column].notna().astype(float)
                ohlc_df[staleness_column] = (
                    (pd.to_datetime(ohlc_df["Date"]) - pd.to_datetime(ohlc_df[marker_column]))
                    .dt.days.astype(float)
                )
                ohlc_df.loc[ohlc_df[available_column] == 0.0, staleness_column] = 9999.0
                derived_macro_columns.extend([available_column, staleness_column])
            sequence_feature_columns = (
                list(BASE_SEQUENCE_FEATURE_COLUMNS)
                + list(macro_feature_columns)
                + derived_macro_columns
            )
        eps_df = eps_df[(eps_df["periodType"] == "3M") & eps_df["BasicEPS"].notna()].copy()
        eps_df = eps_df.sort_values("publishedDate").reset_index(drop=True)
        if eps_df.empty:
            continue

        for idx, row in eps_df.iterrows():
            published_date = pd.Timestamp(row["publishedDate"])
            market_history = ohlc_df[ohlc_df["Date"] < published_date].copy()
            if len(market_history) < config.seq_len:
                continue

            market_window = market_history.tail(config.seq_len).copy()
            if market_window[BASE_SEQUENCE_FEATURE_COLUMNS].isna().any().any():
                continue

            prior_eps = eps_df.iloc[:idx].copy()
            static_features = _compute_static_features(prior_eps, row)
            static_values = np.array(
                [static_features[column] for column in STATIC_FEATURE_COLUMNS], dtype=np.float32
            )

            sequence_rows.append(
                market_window[sequence_feature_columns].to_numpy(dtype=np.float32)
            )
            static_rows.append(static_values)
            targets.append(float(row["BasicEPS"]))
            metadata_rows.append(
                {
                    "sample_id": len(metadata_rows),
                    "ticker": ticker,
                    "ticker_id": config.tickers.index(ticker),
                    "target_as_of_date": pd.Timestamp(row["asOfDate"]).isoformat(),
                    "target_published_date": published_date.isoformat(),
                    "split": _split_name(published_date, config),
                    "target_basic_eps": float(row["BasicEPS"]),
                    "last_observed_market_date": pd.Timestamp(market_window["Date"].iloc[-1]).isoformat(),
                    "last_prior_eps": static_features["prev_eps_1"],
                }
            )

    metadata = pd.DataFrame(metadata_rows)
    if metadata.empty:
        raise RuntimeError("No samples were generated. Check ticker coverage and sequence length.")

    sequences = np.stack(sequence_rows).astype(np.float32)
    static = np.stack(static_rows).astype(np.float32)
    target_array = np.array(targets, dtype=np.float32)
    metadata.attrs["sequence_feature_columns"] = sequence_feature_columns
    return metadata, sequences, static, target_array


def _fit_normalization(
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    static: np.ndarray,
    sequence_feature_columns: list[str],
) -> dict[str, list[float]]:
    train_idx = metadata.index[metadata["split"] == "train"].to_numpy()
    if len(train_idx) == 0:
        raise RuntimeError("No train samples found for normalization.")

    seq_train = sequences[train_idx]
    static_train = static[train_idx]

    seq_mean = np.nanmean(seq_train, axis=(0, 1))
    seq_std = np.nanstd(seq_train, axis=(0, 1))
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)

    static_mean = np.nanmean(static_train, axis=0)
    static_std = np.nanstd(static_train, axis=0)
    static_std = np.where(static_std < 1e-6, 1.0, static_std)

    return {
        "sequence_feature_columns": sequence_feature_columns,
        "static_feature_columns": STATIC_FEATURE_COLUMNS,
        "sequence_mean": seq_mean.tolist(),
        "sequence_std": seq_std.tolist(),
        "static_mean": static_mean.tolist(),
        "static_std": static_std.tolist(),
    }


def save_dataset(
    output_dir: str | Path,
    config: PrototypeConfig,
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    static: np.ndarray,
    targets: np.ndarray,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sequence_feature_columns = metadata.attrs.get(
        "sequence_feature_columns",
        BASE_SEQUENCE_FEATURE_COLUMNS,
    )
    normalization = _fit_normalization(metadata, sequences, static, sequence_feature_columns)

    metadata.to_csv(output_path / "event_metadata.csv", index=False)
    np.savez_compressed(
        output_path / "dataset_arrays.npz",
        sequences=sequences,
        static=static,
        targets=targets,
    )
    (output_path / "normalization.json").write_text(json.dumps(normalization, indent=2))
    (output_path / "config.json").write_text(
        json.dumps(
            {
                "tickers": config.tickers,
                "seq_len": config.seq_len,
                "train_end": config.train_end,
                "val_end": config.val_end,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a leak-free EPS event dataset.")
    parser.add_argument("--config", default="prototype_config.json", help="Path to config JSON.")
    parser.add_argument("--env-file", default=".env", help="Path to env file.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/prototype_dataset",
        help="Directory for metadata and array artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PrototypeConfig.from_path(args.config)
    helper = KairosDatabaseHelper(args.env_file)
    metadata, sequences, static, targets = build_event_dataset(helper, config, env_file=args.env_file)
    save_dataset(args.output_dir, config, metadata, sequences, static, targets)
    split_counts = metadata["split"].value_counts().to_dict()
    print(
        json.dumps(
            {
                "num_samples": int(len(metadata)),
                "split_counts": split_counts,
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
