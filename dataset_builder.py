from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
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

BASE_STATIC_FEATURE_COLUMNS = [
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

BASELINE_COLUMNS = [
    "baseline_persistence",
    "baseline_seasonal_naive",
    "baseline_trailing_mean",
    "baseline_trend",
    "baseline_sector_peer_median",
]

DEFAULT_FUNDAMENTAL_FEATURE_GROUPS: dict[str, list[str]] = {
    "income": [
        "TotalRevenue",
        "GrossProfit",
        "OperatingIncome",
        "NetIncome",
        "EBITDA",
        "ResearchAndDevelopment",
        "SellingGeneralAndAdministration",
    ],
    "balance_sheet": [
        "CashAndCashEquivalents",
        "CurrentAssets",
        "CurrentLiabilities",
        "Inventory",
        "AccountsReceivable",
        "TotalDebt",
        "NetDebt",
        "StockholdersEquity",
        "WorkingCapital",
        "TotalAssets",
    ],
    "cash_flow": [
        "OperatingCashFlow",
        "FreeCashFlow",
        "CapitalExpenditure",
        "DepreciationAndAmortization",
        "StockBasedCompensation",
    ],
    "valuation": [
        "MarketCap",
        "EnterpriseValue",
        "PeRatio",
        "ForwardPeRatio",
        "PsRatio",
        "PbRatio",
        "EnterpriseValueRevenueRatio",
        "EnterprisesValueEBITDARatio",
    ],
}

TABLE_SUFFIX_BY_GROUP = {
    "income": "_q_income",
    "balance_sheet": "_q_balance_sheet",
    "cash_flow": "_q_cash_flow",
    "valuation": "_q_valuation",
}

FLOW_FIELD_GROUPS = {
    "income": {"TotalRevenue", "GrossProfit", "OperatingIncome", "NetIncome", "EBITDA", "ResearchAndDevelopment", "SellingGeneralAndAdministration"},
    "cash_flow": {"OperatingCashFlow", "FreeCashFlow", "CapitalExpenditure", "DepreciationAndAmortization", "StockBasedCompensation"},
}

RATIO_SPECS = [
    ("gross_margin", "income", "GrossProfit", "income", "TotalRevenue"),
    ("operating_margin", "income", "OperatingIncome", "income", "TotalRevenue"),
    ("net_margin", "income", "NetIncome", "income", "TotalRevenue"),
    ("free_cash_flow_margin", "cash_flow", "FreeCashFlow", "income", "TotalRevenue"),
    ("debt_to_equity", "balance_sheet", "TotalDebt", "balance_sheet", "StockholdersEquity"),
    ("working_capital_to_assets", "balance_sheet", "WorkingCapital", "balance_sheet", "TotalAssets"),
]

TRAILING_FOUR_SUM_FIELDS = [
    ("income", "TotalRevenue"),
    ("income", "EBITDA"),
    ("cash_flow", "OperatingCashFlow"),
    ("cash_flow", "FreeCashFlow"),
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
    fred_series: list[str] = field(default_factory=list)
    market_context_tickers: list[str] = field(default_factory=list)
    refresh_external_cache: bool = False
    use_pre_release_fundamentals: bool = True
    fundamental_feature_groups: dict[str, list[str]] = field(
        default_factory=lambda: {key: list(values) for key, values in DEFAULT_FUNDAMENTAL_FEATURE_GROUPS.items()}
    )
    sector_modeling_mode: str = "per_sector"
    sector_min_train_samples: int = 250
    sector_fallback_bucket: str = "OTHER"
    baseline_family: list[str] = field(default_factory=lambda: list(BASELINE_COLUMNS))
    prediction_objective: str = "point"
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    linear_include_ticker_fixed_effects: bool = True
    linear_fit_intercept: bool = True
    linear_quantile_alpha: float = 1e-4
    volatility_trim_enabled: bool = False
    volatility_trim_fraction: float = 0.0
    volatility_trim_min_history: int = 252

    @classmethod
    def from_path(cls, path: str | Path) -> "PrototypeConfig":
        payload = json.loads(Path(path).read_text())
        if "fred_series" not in payload or payload["fred_series"] is None:
            payload["fred_series"] = []
        if "market_context_tickers" not in payload or payload["market_context_tickers"] is None:
            payload["market_context_tickers"] = []
        if "fundamental_feature_groups" not in payload or payload["fundamental_feature_groups"] is None:
            payload["fundamental_feature_groups"] = {
                key: list(values) for key, values in DEFAULT_FUNDAMENTAL_FEATURE_GROUPS.items()
            }
        if "baseline_family" not in payload or payload["baseline_family"] is None:
            payload["baseline_family"] = list(BASELINE_COLUMNS)
        if "prediction_objective" not in payload or payload["prediction_objective"] is None:
            payload["prediction_objective"] = "point"
        if "quantiles" not in payload or payload["quantiles"] is None:
            payload["quantiles"] = [0.1, 0.5, 0.9]
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


def _realized_ticker_volatility(ohlc_df: pd.DataFrame, min_history: int) -> float:
    if ohlc_df.empty:
        return np.nan
    closes = pd.to_numeric(ohlc_df["Close"], errors="coerce")
    returns = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < max(2, min_history):
        return np.nan
    return float(returns.std() * np.sqrt(252.0))


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
    frame = frame.dropna(subset=["date", "realtime_start", "value"]).sort_values(["realtime_start", "date"])
    if frame.empty:
        return pd.DataFrame(columns=["Date"]), [], ""

    value_by_release_day = (
        frame.sort_values(["realtime_start", "date"])
        .drop_duplicates(subset=["realtime_start", "date"], keep="last")
        .rename(columns={"realtime_start": "Date"})[["Date", "value"]]
    )
    daily = value_by_release_day.groupby("Date", as_index=False).last().sort_values("Date").reset_index(drop=True)
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


def _build_macro_feature_bundle(config: PrototypeConfig, env_file: str | Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    cache = ExternalSeriesCache(env_file=env_file, cache_dir=config.macro_cache_dir)
    frames: list[pd.DataFrame] = []
    feature_columns: list[str] = []
    marker_columns: list[str] = []

    for series_id in config.fred_series:
        daily, daily_features, marker_column = _compute_fred_daily_features(
            series_id,
            cache.get_fred_series(series_id, refresh=config.refresh_external_cache),
        )
        if daily.empty:
            continue
        frames.append(daily)
        feature_columns.extend(daily_features)
        marker_columns.append(marker_column)

    for symbol in config.market_context_tickers:
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
        merged = pd.merge_ordered(merged, frame.sort_values("Date"), on="Date", how="outer")
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged[feature_columns] = merged[feature_columns].ffill()
    if marker_columns:
        merged[marker_columns] = merged[marker_columns].ffill()
    return merged, feature_columns, marker_columns


def _safe_divide(numerator: float | int | np.floating | None, denominator: float | int | np.floating | None) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    if abs(float(denominator)) < 1e-9:
        return np.nan
    return float(numerator) / float(denominator)


def _sanitize_feature_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name)


def _fundamental_static_feature_columns(config: PrototypeConfig) -> list[str]:
    columns: list[str] = []
    for group_name, field_names in config.fundamental_feature_groups.items():
        for field_name in field_names:
            feature_key = _sanitize_feature_name(field_name)
            columns.extend(
                [
                    f"fund_{group_name}_{feature_key}_latest",
                    f"fund_{group_name}_{feature_key}_delta1",
                    f"fund_{group_name}_{feature_key}_delta4",
                    f"fund_{group_name}_{feature_key}_missing",
                ]
            )
            if field_name in FLOW_FIELD_GROUPS.get(group_name, set()):
                columns.append(f"fund_{group_name}_{feature_key}_ttm")
    for ratio_name, *_ in RATIO_SPECS:
        columns.extend([f"ratio_{ratio_name}", f"ratio_{ratio_name}_missing"])
    return columns


def _all_static_feature_columns(config: PrototypeConfig) -> list[str]:
    columns = list(BASE_STATIC_FEATURE_COLUMNS)
    if config.use_pre_release_fundamentals:
        columns.extend(_fundamental_static_feature_columns(config))
    return columns


def _compute_eps_static_features(history: pd.DataFrame, target_row: pd.Series) -> dict[str, float]:
    previous_eps = pd.to_numeric(history.sort_values("publishedDate")["BasicEPS"], errors="coerce").tolist()
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
        days_since_last_eps = float((pd.Timestamp(target_row["publishedDate"]) - last_published).days)

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


def _prepare_fundamental_frames(helper: KairosDatabaseHelper, ticker: str, config: PrototypeConfig) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if not config.use_pre_release_fundamentals:
        return frames

    for group_name, columns in config.fundamental_feature_groups.items():
        table_suffix = TABLE_SUFFIX_BY_GROUP.get(group_name)
        if table_suffix is None:
            continue
        frame = helper.get_quarterly_table_dataframe(
            ticker,
            table_suffix=table_suffix,
            columns=["asOfDate", "periodType", *columns],
            period_types=("3M",),
        )
        if frame.empty:
            continue
        if "periodType" in frame.columns:
            frame = frame[frame["periodType"] == "3M"].copy()
        frame = frame.sort_values("publishedDate").reset_index(drop=True)
        frames[group_name] = frame
    return frames


def _series_latest(series: pd.Series, offset: int = 0) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().tolist()
    if len(clean) <= offset:
        return np.nan
    return float(clean[-1 - offset])


def _build_fundamental_static_features(
    frames_by_group: dict[str, pd.DataFrame],
    target_published_date: pd.Timestamp,
    config: PrototypeConfig,
) -> dict[str, float]:
    features: dict[str, float] = {}
    prior_frames: dict[str, pd.DataFrame] = {}
    for group_name, frame in frames_by_group.items():
        prior_frames[group_name] = frame[frame["publishedDate"] < target_published_date].copy()

    for group_name, field_names in config.fundamental_feature_groups.items():
        frame = prior_frames.get(group_name, pd.DataFrame())
        for field_name in field_names:
            feature_key = _sanitize_feature_name(field_name)
            series = frame[field_name] if not frame.empty and field_name in frame.columns else pd.Series(dtype=float)
            latest = _series_latest(series, 0)
            prev1 = _series_latest(series, 1)
            prev4 = _series_latest(series, 4)
            features[f"fund_{group_name}_{feature_key}_latest"] = latest
            features[f"fund_{group_name}_{feature_key}_delta1"] = latest - prev1 if pd.notna(latest) and pd.notna(prev1) else np.nan
            features[f"fund_{group_name}_{feature_key}_delta4"] = latest - prev4 if pd.notna(latest) and pd.notna(prev4) else np.nan
            features[f"fund_{group_name}_{feature_key}_missing"] = 0.0 if pd.notna(latest) else 1.0
            if field_name in FLOW_FIELD_GROUPS.get(group_name, set()):
                clean = pd.to_numeric(series, errors="coerce").dropna().tolist()
                features[f"fund_{group_name}_{feature_key}_ttm"] = float(np.sum(clean[-4:])) if clean else np.nan

    for ratio_name, num_group, num_field, den_group, den_field in RATIO_SPECS:
        numerator_frame = prior_frames.get(num_group, pd.DataFrame())
        denominator_frame = prior_frames.get(den_group, pd.DataFrame())
        numerator = _series_latest(numerator_frame[num_field], 0) if not numerator_frame.empty and num_field in numerator_frame.columns else np.nan
        denominator = _series_latest(denominator_frame[den_field], 0) if not denominator_frame.empty and den_field in denominator_frame.columns else np.nan
        ratio = _safe_divide(numerator, denominator)
        features[f"ratio_{ratio_name}"] = ratio
        features[f"ratio_{ratio_name}_missing"] = 0.0 if pd.notna(ratio) else 1.0

    return features


def _split_name(published_date: pd.Timestamp, config: PrototypeConfig) -> str:
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)
    if published_date <= train_end:
        return "train"
    if published_date <= val_end:
        return "val"
    return "test"


def _merge_macro_features(
    ohlc_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    macro_feature_columns: list[str],
    macro_marker_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if macro_df.empty:
        return ohlc_df, list(BASE_SEQUENCE_FEATURE_COLUMNS)

    merged = pd.merge_ordered(ohlc_df.sort_values("Date"), macro_df, on="Date", how="left")
    merged[macro_feature_columns] = merged[macro_feature_columns].ffill()
    if macro_marker_columns:
        merged[macro_marker_columns] = merged[macro_marker_columns].ffill()
    derived_columns: dict[str, pd.Series] = {}
    for marker_column in macro_marker_columns:
        base_prefix = marker_column[: -len("_source_date")]
        available_column = f"{base_prefix}_available"
        staleness_column = f"{base_prefix}_staleness_days"
        marker_ts = pd.to_datetime(merged[marker_column], errors="coerce")
        derived_columns[available_column] = marker_ts.notna().astype(float)
        staleness = (pd.to_datetime(merged["Date"], errors="coerce") - marker_ts).dt.days.astype(float)
        staleness[marker_ts.isna()] = 9999.0
        derived_columns[staleness_column] = staleness
    if derived_columns:
        merged = pd.concat([merged, pd.DataFrame(derived_columns, index=merged.index)], axis=1)
    sequence_feature_columns = list(BASE_SEQUENCE_FEATURE_COLUMNS) + list(macro_feature_columns) + list(derived_columns.keys())
    return merged, sequence_feature_columns


def _preload_ticker_state(
    helper: KairosDatabaseHelper,
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[str]]]:
    eps_by_ticker: dict[str, pd.DataFrame] = {}
    ticker_info: dict[str, dict[str, Any]] = {}
    eps_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sector_to_tickers: dict[str, list[str]] = {}
    for ticker in tickers:
        info = helper.get_ticker_info(ticker)
        sector = str(info.get("Sector") or info.get("SectorDisp") or "UNKNOWN").strip() or "UNKNOWN"
        info = {
            "sector": sector,
            "sector_key": str(info.get("SectorKey") or sector),
            "industry": str(info.get("Industry") or info.get("IndustryDisp") or "UNKNOWN").strip() or "UNKNOWN",
            "industry_key": str(info.get("IndustryKey") or info.get("Industry") or "UNKNOWN"),
        }
        ticker_info[ticker] = info
        sector_to_tickers.setdefault(info["sector"], []).append(ticker)
        eps_df = helper.get_eps_dataframe(ticker, period_types=("3M",), eps_columns=("BasicEPS",))
        eps_df = eps_df[(eps_df["periodType"] == "3M") & eps_df["BasicEPS"].notna()].copy()
        eps_df = eps_df.sort_values("publishedDate").reset_index(drop=True)
        eps_by_ticker[ticker] = eps_df
        eps_cache[ticker] = (
            eps_df["publishedDate"].to_numpy(dtype="datetime64[ns]"),
            pd.to_numeric(eps_df["BasicEPS"], errors="coerce").to_numpy(dtype=np.float32),
        )
    return eps_by_ticker, ticker_info, eps_cache, sector_to_tickers


def _apply_volatility_trim(
    helper: KairosDatabaseHelper,
    config: PrototypeConfig,
) -> tuple[list[str], dict[str, Any]]:
    requested_tickers = list(config.tickers)
    summary: dict[str, Any] = {
        "enabled": bool(config.volatility_trim_enabled),
        "requested_count": len(requested_tickers),
        "effective_count": len(requested_tickers),
        "trimmed_count": 0,
        "trim_fraction": float(config.volatility_trim_fraction),
        "trim_threshold": None,
        "trimmed_tickers": [],
        "kept_tickers": requested_tickers,
    }
    if not config.volatility_trim_enabled or config.volatility_trim_fraction <= 0.0 or len(requested_tickers) < 2:
        return requested_tickers, summary

    scores: list[tuple[str, float]] = []
    for ticker in requested_tickers:
        ohlc_df = helper.get_ohlc_dataframe(ticker)
        score = _realized_ticker_volatility(ohlc_df, config.volatility_trim_min_history)
        if np.isfinite(score):
            scores.append((ticker, float(score)))

    if len(scores) < 2:
        summary["note"] = "insufficient_valid_volatility_scores"
        return requested_tickers, summary

    trim_count = min(len(scores) - 1, int(np.ceil(len(scores) * float(config.volatility_trim_fraction))))
    if trim_count <= 0:
        return requested_tickers, summary

    scores_sorted = sorted(scores, key=lambda item: item[1], reverse=True)
    trimmed = {ticker for ticker, _ in scores_sorted[:trim_count]}
    kept = [ticker for ticker in requested_tickers if ticker not in trimmed]
    threshold = scores_sorted[trim_count - 1][1]
    summary.update(
        {
            "effective_count": len(kept),
            "trimmed_count": len(trimmed),
            "trim_threshold": float(threshold),
            "trimmed_tickers": [ticker for ticker, _ in scores_sorted[:trim_count]],
            "kept_tickers": kept,
            "volatility_scores_desc": [
                {"ticker": ticker, "realized_volatility": float(score)} for ticker, score in scores_sorted
            ],
        }
    )
    return kept, summary


def _sector_peer_median(
    ticker: str,
    sector: str,
    published_date: pd.Timestamp,
    sector_to_tickers: dict[str, list[str]],
    eps_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    peer_values: list[float] = []
    target_ts = np.datetime64(published_date.to_datetime64())
    for peer in sector_to_tickers.get(sector, []):
        if peer == ticker:
            continue
        dates, eps_values = eps_cache.get(peer, (np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.float32)))
        if len(dates) == 0:
            continue
        idx = np.searchsorted(dates, target_ts, side="left") - 1
        if idx < 0:
            continue
        value = float(eps_values[idx])
        if np.isfinite(value):
            peer_values.append(value)
    return float(np.median(peer_values)) if peer_values else np.nan


def _compute_baselines(
    ticker: str,
    target_row: pd.Series,
    prior_eps: pd.DataFrame,
    sector: str,
    sector_to_tickers: dict[str, list[str]],
    eps_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    history = prior_eps.sort_values("publishedDate").copy()
    eps_history = pd.to_numeric(history["BasicEPS"], errors="coerce").dropna().tolist()
    persistence = float(eps_history[-1]) if eps_history else np.nan
    trailing_mean = float(np.mean(eps_history[-4:])) if eps_history else np.nan
    trend = np.nan
    if len(eps_history) >= 3:
        deltas = np.diff(np.array(eps_history[-3:], dtype=float))
        trend = float(eps_history[-1] + np.mean(deltas))
    seasonal = np.nan
    if not history.empty:
        target_as_of = pd.Timestamp(target_row["asOfDate"])
        target_year = int(target_as_of.year) - 1
        target_quarter = int(target_as_of.quarter)
        seasonal_candidates = history[
            (history["asOfDate"].dt.year == target_year) & (history["asOfDate"].dt.quarter == target_quarter)
        ]
        if seasonal_candidates.empty and len(eps_history) >= 4:
            seasonal = float(eps_history[-4])
        elif not seasonal_candidates.empty:
            seasonal = float(pd.to_numeric(seasonal_candidates["BasicEPS"], errors="coerce").dropna().iloc[-1])
    sector_peer = _sector_peer_median(
        ticker,
        sector,
        pd.Timestamp(target_row["publishedDate"]),
        sector_to_tickers,
        eps_cache,
    )
    return {
        "baseline_persistence": persistence,
        "baseline_seasonal_naive": seasonal,
        "baseline_trailing_mean": trailing_mean,
        "baseline_trend": trend,
        "baseline_sector_peer_median": sector_peer,
    }


def _assign_sector_buckets(metadata: pd.DataFrame, config: PrototypeConfig) -> tuple[pd.Series, dict[str, Any]]:
    if metadata.empty:
        return pd.Series(dtype=str), {}
    if config.sector_modeling_mode != "per_sector":
        sector_bucket = pd.Series(["GLOBAL"] * len(metadata), index=metadata.index, dtype=object)
        return sector_bucket, {"GLOBAL": metadata["split"].value_counts().to_dict()}

    train_counts = metadata[metadata["split"] == "train"].groupby("sector").size().to_dict()
    fallback = config.sector_fallback_bucket
    eligible = {sector for sector, count in train_counts.items() if count >= config.sector_min_train_samples}
    sector_bucket = metadata["sector"].apply(lambda sector: sector if sector in eligible else fallback)
    summary: dict[str, Any] = {}
    for bucket, frame in metadata.assign(sector_bucket=sector_bucket).groupby("sector_bucket"):
        summary[str(bucket)] = {
            "num_tickers": int(frame["ticker"].nunique()),
            "tickers": sorted(frame["ticker"].unique().tolist()),
            "split_counts": {key: int(val) for key, val in frame["split"].value_counts().to_dict().items()},
        }
    return sector_bucket, summary


def build_event_dataset(
    helper: KairosDatabaseHelper,
    config: PrototypeConfig,
    env_file: str | Path = ".env",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    macro_df, macro_feature_columns, macro_marker_columns = _build_macro_feature_bundle(config, env_file)
    static_feature_columns = _all_static_feature_columns(config)
    effective_tickers, volatility_trim_summary = _apply_volatility_trim(helper, config)
    eps_by_ticker, ticker_info, eps_cache, sector_to_tickers = _preload_ticker_state(helper, effective_tickers)
    ticker_id_map = {ticker: idx for idx, ticker in enumerate(effective_tickers)}

    metadata_rows: list[dict[str, Any]] = []
    sequence_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    targets: list[float] = []
    last_sequence_feature_columns = list(BASE_SEQUENCE_FEATURE_COLUMNS)

    for ticker in effective_tickers:
        eps_df = eps_by_ticker.get(ticker, pd.DataFrame())
        if eps_df.empty:
            continue
        ohlc_df = helper.get_ohlc_dataframe(ticker)
        if ohlc_df.empty:
            continue
        ohlc_df = _compute_market_features(ohlc_df)
        ohlc_df, sequence_feature_columns = _merge_macro_features(
            ohlc_df,
            macro_df,
            macro_feature_columns,
            macro_marker_columns,
        )
        last_sequence_feature_columns = sequence_feature_columns
        fundamental_frames = _prepare_fundamental_frames(helper, ticker, config)
        info = ticker_info[ticker]

        for idx, row in eps_df.iterrows():
            published_date = pd.Timestamp(row["publishedDate"])
            market_history = ohlc_df[ohlc_df["Date"] < published_date].copy()
            if len(market_history) < config.seq_len:
                continue
            market_window = market_history.tail(config.seq_len).copy()
            if market_window[BASE_SEQUENCE_FEATURE_COLUMNS].isna().any().any():
                continue
            prior_eps = eps_df.iloc[:idx].copy()
            eps_features = _compute_eps_static_features(prior_eps, row)
            fund_features = _build_fundamental_static_features(fundamental_frames, published_date, config)
            baselines = _compute_baselines(
                ticker,
                row,
                prior_eps,
                info["sector"],
                sector_to_tickers,
                eps_cache,
            )
            feature_payload = {**eps_features, **fund_features}
            static_values = np.array([feature_payload.get(column, np.nan) for column in static_feature_columns], dtype=np.float32)
            sequence_rows.append(market_window[sequence_feature_columns].to_numpy(dtype=np.float32))
            static_rows.append(static_values)
            targets.append(float(row["BasicEPS"]))
            metadata_rows.append(
                {
                    "sample_id": len(metadata_rows),
                    "ticker": ticker,
                    "ticker_id": ticker_id_map[ticker],
                    "sector": info["sector"],
                    "sector_key": info["sector_key"],
                    "industry": info["industry"],
                    "industry_key": info["industry_key"],
                    "target_as_of_date": pd.Timestamp(row["asOfDate"]).isoformat(),
                    "target_published_date": published_date.isoformat(),
                    "split": _split_name(published_date, config),
                    "target_basic_eps": float(row["BasicEPS"]),
                    "last_observed_market_date": pd.Timestamp(market_window["Date"].iloc[-1]).isoformat(),
                    "last_prior_eps": eps_features["prev_eps_1"],
                    **baselines,
                }
            )

    metadata = pd.DataFrame(metadata_rows)
    if metadata.empty:
        raise RuntimeError("No samples were generated. Check ticker coverage and sequence length.")

    sector_bucket, sector_summary = _assign_sector_buckets(metadata, config)
    metadata["sector_bucket"] = sector_bucket
    sector_id_map = {sector: idx for idx, sector in enumerate(sorted(metadata["sector_bucket"].unique()))}
    industry_id_map = {industry: idx for idx, industry in enumerate(sorted(metadata["industry"].unique()))}
    metadata["sector_id"] = metadata["sector_bucket"].map(sector_id_map).astype(int)
    metadata["industry_id"] = metadata["industry"].map(industry_id_map).astype(int)

    sequences = np.stack(sequence_rows).astype(np.float32)
    static = np.stack(static_rows).astype(np.float32)
    target_array = np.array(targets, dtype=np.float32)
    metadata.attrs["sequence_feature_columns"] = last_sequence_feature_columns
    metadata.attrs["static_feature_columns"] = static_feature_columns
    metadata.attrs["sector_summary"] = sector_summary
    metadata.attrs["volatility_trim_summary"] = volatility_trim_summary
    return metadata, sequences, static, target_array, sector_summary


def _fit_normalization(
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    static: np.ndarray,
    sequence_feature_columns: list[str],
    static_feature_columns: list[str],
) -> dict[str, list[float]]:
    train_idx = metadata.loc[metadata["split"] == "train", "sample_id"].to_numpy(dtype=np.int64)
    if len(train_idx) == 0:
        raise RuntimeError("No train samples found for normalization.")
    seq_train = sequences[train_idx]
    static_train = static[train_idx]
    seq_mean = np.nanmean(seq_train, axis=(0, 1))
    seq_std = np.nanstd(seq_train, axis=(0, 1))
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)
    valid_static = np.isfinite(static_train).any(axis=0)
    static_train_safe = static_train.copy()
    static_train_safe[:, ~valid_static] = 0.0
    static_mean = np.nanmean(static_train_safe, axis=0)
    static_std = np.nanstd(static_train_safe, axis=0)
    static_mean = np.nan_to_num(static_mean, nan=0.0, posinf=0.0, neginf=0.0)
    static_std = np.nan_to_num(static_std, nan=1.0, posinf=1.0, neginf=1.0)
    static_mean[~valid_static] = 0.0
    static_std[~valid_static] = 1.0
    static_std = np.where(static_std < 1e-6, 1.0, static_std)
    return {
        "sequence_feature_columns": sequence_feature_columns,
        "static_feature_columns": static_feature_columns,
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
    sector_summary: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sequence_feature_columns = metadata.attrs.get("sequence_feature_columns", BASE_SEQUENCE_FEATURE_COLUMNS)
    static_feature_columns = metadata.attrs.get("static_feature_columns", BASE_STATIC_FEATURE_COLUMNS)
    normalization = _fit_normalization(
        metadata,
        sequences,
        static,
        sequence_feature_columns,
        static_feature_columns,
    )
    metadata.to_csv(output_path / "event_metadata.csv", index=False)
    np.savez_compressed(output_path / "dataset_arrays.npz", sequences=sequences, static=static, targets=targets)
    (output_path / "normalization.json").write_text(json.dumps(normalization, indent=2))
    (output_path / "sector_summary.json").write_text(json.dumps(sector_summary, indent=2))
    (output_path / "volatility_trim_summary.json").write_text(
        json.dumps(metadata.attrs.get("volatility_trim_summary", {}), indent=2)
    )
    (output_path / "config.json").write_text(
        json.dumps(
            {
                "tickers": config.tickers,
                "seq_len": config.seq_len,
                "train_end": config.train_end,
                "val_end": config.val_end,
                "sector_modeling_mode": config.sector_modeling_mode,
                "sector_min_train_samples": config.sector_min_train_samples,
                "sector_fallback_bucket": config.sector_fallback_bucket,
                "volatility_trim_enabled": config.volatility_trim_enabled,
                "volatility_trim_fraction": config.volatility_trim_fraction,
                "volatility_trim_min_history": config.volatility_trim_min_history,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a leak-free EPS event dataset.")
    parser.add_argument("--config", default="configs/prototype_config.json", help="Path to config JSON.")
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
    metadata, sequences, static, targets, sector_summary = build_event_dataset(helper, config, env_file=args.env_file)
    save_dataset(args.output_dir, config, metadata, sequences, static, targets, sector_summary)
    split_counts = metadata["split"].value_counts().to_dict()
    print(
        json.dumps(
            {
                "num_samples": int(len(metadata)),
                "split_counts": split_counts,
                "num_sector_buckets": int(metadata["sector_bucket"].nunique()),
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
