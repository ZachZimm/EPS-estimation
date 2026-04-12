from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line or line.startswith("export "):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _utc_naive(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce", utc=True)
    return values.dt.tz_convert("UTC").dt.tz_localize(None)


@dataclass
class ExternalSeriesCache:
    env_file: str | os.PathLike[str] = ".env"
    cache_dir: str | os.PathLike[str] = "artifacts/external_cache"

    def __post_init__(self) -> None:
        env_values = _parse_env_file(Path(self.env_file))
        self._env = {**env_values, **os.environ}
        self.cache_dir = str(Path(self.cache_dir))
        self._fred_api_key = self._env.get("FRED_API_KEY")
        self._fred: Fred | None = None

    @property
    def _cache_root(self) -> Path:
        root = Path(self.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _get_fred_client(self) -> Fred:
        if not self._fred_api_key:
            raise RuntimeError("FRED_API_KEY is not configured.")
        if self._fred is None:
            self._fred = Fred(api_key=self._fred_api_key)
        return self._fred

    def get_fred_series(self, series_id: str, refresh: bool = False) -> pd.DataFrame:
        cache_path = self._cache_root / "fred" / f"{_sanitize_filename(series_id)}.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not refresh:
            frame = pd.read_csv(cache_path)
        else:
            fred = self._get_fred_client()
            try:
                frame = fred.get_series_all_releases(series_id).dropna().reset_index()
                if "index" in frame.columns:
                    frame = frame.drop(columns=["index"])
                keep_columns = [
                    column for column in ["date", "realtime_start", "value"] if column in frame.columns
                ]
                frame = frame[keep_columns].copy()
                if keep_columns != ["date", "realtime_start", "value"]:
                    raise RuntimeError(
                        f"Unexpected FRED payload columns for {series_id!r}: {frame.columns.tolist()}"
                    )
            except Exception:
                series = fred.get_series(series_id).dropna()
                frame = pd.DataFrame({"date": series.index, "value": series.values})
                frame["realtime_start"] = frame["date"]
            frame["series_id"] = series_id
            frame.to_csv(cache_path, index=False)

        frame["date"] = _utc_naive(frame["date"])
        frame["realtime_start"] = _utc_naive(frame["realtime_start"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["series_id"] = series_id
        frame = frame.dropna(subset=["date", "realtime_start", "value"]).sort_values(
            ["realtime_start", "date"]
        )
        return frame.reset_index(drop=True)

    def get_market_series(self, symbol: str, refresh: bool = False) -> pd.DataFrame:
        cache_path = self._cache_root / "yahoo" / f"{_sanitize_filename(symbol)}.csv"
        meta_path = self._cache_root / "yahoo" / f"{_sanitize_filename(symbol)}.meta.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not refresh:
            frame = pd.read_csv(cache_path)
        else:
            history = yf.Ticker(symbol).history(period="max", auto_adjust=False)
            if history.empty:
                raise RuntimeError(f"No market history returned for symbol {symbol!r}")
            history = history.reset_index()
            if "Date" not in history.columns and "Datetime" in history.columns:
                history = history.rename(columns={"Datetime": "Date"})
            frame = history
            frame.to_csv(cache_path, index=False)
            meta_path.write_text(json.dumps({"symbol": symbol}, indent=2))

        frame["Date"] = _utc_naive(frame["Date"])
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["symbol"] = symbol
        frame = frame.dropna(subset=["Date"]).sort_values("Date")
        return frame.reset_index(drop=True)
