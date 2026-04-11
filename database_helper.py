from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

import pandas as pd


_IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")
_DIRECT_DSN_KEYS = ("DATABASE_URL", "POSTGRES_DSN", "POSTGRES_URL")
_DEFAULT_CIK_MAP_PATH = Path("/home/zach/projects/kairos/backend/data/company_tickers.json")


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values


def _coalesce(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _quote_identifier(identifier: str) -> str:
    if not _IDENT_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalize_date(value: dt.date | dt.datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()

    text = value.strip()
    if not text:
        return None
    if "T" in text:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    return dt.date.fromisoformat(text).isoformat()


def _maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        as_float = _maybe_float(text)
        return None if as_float is None else int(as_float)


def _parse_timestamptz(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return dt.datetime.fromisoformat(text.replace(" ", "T").replace("+00", "+00:00"))


def _parse_date(value: str | None) -> dt.date | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return dt.date.fromisoformat(text)


def _normalize_to_naive_date(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _normalize_publication_map(
    publication_date_map: dict[pd.Timestamp, pd.Timestamp],
) -> dict[pd.Timestamp, pd.Timestamp]:
    normalized_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for period_end, filed_date in publication_date_map.items():
        normalized_period_end = _normalize_to_naive_date(period_end)
        normalized_filed_date = _normalize_to_naive_date(filed_date)
        if normalized_period_end is not None and normalized_filed_date is not None:
            normalized_map[normalized_period_end] = normalized_filed_date
    return normalized_map


def _lookup_publication_date(
    as_of_date: pd.Timestamp | str | None,
    publication_date_map: dict[pd.Timestamp, pd.Timestamp],
    loose_match_days: int = 50,
    fallback_days: int = 50,
) -> pd.Timestamp | None:
    normalized_as_of = _normalize_to_naive_date(as_of_date)
    if normalized_as_of is None:
        return None

    exact = publication_date_map.get(normalized_as_of)
    if exact is not None:
        return exact

    nearest_filed_date = None
    nearest_delta_days = None
    for period_end, filed_date in publication_date_map.items():
        delta_days = abs((period_end - normalized_as_of).days)
        if nearest_delta_days is None or delta_days < nearest_delta_days:
            nearest_delta_days = delta_days
            nearest_filed_date = filed_date

    if nearest_delta_days is not None and nearest_delta_days <= loose_match_days:
        return nearest_filed_date

    return normalized_as_of + pd.Timedelta(days=fallback_days)


def _build_published_date_series(
    as_of_series: pd.Series,
    publication_date_map: dict[pd.Timestamp, pd.Timestamp],
    loose_match_days: int = 50,
    fallback_days: int = 50,
) -> pd.Series:
    normalized_map = _normalize_publication_map(publication_date_map)
    published_dates = as_of_series.apply(
        lambda value: _lookup_publication_date(
            value,
            normalized_map,
            loose_match_days=loose_match_days,
            fallback_days=fallback_days,
        )
    )
    published_dates = pd.to_datetime(published_dates, errors="coerce", utc=True)
    return published_dates.dt.tz_convert("UTC").dt.tz_localize(None)


@dataclass(frozen=True)
class DbTarget:
    host: str
    port: str
    user: str
    password: str
    database: str

    @classmethod
    def from_dsn(cls, dsn: str) -> "DbTarget":
        parsed = urlparse(dsn)
        if parsed.scheme not in {"postgresql", "postgres"}:
            raise ValueError(f"Unsupported DSN scheme: {parsed.scheme!r}")
        if not parsed.hostname or not parsed.username or not parsed.path:
            raise ValueError("DSN must include hostname, username, and database name")
        return cls(
            host=parsed.hostname,
            port=str(parsed.port or 5432),
            user=unquote(parsed.username),
            password=unquote(parsed.password or ""),
            database=parsed.path.lstrip("/"),
        )


class KairosDatabaseHelper:
    def __init__(self, env_path: str | os.PathLike[str] = ".env") -> None:
        env_values = _parse_env_file(Path(env_path))
        self._env = {**env_values, **os.environ}
        self._price_db = self._resolve_target(fund=False)
        self._fund_db = self._resolve_target(fund=True)
        cik_map_path = self._env.get("KAIROS_COMPANY_TICKERS_PATH")
        self._cik_map_path = Path(cik_map_path).expanduser() if cik_map_path else _DEFAULT_CIK_MAP_PATH
        self._ticker_cik_map: dict[str, int] | None = None

    def _resolve_target(self, *, fund: bool) -> DbTarget:
        if fund:
            fund_direct = self._env.get("FUND_DATABASE_URL")
            if fund_direct:
                return DbTarget.from_dsn(fund_direct)

        for key in _DIRECT_DSN_KEYS:
            direct = self._env.get(key)
            if direct:
                return DbTarget.from_dsn(direct)

        host = _coalesce(self._env.get("DATABASE_HOST"), self._env.get("POSTGRES_HOST"), "localhost")
        port = _coalesce(self._env.get("DATABASE_PORT"), self._env.get("POSTGRES_PORT"), "5432")
        user = _coalesce(self._env.get("DATABASE_USER"), self._env.get("POSTGRES_USER"))
        password = _coalesce(self._env.get("DATABASE_PASSWORD"), self._env.get("POSTGRES_PASSWORD"), "")
        database = (
            _coalesce(self._env.get("FUND_DATABASE_NAME"), "financials")
            if fund
            else _coalesce(self._env.get("DATABASE_NAME"), self._env.get("POSTGRES_DB"), "postgres")
        )

        if not user:
            raise RuntimeError("Database user not configured in environment")

        return DbTarget(
            host=host or "localhost",
            port=port or "5432",
            user=user,
            password=password or "",
            database=database or "postgres",
        )

    def _run_csv_query(self, target: DbTarget, sql: str) -> list[dict[str, str]]:
        copy_sql = f"\\copy ({sql}) TO STDOUT WITH CSV HEADER"
        proc = subprocess.run(
            [
                "psql",
                "-h",
                target.host,
                "-p",
                target.port,
                "-U",
                target.user,
                "-d",
                target.database,
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                copy_sql,
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PGPASSWORD": target.password, "PGTZ": "UTC"},
        )
        return list(csv.DictReader(io.StringIO(proc.stdout)))

    def _load_ticker_cik_map(self) -> dict[str, int]:
        if self._ticker_cik_map is not None:
            return self._ticker_cik_map

        payload = json.loads(self._cik_map_path.read_text())
        ticker_cik_map: dict[str, int] = {}
        for row in payload.values():
            ticker = str(row.get("ticker", "")).upper().strip()
            cik = row.get("cik_str")
            if not ticker or cik is None:
                continue
            try:
                ticker_cik_map[ticker] = int(cik)
            except (TypeError, ValueError):
                continue

        self._ticker_cik_map = ticker_cik_map
        return ticker_cik_map

    def get_ticker_cik(self, ticker: str) -> int | None:
        return self._load_ticker_cik_map().get(ticker.upper())

    def get_report_publication_dates(
        self,
        ticker: str,
        interval: str = "q",
    ) -> dict[pd.Timestamp, pd.Timestamp]:
        cik = self.get_ticker_cik(ticker)
        if cik is None:
            return {}
        if interval not in {"q", "a"}:
            raise ValueError("interval must be 'q' or 'a'")

        forms = ("10-Q", "10-Q/A") if interval == "q" else ("10-K", "10-K/A")
        sql = f"""
            SELECT "period_end", MIN("filed_date") AS "filed_date"
            FROM "EDGAR_REPORT_PUBLICATION"
            WHERE "cik"::text = {_quote_literal(str(cik))}
              AND "form" IN ({", ".join(_quote_literal(form) for form in forms)})
            GROUP BY "period_end"
            ORDER BY "period_end" ASC
        """
        rows = self._run_csv_query(self._price_db, sql)
        publication_dates: dict[pd.Timestamp, pd.Timestamp] = {}
        for row in rows:
            period_end = row.get("period_end")
            filed_date = row.get("filed_date")
            if not period_end or not filed_date:
                continue
            publication_dates[pd.Timestamp(period_end)] = pd.Timestamp(filed_date)
        return publication_dates

    def get_ohlc(
        self,
        ticker: str,
        start_date: dt.date | dt.datetime | str | None = None,
        end_date: dt.date | dt.datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        table_ident = _quote_identifier(ticker.upper())
        start = _normalize_date(start_date)
        end = _normalize_date(end_date)

        where_parts: list[str] = []
        if start:
            where_parts.append(f'"Date" >= {_quote_literal(start)}')
        if end:
            where_parts.append(f'"Date" <= {_quote_literal(end)}')

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        sql = f"""
            SELECT
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Returns",
                "Dividends",
                "Stock_Splits"
            FROM {table_ident}
            {where_sql}
            ORDER BY "Date" ASC
        """
        rows = self._run_csv_query(self._price_db, sql)
        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "Date": _parse_timestamptz(row.get("Date")),
                    "Open": _maybe_float(row.get("Open")),
                    "High": _maybe_float(row.get("High")),
                    "Low": _maybe_float(row.get("Low")),
                    "Close": _maybe_float(row.get("Close")),
                    "Volume": _maybe_int(row.get("Volume")),
                    "Returns": _maybe_float(row.get("Returns")),
                    "Dividends": _maybe_float(row.get("Dividends")),
                    "Stock_Splits": _maybe_float(row.get("Stock_Splits")),
                }
            )
        return output

    def get_ohlc_dataframe(
        self,
        ticker: str,
        start_date: dt.date | dt.datetime | str | None = None,
        end_date: dt.date | dt.datetime | str | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(self.get_ohlc(ticker, start_date=start_date, end_date=end_date))
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Returns",
                    "Dividends",
                    "Stock_Splits",
                ]
            )
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        return df.sort_values("Date").reset_index(drop=True)

    def get_eps(
        self,
        ticker: str,
        period_types: Sequence[str] = ("3M", "TTM"),
        start_date: dt.date | dt.datetime | str | None = None,
        end_date: dt.date | dt.datetime | str | None = None,
        eps_columns: Sequence[str] = ("BasicEPS", "DilutedEPS"),
    ) -> list[dict[str, Any]]:
        valid_periods = [period.strip().upper() for period in period_types if period.strip()]
        if not valid_periods:
            raise ValueError("period_types must contain at least one period")

        selected_columns = ["asOfDate", "periodType", *eps_columns]
        quoted_columns = ", ".join(_quote_identifier(column) for column in selected_columns)
        table_ident = _quote_identifier(f"{ticker.upper()}_q_income")
        start = _normalize_date(start_date)
        end = _normalize_date(end_date)

        where_parts = [f'"periodType" IN ({", ".join(_quote_literal(p) for p in valid_periods)})']
        if start:
            where_parts.append(f'"asOfDate" >= {_quote_literal(start)}')
        if end:
            where_parts.append(f'"asOfDate" <= {_quote_literal(end)}')

        sql = f"""
            SELECT {quoted_columns}
            FROM {table_ident}
            WHERE {' AND '.join(where_parts)}
            ORDER BY "asOfDate" ASC, "periodType" ASC
        """
        rows = self._run_csv_query(self._fund_db, sql)
        output: list[dict[str, Any]] = []
        for row in rows:
            parsed: dict[str, Any] = {
                "asOfDate": _parse_date(row.get("asOfDate")),
                "periodType": row.get("periodType"),
            }
            for column in eps_columns:
                parsed[column] = _maybe_float(row.get(column))
            output.append(parsed)
        return output

    def get_eps_dataframe(
        self,
        ticker: str,
        period_types: Sequence[str] = ("3M", "TTM"),
        start_date: dt.date | dt.datetime | str | None = None,
        end_date: dt.date | dt.datetime | str | None = None,
        eps_columns: Sequence[str] = ("BasicEPS", "DilutedEPS"),
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            self.get_eps(
                ticker,
                period_types=period_types,
                start_date=None,
                end_date=None,
                eps_columns=eps_columns,
            )
        )
        if df.empty:
            return pd.DataFrame(columns=["asOfDate", "publishedDate", "periodType", *eps_columns])

        df["asOfDate"] = pd.to_datetime(df["asOfDate"], errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        publication_map = self.get_report_publication_dates(ticker=ticker, interval="q")
        df["publishedDate"] = _build_published_date_series(
            df["asOfDate"],
            publication_map,
            loose_match_days=50,
            fallback_days=50,
        )

        published_start = pd.Timestamp(start_date).normalize() if start_date is not None else None
        published_end = pd.Timestamp(end_date).normalize() if end_date is not None else None
        if published_start is not None:
            df = df[df["publishedDate"] >= published_start]
        if published_end is not None:
            df = df[df["publishedDate"] <= published_end]

        for column in eps_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        return df.sort_values(["publishedDate", "asOfDate", "periodType"]).reset_index(drop=True)

    def get_merged_ohlc_eps_dataframe(
        self,
        ticker: str,
        start_date: dt.date | dt.datetime | str | None = None,
        end_date: dt.date | dt.datetime | str | None = None,
        period_types: Sequence[str] = ("3M", "TTM"),
        eps_columns: Sequence[str] = ("BasicEPS", "DilutedEPS"),
    ) -> pd.DataFrame:
        ohlc_df = self.get_ohlc_dataframe(ticker, start_date=start_date, end_date=end_date)
        eps_df = self.get_eps_dataframe(
            ticker,
            period_types=period_types,
            start_date=start_date,
            end_date=end_date,
            eps_columns=eps_columns,
        )

        if ohlc_df.empty:
            return ohlc_df
        if eps_df.empty:
            return ohlc_df.copy()

        event_rows: list[dict[str, Any]] = []
        for row in eps_df.itertuples(index=False):
            event: dict[str, Any] = {"Date": row.publishedDate}
            period_type = str(row.periodType)
            event[f"EPS_asOfDate_{period_type}"] = row.asOfDate
            event[f"EPS_publishedDate_{period_type}"] = row.publishedDate
            for column in eps_columns:
                event[f"{column}_{period_type}"] = getattr(row, column)
            event_rows.append(event)

        eps_events_df = pd.DataFrame(event_rows)
        eps_events_df = (
            eps_events_df.groupby("Date", as_index=False)
            .last()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        merged_df = pd.merge_asof(
            ohlc_df.sort_values("Date"),
            eps_events_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        return merged_df.reset_index(drop=True)

    def get_ohlc_and_eps(
        self,
        ticker: str,
        start_date: dt.date | dt.datetime | str | None = None,
        end_date: dt.date | dt.datetime | str | None = None,
        period_types: Sequence[str] = ("3M", "TTM"),
        eps_columns: Sequence[str] = ("BasicEPS", "DilutedEPS"),
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "ohlc": self.get_ohlc(ticker, start_date=start_date, end_date=end_date),
            "eps": self.get_eps(
                ticker,
                period_types=period_types,
                start_date=start_date,
                end_date=end_date,
                eps_columns=eps_columns,
            ),
        }

    def get_merged_ohlc_eps(
        self,
        ticker: str,
        start_date: dt.date | dt.datetime | str | None = None,
        end_date: dt.date | dt.datetime | str | None = None,
        period_types: Sequence[str] = ("3M", "TTM"),
        eps_columns: Sequence[str] = ("BasicEPS", "DilutedEPS"),
    ) -> list[dict[str, Any]]:
        return self.get_merged_ohlc_eps_dataframe(
            ticker,
            start_date=start_date,
            end_date=end_date,
            period_types=period_types,
            eps_columns=eps_columns,
        ).to_dict(orient="records")


def available_eps_columns(helper: KairosDatabaseHelper, ticker: str) -> list[str]:
    sql = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = {_quote_literal(f"{ticker.upper()}_q_income")}
          AND (column_name ILIKE '%eps%' OR column_name = 'periodType' OR column_name = 'asOfDate')
        ORDER BY ordinal_position
    """
    rows = helper._run_csv_query(helper._fund_db, sql)
    return [row["column_name"] for row in rows]


def to_json(value: Any) -> str:
    def json_safe(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, float):
            return obj if pd.notna(obj) else None
        if isinstance(obj, (str, bool, int)):
            return obj
        if obj is pd.NaT:
            return None
        if isinstance(obj, pd.Timestamp):
            return None if pd.isna(obj) else obj.isoformat()
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return json_safe(obj.to_dict(orient="records"))
        if isinstance(obj, pd.Series):
            return json_safe(obj.tolist())
        if isinstance(obj, dict):
            return {str(key): json_safe(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(item) for item in obj]
        return obj

    return json.dumps(json_safe(value), indent=2)
