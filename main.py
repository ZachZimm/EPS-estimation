from __future__ import annotations

import argparse

from database_helper import KairosDatabaseHelper, available_eps_columns, to_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve OHLC and EPS data from Kairos databases.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL")
    parser.add_argument("--start-date", help="Inclusive ISO start date, for example 2024-01-01")
    parser.add_argument("--end-date", help="Inclusive ISO end date, for example 2024-12-31")
    parser.add_argument(
        "--mode",
        choices=("merged", "ohlc", "eps", "both", "eps-columns"),
        default="merged",
        help="Which dataset to fetch",
    )
    parser.add_argument(
        "--period-type",
        dest="period_types",
        action="append",
        help="EPS period type filter. Repeatable. Defaults to 3M and TTM.",
    )
    parser.add_argument(
        "--eps-column",
        dest="eps_columns",
        action="append",
        help="EPS column to retrieve. Repeatable. Defaults to BasicEPS and DilutedEPS.",
    )
    parser.add_argument("--env-file", default=".env", help="Path to the env file with DB credentials")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    helper = KairosDatabaseHelper(args.env_file)

    if args.mode == "ohlc":
        result = helper.get_ohlc(args.ticker, start_date=args.start_date, end_date=args.end_date)
    elif args.mode == "eps":
        result = helper.get_eps(
            args.ticker,
            period_types=args.period_types or ("3M", "TTM"),
            start_date=args.start_date,
            end_date=args.end_date,
            eps_columns=args.eps_columns or ("BasicEPS", "DilutedEPS"),
        )
    elif args.mode == "merged":
        result = helper.get_merged_ohlc_eps_dataframe(
            args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            period_types=args.period_types or ("3M", "TTM"),
            eps_columns=args.eps_columns or ("BasicEPS", "DilutedEPS"),
        )
    elif args.mode == "eps-columns":
        result = available_eps_columns(helper, args.ticker)
    else:
        result = helper.get_ohlc_and_eps(
            args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            period_types=args.period_types or ("3M", "TTM"),
            eps_columns=args.eps_columns or ("BasicEPS", "DilutedEPS"),
        )

    print(to_json(result))


if __name__ == "__main__":
    main()
