"""Fetch real SPY option data from Yahoo Finance with rate limiting.

Usage:
    python scripts/fetch_real_data.py --ticker SPY --output data/spy_real_options.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import structlog

from volsurf.data.yahoo import YahooDataSource
from volsurf.utils.logging import configure_logging

configure_logging("INFO")
logger = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (SPY, SPX, QQQ, etc.)")
    parser.add_argument("--delay", type=float, default=2.5, help="Seconds between Yahoo requests")
    parser.add_argument("--output", type=str, default="data/spy_real_options.csv", help="Output CSV path")
    args = parser.parse_args()

    logger.info("Starting real data fetch", ticker=args.ticker, delay=args.delay)

    source = YahooDataSource(delay_seconds=args.delay)
    chains = source.load_chain(args.ticker)

    if not chains:
        raise ValueError(f"No chains returned for {args.ticker}")

    # Flatten all chains into a single DataFrame
    rows: list[dict] = []
    for chain in chains:
        for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
            for _, row in df.iterrows():
                rows.append({
                    "underlying": chain.underlying,
                    "spot": chain.spot,
                    "observation_date": chain.observation_date.isoformat(),
                    "expiration_date": chain.expiration_date.isoformat(),
                    "days_to_expiry": chain.days_to_expiry,
                    "T": round(chain.days_to_expiry / 365.0, 6),
                    "option_type": opt_type,
                    "strike": row["strike"],
                    "moneyness": round(row["strike"] / chain.spot, 4),
                    "bid": row["bid"],
                    "ask": row["ask"],
                    "lastPrice": row.get("lastPrice", row["midPrice"]),
                    "midPrice": row["midPrice"],
                    "impliedVolatility": row.get("impliedVolatility", None),
                    "volume": row.get("volume", 0),
                    "openInterest": row.get("openInterest", 0),
                })

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    logger.info(
        "Fetch complete",
        ticker=args.ticker,
        total_rows=len(df),
        expirations=df["expiration_date"].nunique(),
        iv_available=df["impliedVolatility"].notna().sum(),
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"  REAL OPTION DATA: {args.ticker}")
    print("=" * 60)
    print(f"  Spot price:       {chain.spot:.2f}")
    print(f"  Total quotes:     {len(df)}")
    print(f"  Calls:            {len(df[df['option_type']=='call'])}")
    print(f"  Puts:             {len(df[df['option_type']=='put'])}")
    print(f"  Expirations:      {df['expiration_date'].nunique()}")
    print(f"  DTE range:        {df['days_to_expiry'].min()} - {df['days_to_expiry'].max()}")
    if df["impliedVolatility"].notna().any():
        print(f"  IV range:         {df['impliedVolatility'].min():.2%} - {df['impliedVolatility'].max():.2%}")
    print(f"  Saved to:         {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()