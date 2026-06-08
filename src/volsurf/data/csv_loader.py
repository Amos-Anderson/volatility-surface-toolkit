"""CSV option data loader for preprocessed datasets."""
from datetime import date
from pathlib import Path

import pandas as pd

from volsurf.data.base import DataSource, OptionChain


class CSVDataSource(DataSource):
    def __init__(self, csv_path: str | Path, spot: float, observation_date: date | None = None):
        self.csv_path = Path(csv_path)
        self.spot = spot
        self.observation_date = observation_date or date.today()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

    def get_spot(self, ticker=None, observation_date=None):
        return self.spot

    def load_chain(self, ticker=None, observation_date=None) -> list[OptionChain]:
        df = pd.read_csv(self.csv_path)
        for col in ["expiration_date", "Maturity"]:
            if col in df.columns and df[col].dtype == object:
                df[col] = pd.to_datetime(df[col]).dt.date
        if "option_type" in df.columns:
            df["option_type"] = df["option_type"].str.lower()
        if "days_to_expiry" not in df.columns and "T" in df.columns:
            df["days_to_expiry"] = (df["T"] * 365).round().astype(int)
        chains = []
        for exp_date, group in df.groupby("expiration_date"):
            dte = int(group["days_to_expiry"].iloc[0])
            if dte <= 0:
                continue
            calls = group[group["option_type"] == "call"].copy()
            puts = group[group["option_type"] == "put"].copy()
            chains.append(OptionChain(
                underlying=ticker or "GLD", spot=self.spot,
                observation_date=self.observation_date, expiration_date=exp_date,
                days_to_expiry=dte, calls=calls, puts=puts,
            ))
        return chains