from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COLUMN = "customerID"


def download_data(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(DATA_URL, destination)
    return destination


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    if ID_COLUMN in cleaned:
        cleaned = cleaned.drop(columns=[ID_COLUMN])
    if "TotalCharges" in cleaned:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    return cleaned


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(download_data(path))
    target = raw[TARGET].map({"Yes": 1, "No": 0})
    return clean_frame(raw.drop(columns=[TARGET])), target
