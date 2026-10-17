"""Serialization and I/O utilities for models, FAISS indices, datasets, and reports."""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(data: Any, path: Path) -> None:
    """Saves data to a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Loads data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    """Serializes an object to pickle format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Loads a pickled object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Saves DataFrame as parquet if available, otherwise compressed csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv.gz"), index=False, compression="gzip")


def load_dataframe(path: Path) -> pd.DataFrame:
    """Loads DataFrame from parquet or csv."""
    path = Path(path)
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_gz_path = path.with_suffix(".csv.gz")
    if csv_gz_path.exists():
        return pd.read_csv(csv_gz_path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No dataframe found matching base path: {path}")
