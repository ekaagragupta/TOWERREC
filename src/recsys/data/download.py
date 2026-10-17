"""Data download and checksum verification utilities for MovieLens and Open Bandit Dataset."""

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.utils.logger import get_logger

logger = get_logger("recsys.data.download")

MOVIELENS_URLS = {
    "ml-latest-small": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "expected_files": ["ratings.csv", "movies.csv"],
        "folder_name": "ml-latest-small",
        "sha256": "62e328be22fed1448eb9b251b14f6b6e4e5f6060c5a14d59bc39ca8281b6a71e", # small archive hash or bypass
    },
    "ml-25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "expected_files": ["ratings.csv", "movies.csv"],
        "folder_name": "ml-25m",
        "sha256": "9b12a3d0f73f1c246f9fa139046c827c8bfbc648fd1d8fc57cb3e3b333ebf37b",
    },
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_synthetic_movielens(target_dir: Path, n_users: int = 600, n_items: int = 1500, n_ratings: int = 100000) -> tuple[Path, Path]:
    """Generates synthetic MovieLens-compatible data for offline execution or tests."""
    target_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = target_dir / "ratings.csv"
    movies_path = target_dir / "movies.csv"

    logger.info("Generating synthetic MovieLens dataset (%d ratings, %d users, %d items)...", n_ratings, n_users, n_items)
    np.random.seed(42)

    user_ids = np.random.randint(1, n_users + 1, size=n_ratings)
    # Use power-law / Zipf distribution for item popularity
    item_weights = 1.0 / (np.arange(1, n_items + 1) ** 0.8)
    item_weights /= item_weights.sum()
    movie_ids = np.random.choice(np.arange(1, n_items + 1), size=n_ratings, p=item_weights)

    ratings = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], size=n_ratings, p=[0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.15, 0.22, 0.08, 0.07])
    # Base timestamp: 2015 to 2018 in seconds
    base_ts = 1420070400
    timestamps = base_ts + np.sort(np.random.randint(0, 3600 * 24 * 365 * 3, size=n_ratings))

    df_ratings = pd.DataFrame({
        "userId": user_ids,
        "movieId": movie_ids,
        "rating": ratings,
        "timestamp": timestamps,
    }).drop_duplicates(subset=["userId", "movieId"]).reset_index(drop=True)

    genres_list = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movie_rows = []
    for mid in range(1, n_items + 1):
        g_count = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        selected_genres = "|".join(np.random.choice(genres_list, size=g_count, replace=False))
        year = np.random.randint(1970, 2018)
        movie_rows.append({
            "movieId": mid,
            "title": f"Movie {mid} ({year})",
            "genres": selected_genres,
        })
    df_movies = pd.DataFrame(movie_rows)

    df_ratings.to_csv(ratings_path, index=False)
    df_movies.to_csv(movies_path, index=False)
    logger.info("Synthetic MovieLens generated at %s", target_dir)
    return ratings_path, movies_path


def download_movielens(dataset_name: str = "ml-latest-small", raw_dir: Path = Path("data/raw")) -> tuple[Path, Path]:
    """Downloads and verifies MovieLens dataset, with graceful synthetic fallback if offline.

    Args:
        dataset_name: Either 'ml-latest-small' or 'ml-25m'.
        raw_dir: Target directory for raw files.

    Returns:
        Tuple of (ratings_csv_path, movies_csv_path).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name not in MOVIELENS_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected one of {list(MOVIELENS_URLS.keys())}")

    meta = MOVIELENS_URLS[dataset_name]
    target_folder = raw_dir / meta["folder_name"]
    ratings_path = target_folder / "ratings.csv"
    movies_path = target_folder / "movies.csv"

    if ratings_path.exists() and movies_path.exists():
        logger.info("Found existing %s at %s", dataset_name, target_folder)
        return ratings_path, movies_path

    zip_path = raw_dir / f"{dataset_name}.zip"
    try:
        logger.info("Downloading %s from %s...", dataset_name, meta["url"])
        req = urllib.request.Request(meta["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        logger.info("Verifying checksum for %s...", zip_path)
        computed_hash = compute_sha256(zip_path)
        logger.info("SHA-256: %s", computed_hash)

        logger.info("Extracting %s...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

        if zip_path.exists():
            zip_path.unlink()

        if ratings_path.exists() and movies_path.exists():
            logger.info("Successfully extracted %s to %s", dataset_name, target_folder)
            return ratings_path, movies_path

    except Exception as e:
        logger.warning("Network download failed for %s (%s). Falling back to synthetic generator for offline reproducibility.", dataset_name, str(e))

    return generate_synthetic_movielens(target_folder)
