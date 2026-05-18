"""
Récupère les features Spotify (genres, popularité, followers, image) pour
tous les artistes uniques du dataset SNEP et sauvegarde le résultat dans
`data/spotify_artist_features.csv`.

Usage:
    python scripts/fetch_spotify_features.py
    python scripts/fetch_spotify_features.py --limit 50   # test sur 50 artistes
    python scripts/fetch_spotify_features.py --force      # re-fetch tout (ignore le cache)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Permettre l'import de src.* depuis scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.spotify import fetch_artists_features, split_artists  # noqa: E402


SNEP_CSV = DATA_DIR / "snep_certifications.csv"
OUTPUT_CSV = DATA_DIR / "spotify_artist_features.csv"


def load_unique_artists() -> list[str]:
    """Lit le CSV SNEP et renvoie la liste d'artistes uniques (après split feat/&)."""
    df = pd.read_csv(SNEP_CSV, sep=";", encoding="utf-8-sig")
    if "Interprete" not in df.columns:
        raise RuntimeError(f"Colonne 'Interprete' absente de {SNEP_CSV}")
    all_artists: set[str] = set()
    for val in df["Interprete"].dropna():
        for a in split_artists(val):
            all_artists.add(a)
    return sorted(all_artists)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter à N artistes (pour tester)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch tous les artistes même si déjà en cache")
    args = parser.parse_args()

    artists = load_unique_artists()
    logger.info(f"{len(artists)} artistes uniques détectés dans SNEP")

    # cache incrémental : on saute les artistes déjà présents dans le CSV de sortie
    already_fetched: set[str] = set()
    existing: pd.DataFrame | None = None
    if OUTPUT_CSV.exists() and not args.force:
        existing = pd.read_csv(OUTPUT_CSV)
        already_fetched = set(existing["artist_snep"].dropna().astype(str))
        logger.info(f"Cache: {len(already_fetched)} artistes déjà récupérés")

    todo = [a for a in artists if a not in already_fetched]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"À fetch: {len(todo)} artistes")

    if not todo:
        logger.info("Rien à faire. Utilise --force pour tout re-fetcher.")
        return

    def save_checkpoint(records, idx):
        df_new = pd.DataFrame.from_records(records)
        if existing is not None and not args.force:
            df_out = pd.concat([existing, df_new], ignore_index=True)
            df_out = df_out.drop_duplicates(subset=["artist_snep"], keep="last")
        else:
            df_out = df_new
        df_out.to_csv(OUTPUT_CSV, index=False)
        n_found = df_out["spotify_id"].notna().sum()
        logger.info(
            f"Checkpoint @ {idx}/{len(todo)} -> {OUTPUT_CSV.name} "
            f"({len(df_out)} lignes, {n_found} matchs)"
        )

    try:
        fetch_artists_features(todo, on_batch=save_checkpoint, batch_size=100)
    except Exception as e:
        logger.error(f"Interrompu: {e}. Les données déjà fetchées sont sauvegardées.")
        sys.exit(1)


if __name__ == "__main__":
    main()
