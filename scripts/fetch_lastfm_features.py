"""
Récupère via Last.fm les features ML pour tous les artistes uniques SNEP :
  - `data/lastfm_artist_features.csv` : tags, listeners, playcount, mbid
  - `data/lastfm_similar_edges.csv`   : arêtes pondérées (source, target, match)

Cache incrémental : un re-run reprend là où on s'est arrêté.

Usage:
    python scripts/fetch_lastfm_features.py
    python scripts/fetch_lastfm_features.py --limit 50
    python scripts/fetch_lastfm_features.py --no-similar       # tags seulement
    python scripts/fetch_lastfm_features.py --force            # ignore le cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.lastfm import get_artist_info, get_similar_artists  # noqa: E402
from src.spotify import split_artists  # noqa: E402


SNEP_CSV = DATA_DIR / "snep_certifications.csv"
FEATURES_CSV = DATA_DIR / "lastfm_artist_features.csv"
EDGES_CSV = DATA_DIR / "lastfm_similar_edges.csv"


def load_unique_artists() -> list[str]:
    df = pd.read_csv(SNEP_CSV, sep=";", encoding="utf-8-sig")
    s: set[str] = set()
    for v in df["Interprete"].dropna():
        for a in split_artists(v):
            s.add(a)
    return sorted(s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-similar", action="store_true",
                   help="Ne pas récupérer les artistes similaires")
    p.add_argument("--similar-limit", type=int, default=30,
                   help="Nombre max d'artistes similaires par artiste")
    p.add_argument("--force", action="store_true")
    p.add_argument("--sleep", type=float, default=0.05,
                   help="Pause entre requêtes (Last.fm autorise ~5 req/s)")
    args = p.parse_args()

    artists = load_unique_artists()
    logger.info(f"{len(artists)} artistes uniques SNEP")

    # cache
    done_features: set[str] = set()
    done_similar: set[str] = set()
    df_feat_old = pd.DataFrame()
    df_edges_old = pd.DataFrame()
    if FEATURES_CSV.exists() and not args.force:
        df_feat_old = pd.read_csv(FEATURES_CSV)
        done_features = set(df_feat_old["artist_snep"].dropna().astype(str))
    if EDGES_CSV.exists() and not args.force:
        df_edges_old = pd.read_csv(EDGES_CSV)
        done_similar = set(df_edges_old["source"].dropna().astype(str))

    todo = [a for a in artists if a not in done_features or
            (not args.no_similar and a not in done_similar)]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(
        f"Cache: {len(done_features)} features, {len(done_similar)} similar | "
        f"À traiter: {len(todo)}"
    )

    new_features: list[dict] = []
    new_edges: list[dict] = []
    n_found = 0
    for i, name in enumerate(todo, 1):
        if name not in done_features or args.force:
            info = get_artist_info(name)
            if info:
                n_found += 1
                new_features.append(info)
            else:
                # on enregistre un None pour ne pas réinterroger inutilement
                new_features.append({
                    "artist_snep": name, "lastfm_name": None, "mbid": None,
                    "listeners": None, "playcount": None, "tags": None,
                    "lastfm_url": None,
                })
            time.sleep(args.sleep)

        if not args.no_similar and (name not in done_similar or args.force):
            edges = get_similar_artists(name, limit=args.similar_limit)
            new_edges.extend(edges)
            time.sleep(args.sleep)

        if i % 25 == 0 or i == len(todo):
            logger.info(f"Last.fm: {i}/{len(todo)} | matchs: {n_found}")

    # Merge & save
    if new_features:
        df_feat = pd.concat(
            [df_feat_old, pd.DataFrame(new_features)], ignore_index=True
        ).drop_duplicates(subset=["artist_snep"], keep="last")
        df_feat.to_csv(FEATURES_CSV, index=False)
        logger.info(f"-> {FEATURES_CSV} ({len(df_feat)} lignes)")

    if not args.no_similar and new_edges:
        df_edges = pd.concat(
            [df_edges_old, pd.DataFrame(new_edges)], ignore_index=True
        ).drop_duplicates(subset=["source", "target"], keep="last")
        df_edges.to_csv(EDGES_CSV, index=False)
        logger.info(f"-> {EDGES_CSV} ({len(df_edges)} arêtes)")


if __name__ == "__main__":
    main()
