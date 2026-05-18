"""
Client Spotify pour enrichir les artistes SNEP avec des features sémantiques.

Récupère via l'endpoint /v1/artists : genres, popularité, followers, image.
Utilise le flow Client Credentials (pas d'auth utilisateur).

Note: les endpoints /related-artists et /audio-features ont été dépréciés par
Spotify en novembre 2024 pour les nouvelles applications, donc indisponibles ici.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from src.config import ENV_FILE


# ─── Authentification ─────────────────────────────────────────────────────────
def get_spotify_client() -> spotipy.Spotify:
    """Retourne un client Spotipy authentifié via Client Credentials."""
    load_dotenv(ENV_FILE)
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET manquants dans .env"
        )
    auth = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    # retries faibles : un 429 avec Retry-After de plusieurs heures signifie
    # qu'on est blacklisté pour 24h, inutile de boucler.
    return spotipy.Spotify(
        auth_manager=auth,
        requests_timeout=15,
        retries=1,
        status_retries=1,
        backoff_factor=0.5,
    )


# ─── Normalisation des noms d'artistes SNEP ───────────────────────────────────
# Séparateurs reconnus :
#   - "FEAT.", "FT.", "FEATURING" : nécessitent espaces autour
#   - "&", "/", "+" : nécessitent espaces autour (évite "AC/DC", "G&G")
#   - "," : espace requis APRÈS seulement (gère "X, Y, Z" et "X,Y")... mais
#     uniquement si le segment suivant ne ressemble pas à un suffixe label
#   - " AVEC ", " X ", " VS " : nécessitent espaces autour
_FEAT_SPLIT_RE = re.compile(
    r"(?:\s+(?:FEAT\.?|FT\.?|FEATURING|VS\.?|AVEC|&|/|\+|X)\s+|\s*,\s+)",
    flags=re.IGNORECASE,
)


def split_artists(interprete: str) -> list[str]:
    """
    Sépare un champ 'Interprete' SNEP en artistes individuels.

    Exemples:
      "GIMS FEAT. SOOLKING"          -> ["GIMS", "SOOLKING"]
      "BIGFLO & OLI"                  -> ["BIGFLO", "OLI"]
      "-M-, TOUMANI DIABATÉ, SIDIKI" -> ["-M-", "TOUMANI DIABATÉ", "SIDIKI"]
      "JUL"                           -> ["JUL"]
    """
    if not isinstance(interprete, str):
        return []
    parts = _FEAT_SPLIT_RE.split(interprete)
    return [p.strip() for p in parts if p.strip()]


# ─── Recherche & enrichissement ───────────────────────────────────────────────
def search_artist(sp: spotipy.Spotify, name: str) -> dict[str, Any] | None:
    """
    Cherche un artiste sur Spotify et renvoie le meilleur match.

    Retourne None si rien trouvé.
    """
    try:
        results = sp.search(q=f'artist:"{name}"', type="artist", limit=5)
    except spotipy.SpotifyException as e:
        logger.warning(f"Erreur Spotify pour '{name}': {e}")
        return None

    items = results.get("artists", {}).get("items", [])
    if not items:
        # fallback: recherche moins stricte sans le préfixe artist:
        try:
            results = sp.search(q=name, type="artist", limit=5)
        except spotipy.SpotifyException:
            return None
        items = results.get("artists", {}).get("items", [])
        if not items:
            return None

    # heuristique: le 1er résultat est généralement le bon (Spotify trie par pertinence)
    # mais on privilégie un match exact du nom (insensible à la casse)
    name_upper = name.upper()
    for item in items:
        if item["name"].upper() == name_upper:
            return item
    return items[0]


def artist_to_record(name_snep: str, item: dict[str, Any] | None) -> dict[str, Any]:
    """Aplatit la réponse Spotify en un dict compatible CSV."""
    if item is None:
        return {
            "artist_snep": name_snep,
            "spotify_id": None,
            "spotify_name": None,
            "genres": None,
            "popularity": None,
            "followers": None,
            "image_url": None,
            "spotify_url": None,
        }
    images = item.get("images") or []
    return {
        "artist_snep": name_snep,
        "spotify_id": item.get("id"),
        "spotify_name": item.get("name"),
        "genres": "|".join(item.get("genres") or []),
        "popularity": item.get("popularity"),
        "followers": (item.get("followers") or {}).get("total"),
        "image_url": images[0]["url"] if images else None,
        "spotify_url": (item.get("external_urls") or {}).get("spotify"),
    }


def fetch_artists_features(
    names: list[str],
    sleep_between: float = 0.0,
    on_batch=None,
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Récupère les features Spotify pour une liste d'artistes (déjà uniques).

    `sleep_between` : pause optionnelle entre requêtes.
    `on_batch(records, last_index)` : callback appelé tous les `batch_size`
        artistes ET à la fin, pour permettre une sauvegarde incrémentale.
        Si la fonction lève une exception (ex: rate-limit fatal), elle
        remonte avec les `records` accumulés jusqu'ici.
    """
    sp = get_spotify_client()
    records: list[dict[str, Any]] = []
    n = len(names)
    try:
        for i, name in enumerate(names, 1):
            item = search_artist(sp, name)
            records.append(artist_to_record(name, item))
            if i % 25 == 0 or i == n:
                logger.info(f"Spotify: {i}/{n} artistes traités")
            if on_batch and (i % batch_size == 0 or i == n):
                on_batch(records, i)
            if sleep_between:
                time.sleep(sleep_between)
    except spotipy.SpotifyException as e:
        logger.error(f"Arrêt Spotify ({e}). {len(records)} artistes en mémoire.")
        if on_batch and records:
            on_batch(records, len(records))
        raise
    return records
