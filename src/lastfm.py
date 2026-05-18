"""
Client Last.fm pour enrichir les artistes SNEP avec :
  - Tags musicaux (genres) et leur poids
  - Listeners & playcount (proxy de popularité internationale)
  - Artistes similaires (avec score de match) → nouvelles arêtes pour le graphe

API publique (REST/JSON), pas de SDK requis.
Doc : https://www.last.fm/api/intro
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger

from src.config import ENV_FILE


LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"


def _get_api_key() -> str:
    load_dotenv(ENV_FILE)
    key = os.getenv("LASTFM_API_KEY")
    if not key:
        raise RuntimeError("LASTFM_API_KEY manquant dans .env")
    return key


def _call(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Appel générique à l'API Last.fm avec gestion basique du rate-limit."""
    payload = {
        "method": method,
        "api_key": _get_api_key(),
        "format": "json",
        **params,
    }
    for attempt in range(4):
        try:
            r = requests.get(LASTFM_BASE_URL, params=payload, timeout=15)
        except requests.RequestException as e:
            logger.warning(f"Erreur réseau Last.fm ({method}): {e}; retry...")
            time.sleep(1 + attempt)
            continue
        if r.status_code == 429:
            wait = 2 ** attempt
            logger.warning(f"429 rate-limit, attente {wait}s")
            time.sleep(wait)
            continue
        try:
            return r.json()
        except ValueError:
            logger.warning(f"Réponse non-JSON Last.fm ({method})")
            return {}
    return {}


# ─── artist.getInfo : tags + listeners + playcount ───────────────────────────
def get_artist_info(name: str) -> dict[str, Any] | None:
    """Renvoie un dict aplati ou None si artiste introuvable."""
    data = _call("artist.getInfo", {"artist": name, "autocorrect": 1})
    artist = data.get("artist")
    if not artist:
        return None
    stats = artist.get("stats") or {}
    tags_block = (artist.get("tags") or {}).get("tag") or []
    if isinstance(tags_block, dict):
        tags_block = [tags_block]
    tags = [t.get("name") for t in tags_block if t.get("name")]
    return {
        "artist_snep": name,
        "lastfm_name": artist.get("name"),
        "mbid": artist.get("mbid") or None,
        "listeners": _to_int(stats.get("listeners")),
        "playcount": _to_int(stats.get("playcount")),
        "tags": "|".join(tags),
        "lastfm_url": artist.get("url"),
    }


# ─── artist.getSimilar : arêtes pondérées pour le graphe ─────────────────────
def get_similar_artists(
    name: str, limit: int = 30
) -> list[dict[str, Any]]:
    """Renvoie une liste de dicts {source, target, match} (match dans [0,1])."""
    data = _call(
        "artist.getSimilar",
        {"artist": name, "autocorrect": 1, "limit": limit},
    )
    items = (data.get("similarartists") or {}).get("artist") or []
    if isinstance(items, dict):
        items = [items]
    edges: list[dict[str, Any]] = []
    for it in items:
        target = it.get("name")
        if not target:
            continue
        edges.append(
            {
                "source": name,
                "target": target,
                "match": _to_float(it.get("match")),
            }
        )
    return edges


def _to_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
