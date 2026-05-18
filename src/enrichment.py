"""
Enrichissement du graphe et des features SNEP avec les données Last.fm.

Ce module ne touche pas au pipeline existant (`src/data.py`). Il fournit :
  - `load_lastfm_features()`            : charge le CSV Last.fm + normalise les noms
  - `load_lastfm_edges()`               : charge les arêtes de similarité filtrées
  - `build_tag_features()`              : features one-hot des top genres + listeners
  - `enrich_graph_with_similarity()`    : ajoute les arêtes similar à un graphe existant

Toutes les fonctions retournent des noms d'artistes au format `data.py` :
UPPERCASE, sans accents, espaces normalisés.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from src.config import DATA_DIR


# ─── Paths ────────────────────────────────────────────────────────────────────
_LASTFM_FEATURES = DATA_DIR / "lastfm_artist_features.csv"
_LASTFM_EDGES    = DATA_DIR / "lastfm_similar_edges.csv"


# ─── Normalisation (alignée avec src/data.py::_normalize_interprete) ─────────
def _normalize(s: str | float) -> str | None:
    """Majuscules, sans diacritiques, espaces normalisés.

    Doit produire la même sortie que `src.data._normalize_interprete`.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip().upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


# ─── Chargement Last.fm ───────────────────────────────────────────────────────
def load_lastfm_features() -> pd.DataFrame:
    """Charge `lastfm_artist_features.csv` avec colonnes normalisées.

    Ajoute une colonne `artist_norm` (= clé de jointure avec le pipeline data.py).
    """
    df = pd.read_csv(_LASTFM_FEATURES)
    df["artist_norm"] = df["artist_snep"].apply(_normalize)
    # priorité aux lignes matchées si doublons après normalisation
    df = df.sort_values("lastfm_name", na_position="last")
    df = df.drop_duplicates(subset=["artist_norm"], keep="first")
    return df


def load_lastfm_edges() -> pd.DataFrame:
    """Charge `lastfm_similar_edges.csv` avec source/target normalisées."""
    df = pd.read_csv(_LASTFM_EDGES)
    df["source_norm"] = df["source"].apply(_normalize)
    df["target_norm"] = df["target"].apply(_normalize)
    df = df.dropna(subset=["source_norm", "target_norm", "match"])
    df = df[df["source_norm"] != df["target_norm"]]
    return df


# ─── Features tag-based ───────────────────────────────────────────────────────
# Top genres choisis manuellement à partir de l'audit des tags
# (cf. analyse: pop, french, hip-hop, rap, rock, electronic, rnb, dance, indie, soul)
DEFAULT_GENRE_BUCKETS: dict[str, list[str]] = {
    "rap_hiphop":   ["rap", "hip-hop", "hip hop", "trap", "drill", "rap francais"],
    "pop":          ["pop", "pop francaise"],
    "rock":         ["rock", "indie", "alternative", "indie rock", "rock francais"],
    "electronic":   ["electronic", "edm", "house", "techno", "dance", "electro"],
    "rnb_soul":     ["rnb", "r&b", "soul", "neo-soul", "funk"],
    "chanson_fr":   ["chanson francaise", "chanson", "variete", "variete francaise"],
    "metal":        ["metal", "hard rock", "heavy metal"],
    "jazz":         ["jazz", "blues"],
    "reggae_latin": ["reggae", "latin", "afrobeats", "afro", "dancehall"],
    "classical":    ["classical", "soundtrack", "instrumental"],
}


def _tags_to_buckets(tags_str: str, buckets: dict[str, list[str]]) -> dict[str, int]:
    """Convertit '|'-joined tags Last.fm en flags one-hot par bucket."""
    out = {f"genre_{k}": 0 for k in buckets}
    if not isinstance(tags_str, str) or not tags_str:
        return out
    tags_lower = {t.strip().lower() for t in tags_str.split("|")}
    for bucket, keywords in buckets.items():
        if any(kw in tags_lower for kw in keywords):
            out[f"genre_{bucket}"] = 1
    return out


def build_tag_features(
    artists_norm: list[str],
    buckets: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Construit un DataFrame de features Last.fm pour une liste d'artistes (normalisés).

    Colonnes :
      - artist             : nom normalisé (clé de jointure)
      - log_listeners      : log(1 + listeners), 0 si manquant
      - log_playcount      : log(1 + playcount), 0 si manquant
      - n_tags             : nombre de tags Last.fm
      - genre_<bucket>     : one-hot pour chaque bucket de DEFAULT_GENRE_BUCKETS
      - has_lastfm         : 1 si l'artiste a été matché sur Last.fm
    """
    buckets = buckets or DEFAULT_GENRE_BUCKETS
    feat_df = load_lastfm_features()
    feat_df = feat_df[feat_df["artist_norm"].isin(set(artists_norm))]

    rows = []
    for art in artists_norm:
        row = {"artist": art}
        match = feat_df[feat_df["artist_norm"] == art]
        if not match.empty:
            r = match.iloc[0]
            listeners = r["listeners"] if pd.notna(r["listeners"]) else 0
            playcount = r["playcount"] if pd.notna(r["playcount"]) else 0
            tags = r["tags"] if isinstance(r["tags"], str) else ""
            row["log_listeners"] = float(np.log1p(listeners))
            row["log_playcount"] = float(np.log1p(playcount))
            row["n_tags"] = len([t for t in tags.split("|") if t.strip()]) if tags else 0
            row["has_lastfm"] = 1 if pd.notna(r["lastfm_name"]) else 0
            row.update(_tags_to_buckets(tags, buckets))
        else:
            row["log_listeners"] = 0.0
            row["log_playcount"] = 0.0
            row["n_tags"] = 0
            row["has_lastfm"] = 0
            row.update({f"genre_{k}": 0 for k in buckets})
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Enrichissement du graphe ────────────────────────────────────────────────
def enrich_graph_with_similarity(
    G: nx.Graph,
    min_match: float = 0.5,
    similarity_weight_scale: float = 1.0,
    add_new_nodes: bool = False,
    min_snep_anchors: int = 0,
) -> nx.Graph:
    """Ajoute les arêtes de similarité Last.fm au graphe existant.

    Paramètres
    ----------
    G : graphe existant (nœuds = artistes normalisés au format data.py).
    min_match : score Last.fm minimum pour ajouter une arête (0–1).
    similarity_weight_scale : multiplicateur du score Last.fm.
    add_new_nodes : si True, ajoute les artistes Last.fm absents de G comme
        nouveaux nœuds. Sinon (défaut), garde uniquement les arêtes internes.
    min_snep_anchors : si > 0 et add_new_nodes=True, ne conserve un nouveau
        nœud Last.fm que s'il est connecté à au moins N nœuds SNEP originaux
        (évite l'explosion combinatoire par les feuilles).

    Retour : nouveau graphe enrichi (G n'est pas modifié).
    """
    edges = load_lastfm_edges()
    edges = edges[edges["match"] >= min_match]

    nodes_set = set(G.nodes())
    if not add_new_nodes:
        edges = edges[
            edges["source_norm"].isin(nodes_set)
            & edges["target_norm"].isin(nodes_set)
        ]

    G_new = G.copy()
    n_added = 0
    n_updated = 0
    for _, row in edges.iterrows():
        u, v = row["source_norm"], row["target_norm"]
        w_sim = float(row["match"]) * similarity_weight_scale
        if G_new.has_edge(u, v):
            # on additionne le poids existant (collab) avec la similarité
            G_new[u][v]["weight"] = G_new[u][v].get("weight", 0) + w_sim
            n_updated += 1
        else:
            G_new.add_edge(u, v, weight=w_sim)
            n_added += 1

    G_new.graph["enrichment"] = {
        "edges_added": n_added,
        "edges_updated": n_updated,
        "min_match": min_match,
    }

    # Filtrage d'ancrage : un nouveau nœud (hors SNEP) doit être connecté
    # à au moins `min_snep_anchors` nœuds SNEP originaux.
    if add_new_nodes and min_snep_anchors > 0:
        snep_nodes = nodes_set  # nœuds du graphe original
        new_nodes = set(G_new.nodes()) - snep_nodes
        to_remove = [
            n for n in new_nodes
            if sum(1 for nb in G_new.neighbors(n) if nb in snep_nodes) < min_snep_anchors
        ]
        G_new.remove_nodes_from(to_remove)
        G_new.graph["enrichment"]["removed_low_anchor"] = len(to_remove)
        G_new.graph["enrichment"]["min_snep_anchors"] = min_snep_anchors

    return G_new
