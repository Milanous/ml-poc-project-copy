"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.

Tâche ML : clustering non supervisé de communautés musicales dans le graphe de
co-artistes SNEP.

Pipeline :
1. Chargement + nettoyage du CSV brut (``load_clean_df``)
2. Construction du graphe de co-artistes pondéré (``build_graph``)
3. Feature engineering des nœuds (``build_node_features``)
4. Embeddings Node2Vec (walks biaisés + TruncatedSVD) (``build_node2vec_embeddings``)
5. Split des nœuds train/test + labels de communauté Louvain (``load_dataset_split``)

Les choix de transformation sont justifiés dans ``notebooks/feature_engineering.ipynb``.
"""

from __future__ import annotations

import re
import random
import unicodedata
from itertools import combinations
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
_DATA_DIR        = Path(__file__).parent.parent / "data"
_RAW_CSV         = _DATA_DIR / "snep_certifications.csv"
_GRAPH_EDGES_CSV = _DATA_DIR / "graph_edges.csv"
_NODE_FEATS_CSV  = _DATA_DIR / "node_features.csv"
_NODE_EMB_CSV    = _DATA_DIR / "node_embeddings.csv"
_NODE_LABELS_CSV = _DATA_DIR / "node_labels.csv"

# ─── Constants ────────────────────────────────────────────────────────────────
CERT_ORDER = [
    "Or", "Double Or", "Platine", "Double Platine", "Triple Platine",
    "Diamant", "Double Diamant", "Triple Diamant", "Quadruple Diamant",
]
CERT_LEVEL = {cert: i for i, cert in enumerate(CERT_ORDER)}

# Seuil minimum de poids d'arête pour garder une collaboration dans le graphe
MIN_COLLAB_WEIGHT = 2

# Dimensions des embeddings Node2Vec (SVD)
NODE2VEC_DIM = 32

# Colonnes de features de nœuds
NODE_FEATURE_COLS = [
    "degree", "weighted_degree",
    "total_certifs", "max_cert_level", "avg_cert_level",
    "collab_rate", "singles_rate", "annee_median",
]

# ─── Regex partagés ───────────────────────────────────────────────────────────
_COLLAB_PAT = re.compile(r"\s+FEAT\.?\s*|\s+FT\.?\s*", re.IGNORECASE)
_AMP_PAT    = re.compile(r"\s*&\s*")
_COMMA_PAT  = re.compile(r"\s*,\s*")


# ─── Helpers de nettoyage ─────────────────────────────────────────────────────

def _normalize_interprete(s: str) -> str:
    """Majuscules, suppression des diacritiques, espaces normalisés."""
    if pd.isna(s):
        return s
    s = s.strip().upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def _build_known_duos(all_interps: set[str]) -> set[str]:
    """Détecte les duos/groupes avec '&' dont les membres n'existent pas en solo."""
    duos: set[str] = set()
    for name in sorted(all_interps):
        if "&" not in name or re.search(r"\sFEAT\.?\s|\sFT\.?\s", name, re.IGNORECASE):
            continue
        parts = [p.strip() for p in name.split("&")]
        if not any(p in all_interps for p in parts if len(p) > 3):
            duos.add(name)
    return duos


def _split_artists(raw: str, known_duos: set[str]) -> list[str]:
    """Décompose une chaîne d'artistes en liste, en préservant les duos connus."""
    s = raw.strip()
    placeholders: dict[str, str] = {}
    s_cmp = s.upper()
    for i, duo in enumerate(sorted(known_duos, key=len, reverse=True)):
        idx = s_cmp.find(duo.upper())
        if idx >= 0:
            ph = f"__DUO{i}__"
            placeholders[ph] = s[idx:idx + len(duo)]
            s = s[:idx] + ph + s[idx + len(duo):]
            s_cmp = s.upper()
    s = _COLLAB_PAT.sub("|", s)
    s = _AMP_PAT.sub("|", s)
    s = _COMMA_PAT.sub("|", s)
    parts = [p.strip() for p in s.split("|") if p.strip()]
    result = []
    for p in parts:
        for ph, original in placeholders.items():
            p = p.replace(ph, original)
        if p:
            result.append(p)
    return result


# ─── Chargement brut ──────────────────────────────────────────────────────────

def load_clean_df() -> pd.DataFrame:
    """Charge et nettoie le CSV brut.

    Retourne un DataFrame avec colonnes enrichies :
    ``artists_list``, ``is_collab``, ``artiste_principal``,
    ``cert_level``, ``Certification_norm``, ``Categorie_norm``, ``annee_sortie``.
    """
    df = pd.read_csv(
        _RAW_CSV, sep=";", encoding="utf-8-sig",
        parse_dates=["Date de sortie", "Date de constat"],
        dayfirst=True,
    )
    df["jours_obtention"]    = (df["Date de constat"] - df["Date de sortie"]).dt.days
    df                       = df.drop(columns=["Duree obtention"])
    df["Interprete"]         = df["Interprete"].apply(_normalize_interprete)
    df["Certification_norm"] = df["Certification"].str.strip().str.title()
    df["Categorie_norm"]     = df["Categorie"].str.strip().replace("Single", "Singles")
    df["cert_level"]         = df["Certification_norm"].map(CERT_LEVEL)
    df["annee_sortie"]       = df["Date de sortie"].dt.year

    df = df.dropna(subset=["Interprete", "cert_level", "Categorie_norm"]).reset_index(drop=True)

    all_interps      = set(df["Interprete"].unique())
    known_duos       = _build_known_duos(all_interps)
    df["artists_list"]      = df["Interprete"].apply(lambda x: _split_artists(x, known_duos))
    df["is_collab"]         = (df["artists_list"].apply(len) > 1).astype(int)
    df["artiste_principal"] = df["artists_list"].apply(lambda x: x[0] if x else None)

    return df


# ─── Construction du graphe ───────────────────────────────────────────────────

def build_graph(
    df: pd.DataFrame | None = None,
    min_weight: int = MIN_COLLAB_WEIGHT,
) -> nx.Graph:
    """Construit le graphe de co-artistes pondéré et filtré.

    Paramètres
    ----------
    df : DataFrame produit par ``load_clean_df()``. Si None, le recharge.
    min_weight : poids minimum pour conserver une arête (défaut = 2).

    Retour
    ------
    ``nx.Graph`` non orienté, arêtes pondérées, nœuds = artistes.
    Seule la composante connexe principale est conservée.
    """
    if df is None:
        df = load_clean_df()

    edge_counter: Counter = Counter()
    for artists in df[df["is_collab"] == 1]["artists_list"]:
        for a, b in combinations(sorted(set(artists)), 2):
            edge_counter[(a, b)] += 1

    edges = [(a, b, w) for (a, b), w in edge_counter.items() if w >= min_weight]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


# ─── Feature engineering des nœuds ───────────────────────────────────────────

def build_node_features(G: nx.Graph, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Construit le DataFrame de features par nœud (artiste).

    Colonnes produites : voir ``NODE_FEATURE_COLS``.
    """
    if df is None:
        df = load_clean_df()

    degree          = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))

    agg = (
        df.groupby("artiste_principal")
        .agg(
            total_certifs  = ("cert_level", "count"),
            max_cert_level = ("cert_level", "max"),
            avg_cert_level = ("cert_level", "mean"),
            collab_rate    = ("is_collab", "mean"),
            annee_median   = ("annee_sortie", "median"),
        )
        .reset_index()
        .rename(columns={"artiste_principal": "artist"})
    )

    singles_rate = (
        df[df["Categorie_norm"] == "Singles"]
        .groupby("artiste_principal").size()
        / df.groupby("artiste_principal").size()
    ).rename("singles_rate").reset_index().rename(columns={"artiste_principal": "artist"})
    agg = agg.merge(singles_rate, on="artist", how="left").fillna({"singles_rate": 0.0})

    node_df = pd.DataFrame({
        "artist":          list(G.nodes()),
        "degree":          [degree[n] for n in G.nodes()],
        "weighted_degree": [weighted_degree[n] for n in G.nodes()],
    })

    result = (
        node_df
        .merge(agg, on="artist", how="left")
        .fillna({
            "total_certifs": 0, "max_cert_level": 0, "avg_cert_level": 0.0,
            "collab_rate": 0.0, "annee_median": df["annee_sortie"].median(),
            "singles_rate": 0.0,
        })
    )
    return result


# ─── Node2Vec embeddings ──────────────────────────────────────────────────────

def _node2vec_walks(
    G: nx.Graph,
    num_walks: int = 100,
    walk_length: int = 30,
    p: float = 1.0,
    q: float = 0.5,
    seed: int = 42,
) -> list[list]:
    """Génère des random walks biaisés Node2Vec sur le graphe.

    q=0.5 : BFS-biaisé, favorise l'exploration locale (meilleur pour les communautés).
    """
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                cur       = walk[-1]
                neighbors = list(G.neighbors(cur))
                if not neighbors:
                    break
                if len(walk) == 1:
                    walk.append(neighbors[int(rng.integers(len(neighbors)))])
                else:
                    prev    = walk[-2]
                    w_arr   = np.array([
                        1.0 / p if nb == prev
                        else 1.0 if G.has_edge(prev, nb)
                        else 1.0 / q
                        for nb in neighbors
                    ], dtype=np.float64)
                    probs = w_arr / w_arr.sum()
                    walk.append(neighbors[rng.choice(len(neighbors), p=probs)])
            walks.append(walk)
    return walks


def build_node2vec_embeddings(
    G: nx.Graph,
    dim: int = NODE2VEC_DIM,
    num_walks: int = 100,
    walk_length: int = 30,
    p: float = 1.0,
    q: float = 0.5,
    window: int = 5,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Calcule les embeddings Node2Vec par walks + TruncatedSVD.

    Retour
    ------
    nodes : liste des artistes dans l'ordre des lignes
    embeddings : ndarray (n_nodes, dim)
    """
    nodes    = list(G.nodes())
    n        = len(nodes)
    node2idx = {nd: i for i, nd in enumerate(nodes)}

    walks = _node2vec_walks(G, num_walks=num_walks, walk_length=walk_length,
                            p=p, q=q, seed=seed)

    cooc = np.zeros((n, n), dtype=np.float32)
    for walk in walks:
        for i, nd in enumerate(walk):
            i_idx = node2idx.get(nd)
            if i_idx is None:
                continue
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    j_idx = node2idx.get(walk[j])
                    if j_idx is not None:
                        cooc[i_idx, j_idx] += 1.0

    svd        = TruncatedSVD(n_components=dim, random_state=seed)
    embeddings = svd.fit_transform(cooc)
    return nodes, embeddings


# ─── API publique ─────────────────────────────────────────────────────────────

def load_dataset_split(
    *,
    use_cached: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Tâche : clustering de communautés musicales.
    - ``X_*`` = embeddings Node2Vec des artistes (features topologiques du graphe)
    - ``y_*`` = labels de communauté Louvain (pseudo-labels non supervisés)

    Parameters
    ----------
    use_cached : bool
        Si True et que les fichiers CSV existent dans ``data/``, les recharge
        au lieu de recalculer le pipeline complet.
    test_size : float
        Fraction des nœuds réservés pour le test (défaut 0.2).
    random_state : int
        Graine aléatoire pour la reproductibilité.
    scale : bool
        Si True (défaut), applique ``RobustScaler`` aux embeddings.

    Returns
    -------
    X_train, X_test : pd.DataFrame  (colonnes = emb_0 … emb_{NODE2VEC_DIM-1})
    y_train, y_test : pd.Series     (values = louvain_community)
    """
    import community as community_louvain  # python-louvain (import optionnel)

    if (
        use_cached
        and _NODE_EMB_CSV.exists()
        and _NODE_LABELS_CSV.exists()
    ):
        emb_df    = pd.read_csv(_NODE_EMB_CSV)
        labels_df = pd.read_csv(_NODE_LABELS_CSV)
        nodes     = emb_df["artist"].tolist()
        emb_cols  = [c for c in emb_df.columns if c != "artist"]
        X         = emb_df[emb_cols]
        y         = labels_df.set_index("artist").loc[nodes, "louvain_community"]
        y.index   = range(len(y))
    else:
        df = load_clean_df()
        G  = build_graph(df, min_weight=MIN_COLLAB_WEIGHT)

        nodes, embeddings = build_node2vec_embeddings(G, dim=NODE2VEC_DIM)
        partition = community_louvain.best_partition(G, weight="weight", random_state=random_state)

        emb_cols = [f"emb_{i}" for i in range(NODE2VEC_DIM)]
        X = pd.DataFrame(embeddings, columns=emb_cols)
        y = pd.Series([partition.get(n, 0) for n in nodes], name="louvain_community")

    if scale:
        scaler = RobustScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
