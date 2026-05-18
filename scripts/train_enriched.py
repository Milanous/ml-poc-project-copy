"""
Réentraînement des 3 modèles de clustering sur le graphe enrichi Last.fm.

Pipeline :
  1. Charge data/enriched_graph_edges.csv + data/enriched_node_features.csv
  2. Construit nx.Graph -> garde la composante connexe principale
  3. Louvain -> labels (pseudo-vérité terrain)
  4. Node2Vec walks + TruncatedSVD -> embeddings (32 dims)
  5. Concatène les features Last.fm (genres one-hot + log_listeners + n_tags + has_lastfm)
  6. RobustScaler sur le tout
  7. Entraîne KMeans / GMM / BIRCH avec k = #communautés Louvain
  8. Écrase data/node_embeddings.csv + node_labels.csv + node_features.csv + graph_edges.csv
     (les baselines sont déjà sauvés dans data/baseline/)
  9. Écrase models/{kmeans,gmm,birch}.joblib

À l'issue : `python scripts/main.py` chargera ce nouveau cache + ces nouveaux modèles
et écrira results/model_metrics.csv avec ARI/NMI/silhouette enrichis.

Usage : python scripts/train_enriched.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# BIRCH pickle un CFTree récursif : sur 9k+ points la profondeur dépasse 1000.
sys.setrecursionlimit(50000)

import numpy as np
import pandas as pd
import networkx as nx
import joblib
from loguru import logger
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, MODELS_DIR  # noqa: E402
from src.data import build_node2vec_embeddings, NODE2VEC_DIM  # noqa: E402

import community as community_louvain  # noqa: E402

ENRICHED_EDGES = DATA_DIR / "enriched_graph_edges.csv"
ENRICHED_NODES = DATA_DIR / "enriched_node_features.csv"

# Caches utilisés par src/data.py + src/metrics.py
NODE_EMB_CSV    = DATA_DIR / "node_embeddings.csv"
NODE_LABELS_CSV = DATA_DIR / "node_labels.csv"
NODE_FEATS_CSV  = DATA_DIR / "node_features.csv"
GRAPH_EDGES_CSV = DATA_DIR / "graph_edges.csv"

# Features Last.fm à concaténer aux embeddings (en plus des 32 dims Node2Vec)
LASTFM_FEATURE_COLS = [
    "log_listeners", "log_playcount", "n_tags", "has_lastfm",
    "genre_rap_hiphop", "genre_pop", "genre_rock", "genre_electronic",
    "genre_rnb_soul", "genre_chanson_fr", "genre_metal", "genre_jazz",
    "genre_reggae_latin", "genre_classical",
]


def main() -> None:
    logger.info("─── Chargement graphe enrichi ────────────────────────────")
    edges_df = pd.read_csv(ENRICHED_EDGES)
    nodes_df = pd.read_csv(ENRICHED_NODES)
    logger.info(f"  {len(nodes_df)} nœuds, {len(edges_df)} arêtes")

    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["source"], r["target"], weight=float(r["weight"]))
    # plus grande composante connexe (cohérent avec le pipeline original)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    logger.info(f"  LCC : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    # Filtrer les features de nœuds au LCC
    nodes_df = nodes_df[nodes_df["artist"].isin(G.nodes())].reset_index(drop=True)

    # ─── Louvain ──────────────────────────────────────────────────────────
    logger.info("─── Louvain ──────────────────────────────────────────────")
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    k_louvain = len(set(partition.values()))
    logger.info(f"  {k_louvain} communautés détectées")

    # ─── Node2Vec ─────────────────────────────────────────────────────────
    logger.info("─── Node2Vec embeddings ──────────────────────────────────")
    # Sur 9k nœuds, on réduit num_walks pour rester rapide
    nodes, embeddings = build_node2vec_embeddings(
        G, dim=NODE2VEC_DIM, num_walks=20, walk_length=20, p=1.0, q=0.5, seed=42,
    )
    logger.info(f"  embeddings shape : {embeddings.shape}")

    emb_cols = [f"emb_{i}" for i in range(NODE2VEC_DIM)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df.insert(0, "artist", nodes)

    # ─── Concat features Last.fm ──────────────────────────────────────────
    logger.info("─── Concaténation features Last.fm ───────────────────────")
    feat_cols_available = [c for c in LASTFM_FEATURE_COLS if c in nodes_df.columns]
    lastfm_part = (
        nodes_df.set_index("artist")[feat_cols_available]
        .reindex(nodes)
        .fillna(0.0)
        .reset_index(drop=True)
    )
    X_raw = pd.concat([emb_df.drop(columns=["artist"]), lastfm_part], axis=1)
    logger.info(f"  X shape : {X_raw.shape} (Node2Vec + Last.fm)")

    # ─── Labels Louvain alignés sur l'ordre `nodes` ───────────────────────
    y = pd.Series([partition[n] for n in nodes], name="louvain_community")

    # ─── Scaler ───────────────────────────────────────────────────────────
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = pd.DataFrame(X_scaled, columns=X_raw.columns)

    # ─── Sauvegarde caches (pour src/data.py::load_dataset_split) ─────────
    logger.info("─── Sauvegarde caches data/ ─────────────────────────────")
    # node_embeddings.csv : on garde uniquement les 32 dims Node2Vec NON scalées
    # car src/data.py applique RobustScaler ensuite via `scale=True`.
    # Mais pour conserver le bénéfice des features Last.fm, on sauve les 46 dims.
    emb_full = pd.concat(
        [pd.DataFrame({"artist": nodes}), X_raw.reset_index(drop=True)],
        axis=1,
    )
    emb_full.to_csv(NODE_EMB_CSV, index=False)
    logger.info(f"  -> {NODE_EMB_CSV.name} ({len(emb_full)} × {len(emb_full.columns) - 1})")

    labels_df = pd.DataFrame({"artist": nodes, "louvain_community": y.values})
    labels_df.to_csv(NODE_LABELS_CSV, index=False)
    logger.info(f"  -> {NODE_LABELS_CSV.name}")

    # node_features.csv (pour visualisations) -> on prend tout enriched
    nodes_df.to_csv(NODE_FEATS_CSV, index=False)
    logger.info(f"  -> {NODE_FEATS_CSV.name}")

    # graph_edges.csv (pour visualisations app.py)
    edges_df[["source", "target", "weight"]].to_csv(GRAPH_EDGES_CSV, index=False)
    logger.info(f"  -> {GRAPH_EDGES_CSV.name}")

    # ─── Entraînement des 3 modèles ───────────────────────────────────────
    # k aligné sur Louvain : c'est la pseudo-vérité terrain, donc ARI/NMI ne
    # sont comparables que si les modèles produisent le même nombre de clusters.
    k_model = k_louvain
    logger.info(f"─── Entraînement modèles (k={k_model}) ──────────────────")
    kmeans = KMeans(n_clusters=k_model, random_state=42, n_init=10).fit(X)
    gmm    = GaussianMixture(n_components=k_model, covariance_type="full",
                             random_state=42, n_init=3).fit(X)
    birch  = Birch(n_clusters=k_model).fit(X)

    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")
    joblib.dump(gmm,    MODELS_DIR / "gmm.joblib")
    joblib.dump(birch,  MODELS_DIR / "birch.joblib")
    logger.info(f"  -> models/{{kmeans,gmm,birch}}.joblib")

    # ─── Métriques rapides train ──────────────────────────────────────────
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
    )
    logger.info("─── Métriques (sur l'ensemble train, indicatif) ─────────")
    for name, model in [("kmeans", kmeans), ("gmm", gmm), ("birch", birch)]:
        pred = model.predict(X)
        ari = adjusted_rand_score(y, pred)
        nmi = normalized_mutual_info_score(y, pred)
        try:
            sil = silhouette_score(X.values, pred)
        except Exception:
            sil = float("nan")
        logger.info(f"  {name:8s} ARI={ari:.3f}  NMI={nmi:.3f}  silhouette={sil:.3f}  "
                    f"n_clusters={len(set(pred))}")


if __name__ == "__main__":
    main()
