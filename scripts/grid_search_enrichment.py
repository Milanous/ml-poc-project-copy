"""
Grid search sur (min_snep_anchors, min_match) pour trouver la meilleure
config d'enrichissement Last.fm. Évalue ARI/NMI/silhouette sur train.

Usage : python scripts/grid_search_enrichment.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
)
import community as community_louvain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    load_clean_df, build_graph, build_node_features,
    build_node2vec_embeddings, NODE2VEC_DIM, MIN_COLLAB_WEIGHT,
)
from src.enrichment import (  # noqa: E402
    enrich_graph_with_similarity, build_tag_features,
)

LASTFM_FEATURE_COLS = [
    "log_listeners", "log_playcount", "n_tags", "has_lastfm",
    "genre_rap_hiphop", "genre_pop", "genre_rock", "genre_electronic",
    "genre_rnb_soul", "genre_chanson_fr", "genre_metal", "genre_jazz",
    "genre_reggae_latin", "genre_classical",
]


def evaluate(anchor: int, min_match: float, sim_scale: float) -> dict:
    df = load_clean_df()
    G0 = build_graph(df, min_weight=MIN_COLLAB_WEIGHT)
    G1 = enrich_graph_with_similarity(
        G0, min_match=min_match, similarity_weight_scale=sim_scale,
        add_new_nodes=True, min_snep_anchors=anchor,
    )
    largest_cc = max(nx.connected_components(G1), key=len)
    G1 = G1.subgraph(largest_cc).copy()
    n_nodes = G1.number_of_nodes()
    n_edges = G1.number_of_edges()

    partition = community_louvain.best_partition(G1, weight="weight", random_state=42)
    k = len(set(partition.values()))

    nodes, emb = build_node2vec_embeddings(
        G1, dim=NODE2VEC_DIM, num_walks=20, walk_length=20, seed=42,
    )
    topo = build_node_features(G1, df)
    lastfm = build_tag_features(nodes)
    feat = topo.set_index("artist").join(
        lastfm.set_index("artist"), how="left"
    ).reindex(nodes).fillna(0.0)
    X_raw = np.concatenate([emb, feat[LASTFM_FEATURE_COLS].values], axis=1)
    X = RobustScaler().fit_transform(X_raw)
    y = np.array([partition[n] for n in nodes])

    out = {"anchor": anchor, "min_match": min_match, "sim_scale": sim_scale,
           "n_nodes": n_nodes, "n_edges": n_edges, "k_louvain": k}
    for name, model in [
        ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
        ("gmm",    GaussianMixture(n_components=k, covariance_type="full",
                                    random_state=42, n_init=3)),
        ("birch",  Birch(n_clusters=k)),
    ]:
        model.fit(X)
        pred = model.predict(X)
        try:
            sil = silhouette_score(X, pred)
        except Exception:
            sil = float("nan")
        out[f"{name}_ari"] = adjusted_rand_score(y, pred)
        out[f"{name}_nmi"] = normalized_mutual_info_score(y, pred)
        out[f"{name}_sil"] = sil
    return out


def main() -> None:
    grid = list(product(
        [1, 2, 3, 4],         # anchors
        [0.4, 0.5, 0.6],      # min_match
        [1.0],                # sim_scale (fixé pour réduire combinaisons)
    ))
    print(f"Évaluation de {len(grid)} configs...")
    rows = []
    for i, (a, m, s) in enumerate(grid, 1):
        r = evaluate(a, m, s)
        print(f"[{i:2d}/{len(grid)}] anchor={a} min_match={m} | "
              f"nodes={r['n_nodes']:4d} edges={r['n_edges']:5d} k={r['k_louvain']:2d} | "
              f"ARI km/gmm/br = {r['kmeans_ari']:.3f}/{r['gmm_ari']:.3f}/{r['birch_ari']:.3f} | "
              f"sil km/gmm/br = {r['kmeans_sil']:.3f}/{r['gmm_sil']:.3f}/{r['birch_sil']:.3f}")
        rows.append(r)
    df = pd.DataFrame(rows)
    df["avg_ari"] = df[["kmeans_ari", "gmm_ari", "birch_ari"]].mean(axis=1)
    df["avg_sil"] = df[["kmeans_sil", "gmm_sil", "birch_sil"]].mean(axis=1)
    df["score"] = df["avg_ari"] + df["avg_sil"]
    df = df.sort_values("score", ascending=False)
    out_csv = PROJECT_ROOT / "results" / "grid_search_enrichment.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nTop 5 configs (par avg_ari + avg_sil) :")
    print(df.head(5)[["anchor", "min_match", "n_nodes", "k_louvain",
                       "avg_ari", "avg_sil", "score"]].to_string(index=False))
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
