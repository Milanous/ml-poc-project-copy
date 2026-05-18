"""
Construit un graphe enrichi (collaborations SNEP + similarité Last.fm) et
sauvegarde les nouveaux CSVs dans `data/enriched_*.csv` sans toucher aux
originaux.

Sorties :
  - data/enriched_graph_edges.csv   : arêtes (source, target, weight, type)
  - data/enriched_node_features.csv : features par nœud (topologiques + Last.fm)

Usage:
    python scripts/build_enriched_graph.py
    python scripts/build_enriched_graph.py --min-match 0.7    # filtrage plus strict
    python scripts/build_enriched_graph.py --add-new-nodes    # élargit au-delà des artistes SNEP
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import networkx as nx
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.data import (  # noqa: E402
    load_clean_df, build_graph, build_node_features, MIN_COLLAB_WEIGHT,
)
from src.enrichment import (  # noqa: E402
    enrich_graph_with_similarity, build_tag_features,
)


OUT_EDGES = DATA_DIR / "enriched_graph_edges.csv"
OUT_NODES = DATA_DIR / "enriched_node_features.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-match", type=float, default=0.5,
                   help="Score Last.fm minimum pour ajouter une arête (def 0.5)")
    p.add_argument("--min-collab-weight", type=int, default=MIN_COLLAB_WEIGHT,
                   help=f"Poids min des arêtes collab (def {MIN_COLLAB_WEIGHT})")
    p.add_argument("--similarity-weight-scale", type=float, default=1.0,
                   help="Multiplicateur du score similarité pour le pondérer comme une collab")
    p.add_argument("--add-new-nodes", action="store_true",
                   help="Ajouter les artistes Last.fm absents de SNEP comme nouveaux nœuds")
    p.add_argument("--min-snep-anchors", type=int, default=0,
                   help="Filtre d'ancrage : conserver un nouveau nœud ssi connecté à "
                        "≥ N artistes SNEP (def 0, requiert --add-new-nodes)")
    args = p.parse_args()

    # ─── Pipeline original ────────────────────────────────────────────────
    logger.info("Chargement SNEP + construction graphe original...")
    df = load_clean_df()
    G0 = build_graph(df, min_weight=args.min_collab_weight)
    logger.info(f"Graphe original : {G0.number_of_nodes()} nœuds, "
                f"{G0.number_of_edges()} arêtes")

    # ─── Enrichissement ───────────────────────────────────────────────────
    logger.info(f"Enrichissement Last.fm (min_match={args.min_match}, "
                f"add_new_nodes={args.add_new_nodes})...")
    G1 = enrich_graph_with_similarity(
        G0,
        min_match=args.min_match,
        similarity_weight_scale=args.similarity_weight_scale,
        add_new_nodes=args.add_new_nodes,
        min_snep_anchors=args.min_snep_anchors,
    )
    # garder la plus grande composante connexe (cohérence avec build_graph)
    largest_cc = max(nx.connected_components(G1), key=len)
    G1 = G1.subgraph(largest_cc).copy()

    meta = G1.graph.get("enrichment", {})
    logger.info(
        f"Graphe enrichi  : {G1.number_of_nodes()} nœuds, "
        f"{G1.number_of_edges()} arêtes "
        f"(+{meta.get('edges_added', 0)} nouvelles, "
        f"{meta.get('edges_updated', 0)} renforcées)"
    )

    # ─── Sauvegarde arêtes ────────────────────────────────────────────────
    # On marque le "type" pour traçabilité : collab vs similarity vs both
    collab_pairs = {tuple(sorted((u, v))) for u, v in G0.edges()}
    edge_rows = []
    for u, v, data in G1.edges(data=True):
        pair = tuple(sorted((u, v)))
        is_collab = pair in collab_pairs
        edge_rows.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1.0),
            "type": "collab+similar" if is_collab else "similar",
        })
    pd.DataFrame(edge_rows).to_csv(OUT_EDGES, index=False)
    logger.info(f"-> {OUT_EDGES.name} ({len(edge_rows)} arêtes)")

    # ─── Features de nœuds enrichies ─────────────────────────────────────
    logger.info("Calcul des features de nœuds enrichies...")
    # features topologiques + SNEP (du pipeline existant, recalculées sur G1)
    topo_df = build_node_features(G1, df)
    # features Last.fm
    artists = list(G1.nodes())
    lastfm_df = build_tag_features(artists)
    enriched = topo_df.merge(lastfm_df, on="artist", how="left")
    enriched.to_csv(OUT_NODES, index=False)
    logger.info(f"-> {OUT_NODES.name} ({len(enriched)} nœuds, "
                f"{len(enriched.columns)} colonnes)")

    # ─── Stats ────────────────────────────────────────────────────────────
    logger.info("─── Bilan enrichissement ─────────────────────────────────")
    logger.info(f"  Nœuds   : {G0.number_of_nodes()} -> {G1.number_of_nodes()} "
                f"(x{G1.number_of_nodes() / max(G0.number_of_nodes(), 1):.1f})")
    logger.info(f"  Arêtes  : {G0.number_of_edges()} -> {G1.number_of_edges()} "
                f"(x{G1.number_of_edges() / max(G0.number_of_edges(), 1):.1f})")
    density_old = nx.density(G0)
    density_new = nx.density(G1)
    logger.info(f"  Densité : {density_old:.4f} -> {density_new:.4f}")
    logger.info(f"  Couverture Last.fm sur les nœuds : "
                f"{enriched['has_lastfm'].sum()}/{len(enriched)} "
                f"({enriched['has_lastfm'].mean() * 100:.0f}%)")


if __name__ == "__main__":
    main()
