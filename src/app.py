"""Streamlit dashboard — Détection de communautés musicales SNEP.

Trois onglets :
    1. Problème & EDA      — contexte business + exploration du dataset
    2. Modèles & Résultats — choix de modélisation et comparaison des 3 clusterings
    3. Démo                — sélection d'un artiste → recommandations de sa communauté
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    MODEL_METRICS_FILE,
    MODELS,
    DATA_DIR,
    PLOTS_DIR,
)


# ─── Caches ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_metrics() -> pd.DataFrame | None:
    if not MODEL_METRICS_FILE.exists():
        return None
    df = pd.read_csv(MODEL_METRICS_FILE)
    return df.drop(columns=["model_path"], errors="ignore")


@st.cache_data(show_spinner=False)
def _load_embeddings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "node_embeddings.csv")


@st.cache_data(show_spinner=False)
def _load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "node_labels.csv")


@st.cache_data(show_spinner=False)
def _load_node_features() -> pd.DataFrame:
    path = DATA_DIR / "node_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_certifications() -> pd.DataFrame | None:
    path = DATA_DIR / "snep_certifications.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


@st.cache_resource(show_spinner=False)
def _load_models() -> dict:
    loaded = {}
    for key, cfg in MODELS.items():
        path = Path(cfg["path"])
        if path.exists():
            loaded[key] = {
                "model": joblib.load(path),
                "name": cfg["name"],
                "description": cfg.get("description", ""),
            }
    return loaded


def _scaled_matrix() -> tuple[list[str], np.ndarray]:
    """Embeddings déjà scalés (RobustScaler appliqué comme à l'entraînement)."""
    from sklearn.preprocessing import RobustScaler

    emb_df = _load_embeddings()
    artists = emb_df["artist"].tolist()
    feat_cols = [c for c in emb_df.columns if c != "artist"]
    X = RobustScaler().fit_transform(emb_df[feat_cols].values)
    return artists, X


# ─── Sections ─────────────────────────────────────────────────────────────────

def _render_problem_and_eda() -> None:
    st.header("Problème & exploration des données")

    st.markdown(
        """
        ### Application business
        Construire un moteur de **recommandation d'artistes** pour une plateforme
        de streaming musical. *Si un utilisateur aime un artiste, lui recommander
        des artistes de la même **communauté musicale*** (mêmes collaborations,
        même scène, même réseau de labels).

        ### Données
        **8 384 certifications SNEP** (Or, Platine, Diamant…) scrapées depuis
        [snepmusique.com](https://snepmusique.com/les-certifications/). Chaque ligne
        décrit une certification d'un titre (artiste, label, catégorie, date).
        Les collaborations explicites (`FEAT.`, `&`) permettent de construire un
        **graphe de co-artistes pondéré**, enrichi ensuite avec l'API Last.fm.

        ### Tâche ML
        **Clustering non supervisé** des nœuds du graphe. Évaluation contre les
        communautés détectées par **Louvain** (pseudo-vérité terrain).
        """
    )

    st.divider()
    st.subheader("Exploration visuelle")

    plots = {
        "Distribution des certifications": PLOTS_DIR / "certif_distribution.png",
        "Top artistes par nombre de certifications": PLOTS_DIR / "top_artists.png",
        "Top artistes par nombre de collaborations": PLOTS_DIR / "top_collab_artists.png",
        "Distribution des poids d'arêtes (collaborations)": PLOTS_DIR / "edge_weight_distribution.png",
        "Sous-graphe des 50 artistes les plus connectés": PLOTS_DIR / "subgraph_top50.png",
    }
    available = [(title, path) for title, path in plots.items() if path.exists()]

    if not available:
        st.info("Aucun plot EDA disponible dans `plots/`.")
        return

    titles = [t for t, _ in available]
    choice = st.selectbox("Choisir un graphique", titles, key="eda_plot_select")
    selected = dict(available)[choice]
    st.image(str(selected), use_container_width=True)

    certs = _load_certifications()
    if certs is not None:
        with st.expander("Aperçu brut du dataset SNEP"):
            st.dataframe(certs.head(20), use_container_width=True)


def _render_models_and_results() -> None:
    st.header("Modèles & comparaison")

    st.markdown(
        """
        ### Pipeline de features
        1. Construction du graphe de co-artistes pondéré.
        2. **Enrichissement Last.fm** : ajout de features artiste (popularité log,
           tags de genre) + arêtes de similarité tag. Filtre d'ancrage `min_snep_anchors=2`
           pour éviter de noyer le graphe de feuilles externes.
        3. **Embeddings Node2Vec** (32 dim, SVD sur matrice de co-occurrence des walks).
        4. Concaténation **embeddings + features Last.fm = 46 dim**, `RobustScaler`.

        ### Modèles évalués
        """
    )
    for cfg in MODELS.values():
        st.markdown(f"- **{cfg['name']}** — {cfg.get('description', '')}")

    st.subheader("Métriques sur le test set")
    metrics_df = _load_metrics()
    if metrics_df is None:
        st.warning("`results/model_metrics.csv` introuvable. Lance `python scripts/main.py`.")
    else:
        st.dataframe(
            metrics_df.style.format({
                "ari": "{:.3f}",
                "nmi": "{:.3f}",
                "silhouette": "{:.3f}",
                "n_clusters": "{:.0f}",
            }).highlight_max(subset=["ari", "nmi", "silhouette"], color="#2e7d32"),
            use_container_width=True,
        )
        st.caption(
            "**ARI / NMI** : alignement avec les communautés Louvain (plus haut = meilleur). "
            "**Silhouette** : compacité géométrique des clusters dans l'espace des features."
        )

        best_row = metrics_df.loc[metrics_df["ari"].idxmax()]
        st.success(
            f"Meilleur modèle : **{best_row['model_name']}** "
            f"(ARI = {best_row['ari']:.3f}, NMI = {best_row['nmi']:.3f}, "
            f"silhouette = {best_row['silhouette']:.3f})"
        )

    st.divider()
    st.subheader("Visualisation t-SNE des clusters")
    tsne_path = PLOTS_DIR / "clusters_tsne_enriched.png"
    annot_path = PLOTS_DIR / "louvain_tsne_annotated.png"
    if tsne_path.exists():
        st.image(str(tsne_path), use_container_width=True,
                 caption="Louvain (vérité) vs prédictions des 3 modèles")
    if annot_path.exists():
        st.image(str(annot_path), use_container_width=True,
                 caption="Communautés Louvain avec quelques artistes annotés")


def _render_demo() -> None:
    st.header("Démo — Recommandation d'artistes")

    st.markdown(
        "Sélectionne un artiste : le modèle prédit sa communauté musicale "
        "et recommande d'autres artistes du même cluster, triés par proximité "
        "dans l'espace des embeddings."
    )

    models = _load_models()
    if not models:
        st.error("Aucun modèle entraîné dans `models/`.")
        return

    metrics_df = _load_metrics()
    default_key = "birch"
    if metrics_df is not None and "model_key" in metrics_df.columns:
        default_key = metrics_df.loc[metrics_df["ari"].idxmax(), "model_key"]

    model_keys = list(models.keys())
    default_idx = model_keys.index(default_key) if default_key in model_keys else 0

    col1, col2 = st.columns([2, 1])
    with col2:
        chosen_key = st.selectbox(
            "Modèle",
            model_keys,
            index=default_idx,
            format_func=lambda k: models[k]["name"],
        )
        n_reco = st.slider("Nombre de recommandations", 3, 20, 8)

    artists, X = _scaled_matrix()
    labels_df = _load_labels().set_index("artist")
    node_feats = _load_node_features()
    if not node_feats.empty and "artist" in node_feats.columns:
        node_feats = node_feats.set_index("artist")

    with col1:
        target = st.selectbox("Artiste cible", sorted(artists), key="demo_artist")

    model = models[chosen_key]["model"]
    preds = model.predict(X)
    artist_to_idx = {a: i for i, a in enumerate(artists)}

    idx = artist_to_idx[target]
    pred_cluster = int(preds[idx])
    louvain_cluster = (
        int(labels_df.loc[target, "louvain_community"])
        if target in labels_df.index else None
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Cluster prédit", f"c{pred_cluster}")
    if louvain_cluster is not None:
        m2.metric("Communauté Louvain", f"c{louvain_cluster}")
    same_cluster_idx = [i for i, p in enumerate(preds) if p == pred_cluster and i != idx]
    m3.metric("Taille du cluster", len(same_cluster_idx) + 1)

    if not same_cluster_idx:
        st.warning("Aucun autre artiste dans le cluster prédit.")
        return

    target_vec = X[idx]
    dists = np.linalg.norm(X[same_cluster_idx] - target_vec, axis=1)
    order = np.argsort(dists)[:n_reco]

    reco_rows = []
    for j in order:
        artist_idx = same_cluster_idx[j]
        artist_name = artists[artist_idx]
        row = {
            "Artiste": artist_name,
            "Distance": float(dists[j]),
        }
        if not node_feats.empty and artist_name in node_feats.index:
            for col in ("total_certifs", "degree", "weighted_degree"):
                if col in node_feats.columns:
                    row[col] = node_feats.loc[artist_name, col]
        reco_rows.append(row)

    st.subheader(f"Top {len(reco_rows)} artistes recommandés")
    st.dataframe(
        pd.DataFrame(reco_rows).style.format({
            "Distance": "{:.3f}",
            "total_certifs": "{:.0f}",
            "degree": "{:.0f}",
            "weighted_degree": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

def build_app() -> None:
    st.set_page_config(
        page_title="SNEP — Communautés musicales",
        layout="wide",
    )

    st.title("Détection de communautés d'artistes — SNEP")
    st.caption(
        "Clustering non supervisé sur un graphe de co-artistes enrichi Last.fm. "
        "Sources : 8 384 certifications SNEP + API Last.fm."
    )

    tab1, tab2, tab3 = st.tabs([
        "1. Problème & EDA",
        "2. Modèles & résultats",
        "3. Démo",
    ])
    with tab1:
        _render_problem_and_eda()
    with tab2:
        _render_models_and_results()
    with tab3:
        _render_demo()


if __name__ == "__main__":
    build_app()
