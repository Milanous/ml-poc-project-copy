"""
Visualisation t-SNE des embeddings enrichis, coloriée par communauté Louvain
et par cluster prédit (kmeans/gmm/birch). Sauvegarde dans plots/.

Usage : python scripts/plot_clusters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, PLOTS_DIR, MODELS  # noqa: E402
from src.model_io import load_model  # noqa: E402


def main() -> None:
    emb_df = pd.read_csv(DATA_DIR / "node_embeddings.csv")
    lab_df = pd.read_csv(DATA_DIR / "node_labels.csv")

    artists = emb_df["artist"].tolist()
    feat_cols = [c for c in emb_df.columns if c != "artist"]
    X_raw = emb_df[feat_cols].values
    X = RobustScaler().fit_transform(X_raw)
    y_louvain = lab_df.set_index("artist").loc[artists, "louvain_community"].values

    print(f"t-SNE sur {X.shape[0]} nœuds × {X.shape[1]} features...")
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, X.shape[0] // 5)),
                random_state=42, init="pca", learning_rate="auto")
    XY = tsne.fit_transform(X)

    preds = {"Louvain (vérité)": y_louvain}
    for k, cfg in MODELS.items():
        m = load_model(cfg["path"])
        preds[cfg["name"]] = m.predict(X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (title, labels) in zip(axes.flat, preds.items()):
        labels = np.asarray(labels)
        uniq = sorted(set(labels))
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(uniq), 2))
        for i, lab in enumerate(uniq):
            mask = labels == lab
            ax.scatter(XY[mask, 0], XY[mask, 1], s=18, alpha=0.75,
                       color=cmap(i), label=f"c{lab} (n={mask.sum()})",
                       edgecolors="white", linewidths=0.3)
        ax.set_title(f"{title} — {len(uniq)} clusters", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="best", framealpha=0.8)

    fig.suptitle(
        f"Clustering du graphe enrichi Last.fm "
        f"({X.shape[0]} artistes, projection t-SNE)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    out = PLOTS_DIR / "clusters_tsne_enriched.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}")

    # Plot annoté : noms des artistes SNEP les plus visibles (top-degré)
    fig2, ax = plt.subplots(figsize=(13, 10))
    uniq = sorted(set(y_louvain))
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(uniq), 2))
    for i, lab in enumerate(uniq):
        mask = y_louvain == lab
        ax.scatter(XY[mask, 0], XY[mask, 1], s=30, alpha=0.75,
                   color=cmap(i), label=f"c{lab} (n={mask.sum()})",
                   edgecolors="white", linewidths=0.4)

    # Annoter quelques noms représentatifs par communauté
    rng = np.random.default_rng(42)
    for lab in uniq:
        idxs = np.where(y_louvain == lab)[0]
        pick = rng.choice(idxs, size=min(3, len(idxs)), replace=False)
        for j in pick:
            ax.annotate(artists[j], (XY[j, 0], XY[j, 1]),
                        fontsize=7, alpha=0.85,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_title("Communautés Louvain — t-SNE annoté (3 artistes/communauté)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, loc="best", framealpha=0.85)
    fig2.tight_layout()
    out2 = PLOTS_DIR / "louvain_tsne_annotated.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"-> {out2}")


if __name__ == "__main__":
    main()
