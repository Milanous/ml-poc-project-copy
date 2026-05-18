"""Student-owned metrics contract.

Tâche ML : clustering non supervisé des communautés musicales dans le graphe de
co-artistes SNEP.

Métriques retenues
------------------
- **ARI** (Adjusted Rand Index) : mesure l'accord entre les labels prédits et les
  labels Louvain de référence (corrigé du hasard, ∈ [-1, 1], 1 = parfait).
  Métrique principale de comparaison entre les 3 modèles.
- **NMI** (Normalized Mutual Information) : information partagée entre les deux
  partitions (∈ [0, 1], robuste au nombre de clusters différent).
- **Silhouette** : cohésion intra-cluster / séparation inter-cluster dans
  l'espace des embeddings Node2Vec (∈ [-1, 1], > 0.2 = acceptable).
  Chargée depuis les embeddings mis en cache ; ignorée silencieusement si
  le fichier est absent.
- **n_clusters** : nombre de clusters distincts prédits (hors bruit DBSCAN=-1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_NODE_EMB_CSV = Path(__file__).parent.parent / "data" / "node_embeddings.csv"


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Parameters
    ----------
    y_true : array-like of int
        Labels de référence Louvain (pseudo-vérité terrain),
        typiquement un ``pd.Series`` avec les indices de nœuds préservés.
    y_pred : array-like of int
        Labels prédits par le modèle évalué.
        La valeur -1 est interprétée comme du bruit (DBSCAN).

    Returns
    -------
    dict[str, float]
        ``{"ari": ..., "nmi": ..., "silhouette": ..., "n_clusters": ...}``
        ``silhouette`` est omis si les embeddings en cache sont introuvables.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    result: dict[str, float] = {
        "ari":        float(adjusted_rand_score(y_true_arr, y_pred_arr)),
        "nmi":        float(normalized_mutual_info_score(
                          y_true_arr, y_pred_arr, average_method="arithmetic"
                      )),
        "n_clusters": float(len(set(y_pred_arr.tolist()) - {-1})),
    }

    # Silhouette nécessite les features X — chargées depuis le cache si disponible
    try:
        import pandas as pd

        emb_df   = pd.read_csv(_NODE_EMB_CSV)
        emb_cols = [c for c in emb_df.columns if c != "artist"]
        X_all    = emb_df[emb_cols].values

        # Récupérer les lignes correspondant aux nœuds du jeu de test
        idx   = list(y_true.index) if hasattr(y_true, "index") else list(range(len(y_true_arr)))
        X_sub = X_all[idx]

        # Exclure les points de bruit (-1) pour le calcul
        mask = y_pred_arr != -1
        if mask.sum() >= 2 and len(set(y_pred_arr[mask].tolist())) >= 2:
            result["silhouette"] = float(silhouette_score(X_sub[mask], y_pred_arr[mask]))
    except Exception:
        pass

    return result
