# Assignment 4 — Plots clés (≥ 3 plots documentés)

Plus de 20 figures sont versionnées dans le dossier [`plots/`](../plots/). Cette page documente les **3 plots principaux** qui résument le projet, plus 3 plots complémentaires utilisés en EDA.

---

## Plot 1 — EDA : Top 20 artistes par nombre de certifications

**Fichier** : [`plots/top_artists.png`](../plots/top_artists.png)

**Comment il a été généré** : barplot horizontal dans [`notebooks/eda_snep.ipynb`](../notebooks/eda_snep.ipynb), `df.groupby("Interprete").size().nlargest(20).plot.barh()`.

**Ce qu'il montre** :
- **JUL domine** très largement le marché certifié français (98 certifs sur la période).
- Le top 3 (JUL, NINHO, JOHNNY HALLYDAY) couvre à lui seul ~3 % des 8 384 certifications.
- La distribution est **fortement skewed** (long tail) : la majorité des artistes n'ont qu'une seule certification.

**Décision technique tirée de ce plot** :
- Utiliser `RobustScaler` au lieu de `StandardScaler` (médiane/IQR vs moyenne/écart-type) pour limiter l'influence des outliers JUL/NINHO sur les features de degré.
- Garder ces artistes (vrai signal, pas du bruit) plutôt que de les écarter.

---

## Plot 2 — Comparaison Louvain vs 3 modèles ML (t-SNE 2D)

**Fichier** : [`plots/clusters_tsne_enriched.png`](../plots/clusters_tsne_enriched.png)

**Comment il a été généré** : `scripts/plot_clusters.py` — t-SNE 2D (`perplexity=30, random_state=42`) sur la matrice X scalée (258 × 46), puis 4 scatter plots en grille 2×2 (Louvain truth + KMeans + GMM + BIRCH), colorisés par label.

**Ce qu'il montre** :
- **Louvain (top-left)** : 6 communautés clairement séparées dans le projeté 2D, avec une **"île" isolée à droite** (sous-genre niche).
- **K-Means (top-right)** : **fragmente** le méga-cluster central rap mainstream en deux morceaux artificiels → ARI dégradé (0.05).
- **GMM (bottom-left)** : intermédiaire, mais frontières "molles" dans la zone centrale dense.
- **BIRCH (bottom-right)** : **épouse le mieux** la structure de Louvain — garde le méga-cluster compact, isole bien la même île à droite. ARI = 0.27.

**Décision technique tirée de ce plot** :
- Retenir BIRCH comme modèle principal pour la démo Streamlit.
- Justification visuelle du gain ARI quantitatif : la qualité de la partition est bien meilleure, pas juste un artefact de métrique.

---

## Plot 3 — Communautés Louvain annotées (interprétabilité)

**Fichier** : [`plots/louvain_tsne_annotated.png`](../plots/louvain_tsne_annotated.png)

**Comment il a été généré** : `scripts/plot_clusters.py` — t-SNE 2D, scatter colorisé par communauté Louvain, avec **annotations textuelles** des 3-5 artistes les plus représentatifs de chaque communauté (sélectionnés par degré pondéré le plus élevé au sein de chaque cluster).

**Ce qu'il montre** : les 6 communautés correspondent à des **sous-scènes musicales reconnaissables** :
- **c0** — rap conscient (KERY JAMES, YOUSSOUPHA, MEDINE…)
- **c1** — rap underground / club (NEKFEU, ALPHA WANN, ESCOBAR MACSON…)
- **c2** — trap moderne (NINHO, HAMZA, RK, RIM'K…)
- **c3** — drill / mainstream UK-influencé (FRESH LA DOUILLE, LE RAT LUCIANO…)
- **c4** — rap maghrébin / méditerranéen (DJADJA & DINAZ, NAZA…)
- **c5** — variété française / pop FR (JOHNNY HALLYDAY, AYA NAKAMURA, M. POKORA…)

**Décision technique tirée de ce plot** :
- Valider que Louvain produit des partitions **interprétables** (et donc utilisables comme pseudo-vérité terrain).
- Confirmation que l'enrichissement Last.fm a effectivement injecté la "couleur de genre" manquante au graphe brut SNEP — les communautés ne sont plus juste des "voisinages de featuring" mais des **scènes stylistiques cohérentes**.

---

## Plots complémentaires (EDA + feature engineering)

| Plot | Rôle |
|---|---|
| [`certif_distribution.png`](../plots/certif_distribution.png) | Distribution des niveaux de certification (Or, Platine, Diamant) — confirme l'imbalance attendu. |
| [`certifs_by_year.png`](../plots/certifs_by_year.png) | Évolution temporelle des certifications (explosion post-2015 = avènement du streaming). |
| [`fe_umap_louvain.png`](../plots/fe_umap_louvain.png) | Projection UMAP alternative à t-SNE — montre les mêmes communautés sous un autre angle. |
| [`fe_kmeans_k_selection.png`](../plots/fe_kmeans_k_selection.png) | Elbow + silhouette pour le choix de k — justifie k=6 (aligné avec Louvain). |
| [`fe_pca_variance.png`](../plots/fe_pca_variance.png) | Variance cumulée des PCA — justifie le choix de 32 dims pour Node2Vec. |
| [`subgraph_top50.png`](../plots/subgraph_top50.png) | Visualisation NetworkX du sous-graphe des 50 artistes les plus connectés. |
| [`edge_weight_distribution.png`](../plots/edge_weight_distribution.png) | Distribution des poids d'arêtes — justifie `min_weight=2`. |
| [`top_collab_artists.png`](../plots/top_collab_artists.png) | Top artistes par nombre de collaborations distinctes (vs nombre de certifs). |
| [`fe_scaler_comparison.png`](../plots/fe_scaler_comparison.png) | Comparaison StandardScaler vs RobustScaler vs MinMaxScaler — justifie RobustScaler. |

---

## Reproductibilité

Pour régénérer les plots principaux :

```bash
# Plot 1 (EDA)
jupyter nbconvert --execute notebooks/eda_snep.ipynb

# Plots 2 et 3 (clustering)
python scripts/plot_clusters.py
```

Tous les plots sont versionnés dans `plots/` et trackés par Git (≤ 200 KB chacun, pas de Git LFS nécessaire).
