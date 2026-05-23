# Assignment 3 — Description et comparaison des 3 modèles

## 1. Tâche ML et pseudo-vérité terrain

**Tâche** : clustering **non supervisé** de 258 artistes en communautés musicales, à partir d'une matrice `X` de 46 dimensions (embeddings Node2Vec 32d + features Last.fm 14d).

Comme il n'existe pas de vraie "vérité terrain" pour les communautés musicales, on utilise les **communautés Louvain** comme **pseudo-vérité terrain** :
- Louvain opère directement sur le graphe (pas sur les embeddings) → vision indépendante.
- C'est l'algorithme de référence en analyse de réseaux sociaux.
- Il a trouvé **6 communautés** sur le graphe enrichi, qui correspondent qualitativement à des sous-genres reconnaissables (rap conscient, trap moderne, variété FR, etc. — voir `plots/louvain_tsne_annotated.png`).

Les 3 modèles tentent de **retrouver cette partition** en travaillant uniquement sur la géométrie des embeddings, sans accès au graphe.

---

## 2. Les 3 modèles

Tous les modèles sont configurés avec `n_clusters = k_louvain = 6` pour garantir une comparaison équitable.

### 2.1 K-Means — baseline

```python
KMeans(n_clusters=6, random_state=42, n_init=10)
```

- **Principe** : minimisation de l'inertie intra-cluster (somme des distances au carré au centroïde le plus proche). Algorithme de Lloyd.
- **Hypothèses** : clusters **sphériques**, de **variances comparables** et de **tailles équilibrées**.
- **Forces** : rapide, simple, bonnes performances quand les hypothèses sont vérifiées.
- **Faiblesses** : très sensible aux clusters allongés/non sphériques et aux déséquilibres de taille. Sur ce dataset, les communautés Louvain ont des tailles très inégales (9 à 86 nœuds) → K-Means tend à les fragmenter.

### 2.2 Gaussian Mixture Model (GMM)

```python
GaussianMixture(n_components=6, covariance_type="full", random_state=42, n_init=5)
```

- **Principe** : maximise la vraisemblance d'un mélange de gaussiennes via l'algorithme EM. Chaque cluster est une gaussienne avec sa propre moyenne et matrice de covariance.
- **Hypothèses** : la distribution sous-jacente est un mélange gaussien. Avec `covariance_type="full"`, chaque cluster peut être allongé et orienté différemment.
- **Forces** : modélise des clusters elliptiques, fournit des probabilités d'appartenance (soft assignment), bien adapté quand les clusters se chevauchent.
- **Faiblesses** : plus coûteux que K-Means, sensible à l'initialisation, peut sur-fitter avec `covariance_type="full"` en haute dimension et peu de données (cas ici : 206 train, 46 dim).

### 2.3 BIRCH — modèle retenu

```python
Birch(n_clusters=6, threshold=0.5, branching_factor=50)
```

- **Principe** : construit un arbre hiérarchique (**CFTree**) où chaque nœud résume un sous-cluster par 3 statistiques (N, somme linéaire, somme des carrés). Une étape finale d'agglomération produit les `n_clusters` finaux.
- **Hypothèses** : aucune sur la forme — clusters arbitraires.
- **Forces** : pas d'hypothèse de sphéricité, complexité quasi linéaire en `n`, gère bien les déséquilibres de taille et les structures hiérarchiques. Expose un `predict()` natif (via affectation aux sous-clusters CF).
- **Faiblesses** : sensible aux paramètres `threshold` et `branching_factor`. Sa structure de CFTree devient lourde en mémoire à très grande échelle. Le pickle nécessite `sys.setrecursionlimit(50000)` pour la sérialisation.

---

## 3. Choix des métriques

Le clustering non supervisé n'a pas d'accuracy ou de F1 classique. On utilise 3 métriques complémentaires :

| Métrique | Type | Range | Interprétation |
|---|---|---|---|
| **ARI** (Adjusted Rand Index) | externe | [-1, 1] | Concordance des **paires** entre la prédiction et Louvain, corrigée du hasard. 0 = hasard, 1 = identique. |
| **NMI** (Normalized Mutual Information) | externe | [0, 1] | Information partagée entre la prédiction et Louvain. Robuste aux différences de granularité. |
| **Silhouette** | interne | [-1, 1] | Compacité géométrique : (b - a) / max(a, b) où a = distance intra, b = distance au cluster voisin le plus proche. |

**Pourquoi ces 3 et pas l'accuracy ?**
- Le clustering ne prédit pas un label nommé mais un **indice arbitraire** : "cluster 3" du modèle peut correspondre à "cluster 5" de Louvain. ARI et NMI sont **invariants à la permutation** des labels.
- La silhouette mesure la qualité **intrinsèque** (sans label de référence) → utile pour valider que les clusters ne sont pas dégénérés.

---

## 4. Résultats sur le test set (52 nœuds)

Sortie de [`results/model_metrics.csv`](../results/model_metrics.csv) :

| Modèle | ARI | NMI | Silhouette | n_clusters |
|---|---|---|---|---|
| K-Means | 0.051 | 0.268 | −0.045 | 6 |
| GMM | 0.070 | 0.294 | −0.080 | 6 |
| **BIRCH** | **0.270** | **0.446** | **+0.013** | 6 |

**BIRCH est le meilleur sur les 3 métriques simultanément.**

### Comparaison à la baseline (sans Last.fm)

Sortie de [`results/model_metrics_baseline.csv`](../results/model_metrics_baseline.csv) :

| Modèle | ARI baseline | ARI enrichi | Gain |
|---|---|---|---|
| K-Means | 0.112 | 0.051 | -54 % |
| GMM | 0.004 | 0.070 | +1700 % |
| BIRCH | 0.112 | 0.270 | **×2.4** |

L'enrichissement Last.fm bénéficie surtout à BIRCH, qui exploite la structure non sphérique introduite par les buckets de genre.

---

## 5. Pourquoi BIRCH gagne sur ce dataset

1. **Les communautés Louvain ont des tailles très déséquilibrées** (9 à 86 nœuds) → K-Means cherche à les égaliser, fragmente artificiellement la plus grosse.
2. **Les clusters ne sont pas sphériques dans l'espace 46d** — la visualisation t-SNE (`plots/clusters_tsne_enriched.png`) montre des structures allongées et une "île" isolée. BIRCH n'impose pas de forme.
3. **La haute dimension (46 dim, 206 points) fragilise GMM** : avec `covariance_type="full"`, chaque gaussienne a 46² ≈ 2 100 paramètres de covariance → sur-ajustement probable.
4. **BIRCH a un biais structural** vers une partition hiérarchique, ce qui colle naturellement avec la structure scènes/sous-scènes du graphe musical.

---

## 6. Visualisation comparative

Le plot [`plots/clusters_tsne_enriched.png`](../plots/clusters_tsne_enriched.png) montre les 4 partitions côte à côte (Louvain truth + 3 prédictions) projetées en 2D par t-SNE.

Observations :
- Tous les modèles **détectent l'île isolée à droite** (sous-genre clairement séparé).
- **BIRCH** garde le méga-cluster central compact (n=130) ↔ proche de la communauté Louvain c5.
- **K-Means** fragmente ce méga-cluster en deux (c5 n=138 + c1 n=49) — d'où son ARI dégradé.
- **GMM** est intermédiaire.

---

## 7. Modèles sauvegardés

| Fichier | Format | Taille |
|---|---|---|
| [`models/kmeans.joblib`](../models/kmeans.joblib) | joblib | 4.6 KB |
| [`models/gmm.joblib`](../models/gmm.joblib) | joblib | 302 KB |
| [`models/birch.joblib`](../models/birch.joblib) | joblib | 738 KB |

Les modèles **baseline** (sans Last.fm) sont conservés pour la comparaison dans [`models/baseline/`](../models/baseline/).

Tous sont chargeables avec `joblib.load()` ; chacun expose `.predict(X)` qui prend une matrice (n × 46) scalée et retourne les labels de cluster.

---

## 8. Pistes d'amélioration

- **Spectral clustering** : exploite directement la matrice d'adjacence du graphe sans passer par Node2Vec. Bonne théorie pour graphes communautaires mais coût O(n³).
- **HDBSCAN** : trouve `n_clusters` automatiquement et gère les outliers (les laisse en cluster -1). Intéressant pour identifier les artistes "uniques" qui n'appartiennent vraiment à aucune scène.
- **Spectral embedding + KMeans** : utiliser les vecteurs propres du Laplacien à la place de Node2Vec.
- **Augmenter le graphe** : intégrer Spotify (genres + popularité) en plus de Last.fm — actuellement bloqué par rate-limit, donc reporté.
- **Hyperparameter tuning** : Optuna sur les hyperparamètres BIRCH (`threshold`, `branching_factor`) pour squeezer encore quelques points d'ARI.
