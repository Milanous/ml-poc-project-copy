# Assignement 1 - Cadrage du projet

## 1. Description du dataset

Le dataset utilisé est constitué de **8 384 certifications musicales** officielles publiées par le **SNEP** (Syndicat National de l'Édition Phonographique), scraped depuis [snepmusique.com/les-certifications](https://snepmusique.com/les-certifications/).

### Structure du dataset

Chaque ligne représente une certification obtenue par un titre musical en France :

| Colonne | Description |
|---|---|
| `Interprete` | Nom de l'artiste ou du groupe (peut contenir plusieurs artistes : "FEAT.", "&") |
| `Titre` | Titre du single ou de l'album certifié |
| `Editeur / Distributeur` | Label(s) et distributeur(s) associés |
| `Categorie` | Type de produit : Singles, Albums, Vidéos |
| `Certification` | Niveau de certification : Or, Platine, Diamant, Double Platine, etc. |
| `Date de sortie` | Date de publication du titre (format DD/MM/YYYY) |
| `Date de constat` | Date à laquelle la certification a été officiellement constatée |
| `Duree obtention` | Durée entre la sortie et l'obtention de la certification |

### Statistiques clés

- **8 384 certifications** au total (après déduplication)
- **Certifications** : Or (57%), Platine (20%), Diamant (12%), Double Or (5%), etc.
- **Catégories** : Singles (~47%), Albums (~45%), Vidéos (~6%)
- **~1 078 entrées** impliquant plusieurs artistes (collaborations "FEAT." ou "&")
- **Valeurs manquantes** : faibles (72 dates de sortie manquantes, 531 durées manquantes)
- **Artistes prolifiques** : JUL (98 certifications), NINHO (65), JOHNNY HALLYDAY (61), DAMSO (58), CÉLINE DION (53)

---

## 2. Méthode de collecte

Les données ont été collectées par **web scraping** du site officiel du SNEP.

### Approche technique

Le site organise les certifications en **378 pages** paginées. La stratégie de scraping adoptée tient compte de la structure cumulative du site :

- **Pages 1–189** (certifications récentes) : chaque page affiche les 30 dernières certifications ajoutées ; on extrait les 30 derniers éléments de chaque page pour éviter les doublons entre pages consécutives.
- **Pages 190–378** (certifications plus anciennes) : les items ne se chevauchent pas entre pages ; on extrait la totalité des certifications de chaque page.
- **Déduplication globale** par clé composite `(Interprete, Titre, Categorie, Certification)`.

### Outils utilisés

- `requests` pour les requêtes HTTP avec gestion des retry (3 tentatives par page)
- `BeautifulSoup` (lxml) pour le parsing HTML
- `re` (regex) pour l'extraction rapide des blocs `<div class="certification">`
- `ThreadPoolExecutor` (6 workers) pour le scraping parallèle des 378 pages
- Export en CSV encodé UTF-8 BOM pour compatibilité

Le code de scraping est disponible dans `src/scraping.py`.

---

## 3. Justification du choix du dataset

### Adéquation avec l'objectif business

Une plateforme de streaming musical a besoin de recommander des artistes à ses utilisateurs. Les certifications SNEP constituent un **proxy de popularité officiel et fiable** : elles traduisent le volume de streams, ventes et téléchargements d'un titre en France, validé par l'industrie.

### Richesse des relations entre artistes

Le champ `Interprete` contient de nombreuses **collaborations explicites** ("FEAT.", "&", featuring) qui permettent de construire un **graphe de co-artistes** : deux artistes sont connectés s'ils ont collaboré sur au moins un titre certifié. Ce graphe est directement exploitable pour un algorithme de détection de communautés.

### Qualité et accessibilité

- Données **officielles** (source industrielle reconnue), sans biais de plateforme
- Données **publiques** et librement accessibles
- Couvre une large période temporelle (certifications des années 1970 à aujourd'hui)
- Granularité suffisante pour construire un graphe dense avec des artistes francophones et internationaux

---

## 4. Objectif business et ML

### Objectif business

Construire un moteur de **recommandation d'artistes** pour une plateforme de streaming musicale. L'idée : si un utilisateur apprécie un artiste donné, lui recommander des artistes issus de la même **communauté musicale** (même genre, mêmes collaborations, même réseau de labels).

### Approche ML

**Étape 1 – Construction du graphe de co-artistes**  
- Nœuds : artistes uniques  
- Arêtes : une arête entre deux artistes s'ils apparaissent ensemble sur un titre certifié (feat, duo, etc.)  
- Poids possibles : nombre de collaborations, niveau de certification moyen  

**Étape 2 – Détection de communautés (Louvain)**  
- Application de l'**algorithme de Louvain** (maximisation de la modularité) pour identifier des clusters d'artistes partageant un réseau dense de collaborations  
- Ces communautés correspondent intuitivement à des genres ou scènes musicales (rap FR, variété, électro, etc.)  

**Étape 3 – Recommandation**  
- Étant donné un artiste cible, recommander les artistes du même cluster, triés par centralité ou nombre de certifications  

---

## 5. Métriques et objectifs d'évaluation envisagés

### Métriques de qualité du clustering

| Métrique | Description |
|---|---|
| **Modularité (Q)** | Mesure la densité des connexions intra-cluster vs inter-cluster (Louvain optimise directement cette mesure). Valeur cible : Q > 0.3 |
| **Silhouette Score** | Mesure la cohésion intra-cluster et la séparation inter-cluster. Calculé sur une représentation vectorielle des nœuds (embeddings Node2Vec ou features agrégées). Valeur cible : > 0.2 |
| **Davies-Bouldin Index** | Rapport entre dispersion intra-cluster et distance inter-cluster. Plus faible = meilleur. |
| **Conductance** | Fraction d'arêtes sortant d'un cluster (bas = cluster bien isolé) |

### Évaluation qualitative

- **Cohérence sémantique** des clusters : vérification manuelle que les artistes regroupés partagent le même genre musical
- **Couverture** : proportion d'artistes avec au moins un voisin dans le graphe

### Baseline simple

Un recommandeur naïf basé uniquement sur le label/distributeur servira de baseline pour comparer la qualité des recommandations issues des communautés Louvain.

---

## 6. Sélection des modèles

### 6.1 Type de problème

**Clustering non supervisé** sur un graphe de co-artistes.

- Pas de variable cible fournie a priori (pas de genre musical étiqueté, pas de "bonne" communauté connue)
- La tâche est de regrouper les artistes en **communautés cohérentes** basées sur leurs collaborations certifiées
- Les labels Louvain (Q = 0.679) servent de **pseudo-vérité terrain** pour évaluer les modèles d'embedding

### 6.2 Métrique d'évaluation principale

**ARI (Adjusted Rand Index)** — mesure l'accord entre la partition prédite et la partition Louvain de référence.

| Propriété | Valeur |
|---|---|
| Plage | [-1, 1] |
| Référence | 1.0 = accord parfait, 0 = aléatoire, < 0 = pire que le hasard |
| Avantage | Corrigé du hasard, robuste aux différences de taille et de nombre de clusters |
| Limitation | Ne mesure pas la qualité intrinsèque des clusters (besoin de y_true) |

**Métriques complémentaires :**

- **NMI** (Normalized Mutual Information) — information partagée entre les deux partitions, robuste aux déséquilibres de taille
- **Silhouette** — cohésion intra-cluster vs séparation inter-cluster dans l'espace des embeddings Node2Vec (ne nécessite pas y_true, mesure la qualité géométrique)

> Louvain (algorithme de graphe) est utilisé comme **référence**, pas comme "modèle évalué". Les 3 modèles ci-dessous opèrent sur les embeddings Node2Vec et sont comparés à Louvain via ARI/NMI.

### 6.3 Les trois modèles

#### Modèle 1 — K-Means (k=8)

**Principe :** partitionne les données en k clusters en minimisant la somme des distances euclidiennes aux centroides (algorithme EM, convergence garantie vers un minimum local).

| | |
|---|---|
| **Avantages** | Simple, scalable O(nkt), centroides interprétables, reproductible |
| **Limites** | Suppose des clusters **sphériques et de variance homogène** ; k doit être fixé a priori ; sensible à l'initialisation et aux outliers |
| **Adéquation** | Baseline naturel pour tout espace euclidien — les embeddings Node2Vec (SVD) vivent dans ℝ³². Sert de **point de comparaison minimal** : tout autre modèle doit le surpasser |
| **Résultats (55 nœuds)** | ARI=0.25  NMI=0.58  Sil=0.10  k_effectif=5/8 |

> **Pourquoi pas K-Means seul ?** K-Means suppose des clusters sphériques. Les embeddings SVD produisent souvent des clusters allongés (ellipsoïdaux). De plus, avec un graphe clairsemé (55 nœuds, densité 4.3 %), les embeddings n'ont pas assez de structure pour que K-Means trouve les 8 communautés fines identifiées par Louvain. K-Means converge vers 5 macro-groupes au lieu de 8.

---

#### Modèle 2 — Gaussian Mixture Model — GMM (k=8)

**Principe :** modélise chaque cluster comme une distribution gaussienne multivariée. L'algorithme EM maximise la log-vraisemblance. Chaque point reçoit une **probabilité d'appartenance** à chaque composante.

| | |
|---|---|
| **Avantages** | Gère les clusters **elliptiques** (covariance pleine) ; assignement souple (soft) ; peut modéliser l'incertitude d'appartenance ; sélection de k via BIC/AIC |
| **Limites** | Suppose que chaque cluster suit une gaussienne multivariée ; k doit être fixé a priori ; peut sur-paramétrer en haute dimension (covariance pleine = d² paramètres par composante) ; convergence vers minimum local |
| **Adéquation** | Les projections SVD s'approchent de distributions gaussiennes (transformation linéaire). Un artiste "au croisement de deux scènes" (ex: Gims entre RnB et rap) bénéficie de l'assignement probabiliste. **Alternative à K-Means** pour les clusters non-sphériques |
| **Résultats (55 nœuds)** | ARI=0.07  NMI=0.27  Sil=0.20  k_effectif=2/8 |

> **Analyse :** GMM obtient la **meilleure silhouette** (0.20) mais le **plus faible ARI** (0.07). Il détecte 2 macro-groupes très bien séparés géométriquement, mais ne retrouve pas les 8 communautés fines de Louvain. Cela montre que la structure gaussienne des embeddings est pauvre pour ce graphe clairsemé.

---

#### Modèle 3 — BIRCH (Balanced Iterative Reducing and Clustering, k=8)

**Principe :** construit d'abord un **arbre de features de clustering** (CFT) — une structure hiérarchique de micro-clusters — puis applique un algorithme de merging (K-Means agglomératif) sur les feuilles pour obtenir k clusters finaux. Possède un `predict()` natif.

| | |
|---|---|
| **Avantages** | Complexité **O(n)** (un seul passage sur les données) ; ne suppose pas de forme sphérique pour les micro-clusters ; `predict()` natif (contrairement à AgglomerativeClustering) ; moins sensible aux outliers que K-Means |
| **Limites** | Paramètre `threshold` (rayon des micro-clusters) sensible ; moins précis que K-Means/GMM sur petits datasets ; la phase de merging finale utilise K-Means (héritage des limites sphériques) |
| **Adéquation** | Représente l'approche **hiérarchique** dans la comparaison (vs centroïde K-Means, vs probabiliste GMM). Utile pour valider que la structure hiérarchique des collaborations n'est pas mieux capturée que par K-Means simple |
| **Résultats (55 nœuds)** | ARI=0.15  NMI=0.48  Sil=0.07  k_effectif=5/8 |

---

### 6.4 Tableau de comparaison global

| Méthode | Paradigme | ARI ↑ | NMI ↑ | Silhouette ↑ | k effectif | predict() natif |
|---|---|---|---|---|---|---|
| **Louvain** (référence) | Graphe (modularité) | — | — | **0.25** | **8** | Non |
| KMeans (k=8) | Centroïde | **0.25** | **0.58** | 0.10 | 5 | Oui |
| GMM (k=8) | Probabiliste | 0.07 | 0.27 | 0.20 | 2 | Oui |
| BIRCH (k=8) | Hiérarchique | 0.15 | 0.48 | 0.07 | 5 | Oui |

### 6.5 Protocole de comparaison

1. **Données** : embeddings Node2Vec (32 dims, TruncatedSVD), scalés par `RobustScaler`
2. **Split** : 80 % entraînement (44 nœuds) / 20 % test (11 nœuds), graine 42
3. **Entraînement** : chaque modèle est fitté sur `X_train` via `model.fit(X_train)`
4. **Évaluation** : `model.predict(X_test)` → `compute_metrics(y_test, y_pred)` retourne ARI, NMI, Silhouette
5. **Référence** : les labels `y_test` sont les communautés Louvain calculées sur le graphe complet
6. **Reproductibilité** : `random_state=42` pour tous les modèles stochastiques (KMeans, GMM)

**Résultat principal :** KMeans est le meilleur des 3 modèles embedding (ARI=0.25, NMI=0.58), mais aucun ne retrouve les 8 communautés fines de Louvain (k_effectif ≤ 5). Ce résultat valide le choix de Louvain comme méthode primaire : la structure communautaire fine n'est pas capturée par les embeddings SVD avec un graphe aussi clairsemé (55 nœuds, densité 4.3 %, dim=32).