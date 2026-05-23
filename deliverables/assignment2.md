# Assignment 2 — Feature Engineering & Dataset préprocessé

## 1. Vue d'ensemble du pipeline

Le dataset brut (`data/snep_certifications.csv`, 8 384 lignes) est une **liste de certifications** : chaque ligne décrit un titre certifié et son artiste (ou ses artistes en cas de featuring). Ce n'est **pas directement exploitable** par un algorithme de ML — il faut le transformer en une matrice numérique `(n_artistes, n_features)`.

Le pipeline complet, implémenté dans [`src/data.py`](../src/data.py), suit ces étapes :

```
CSV brut SNEP
   │ (1) nettoyage + parsing des featurings
   ▼
DataFrame structuré (artists_list, cert_level, …)
   │ (2) construction du graphe de co-artistes pondéré
   ▼
nx.Graph (1086 nœuds, 3300 arêtes)
   │ (3) enrichissement Last.fm (features + arêtes de similarité)
   ▼
nx.Graph enrichi (258 nœuds après filtre d'ancrage, 91% coverage Last.fm)
   │ (4) features tabulaires + embeddings Node2Vec
   ▼
X = [embeddings_32d | lastfm_14d] = matrice (258 × 46)
   │ (5) RobustScaler + train/test split
   ▼
(X_train, X_test, y_train, y_test) — y = communautés Louvain
```

---

## 2. Détail des transformations

### 2.1 Nettoyage du CSV (`load_clean_df`)

- Normalisation des noms d'artistes : majuscules, suppression des diacritiques, espaces normalisés (ex: `"Aya Nakamura"` → `"AYA NAKAMURA"`).
- Parsing des dates `Date de sortie` / `Date de constat`, calcul de `jours_obtention`.
- Mapping `Certification` → `cert_level` ordinal (Or=0, Double Or=1, Platine=2, …, Quadruple Diamant=8).
- Suppression des lignes avec `Interprete`, `cert_level` ou `Categorie_norm` manquants.
- **Découpage des featurings** : la fonction `_split_artists()` casse `"JUL FEAT. NINHO & SCH"` en `["JUL", "NINHO", "SCH"]` via des regex sur `FEAT.`, `&`, `,`. Les vrais duos (ex: PNL, BIGFLO & OLI) sont détectés au préalable par `_build_known_duos()` pour ne pas être cassés à tort.

### 2.2 Construction du graphe (`build_graph`)

- Chaque titre avec ≥ 2 artistes génère toutes les paires `(A, B)` via `itertools.combinations`.
- Le poids d'une arête `(A, B)` = nombre de titres certifiés où A et B apparaissent ensemble.
- **Filtre `min_weight=2`** : on retire les featurings ponctuels (un seul titre commun) pour réduire le bruit.
- On ne conserve que la **plus grande composante connexe** du graphe (les artistes isolés sont écartés).

### 2.3 Features tabulaires de nœuds (`build_node_features`)

8 features par artiste sont calculées via `pandas.groupby(artiste_principal).agg(...)` :

| Feature | Description |
|---|---|
| `degree` | nombre de collaborateurs uniques |
| `weighted_degree` | somme des poids d'arêtes incidentes (intensité de collab) |
| `total_certifs` | nombre total de certifications |
| `max_cert_level` | plus haute certification atteinte (0–8) |
| `avg_cert_level` | niveau moyen des certifications |
| `collab_rate` | proportion de titres en featuring |
| `singles_rate` | proportion de singles vs albums |
| `annee_median` | année médiane des sorties |

Ces features ne sont pas utilisées telles quelles dans le modèle final, mais servent dans le notebook d'EDA et pour les recommandations dans l'app Streamlit.

### 2.4 Enrichissement Last.fm ([`src/enrichment.py`](../src/enrichment.py))

Le SNEP donne seulement la **structure de collaboration**, pas le **genre** ni la **popularité fine**. Pour pallier ça, on appelle l'API Last.fm pour chaque artiste et on ajoute :

**14 features par nœud :**
- `log_listeners`, `log_playcount` : popularité log (compresse les ordres de grandeur)
- `n_tags`, `has_lastfm` : indicateurs de couverture
- **10 buckets de genre one-hot** (agrégés depuis les 100 top-tags Last.fm) : `genre_rap_hiphop`, `genre_pop`, `genre_rock`, `genre_electronic`, `genre_rnb_soul`, `genre_chanson_fr`, `genre_metal`, `genre_jazz`, `genre_reggae_latin`, `genre_classical`

**Arêtes de similarité Last.fm** : si deux artistes partagent ≥ 40 % de leurs top-tags (Jaccard sur sets), on ajoute une arête `similarity` pondérée plus faiblement que les featurings réels (scale `similarity_weight_scale`).

**Filtre d'ancrage `min_snep_anchors=2`** : un artiste Last.fm non-SNEP n'est gardé dans le graphe que s'il est connecté à au moins 2 artistes SNEP. Sans ce filtre, le graphe explose à 9 000+ feuilles dégénérées (silhouette artificiellement haute, ARI proche de 0).

Cette config (`anchor=2, min_match=0.4`) a été retenue après un grid search 12 configurations dans [`scripts/grid_search_enrichment.py`](../scripts/grid_search_enrichment.py) (sortie : [`results/grid_search_enrichment.csv`](../results/grid_search_enrichment.csv)). Résultat final : **258 nœuds, 3 313 arêtes, 91 % de couverture Last.fm**.

### 2.5 Embeddings Node2Vec (`build_node2vec_embeddings`)

Le graphe brut n'est pas exploitable directement par scikit-learn — il faut transformer chaque nœud en un **vecteur dense**. C'est le rôle de Node2Vec :

1. **Random walks biaisés** depuis chaque nœud (paramètres `p, q` qui contrôlent BFS vs DFS, `num_walks` walks par nœud, longueur `walk_length`).
2. **Matrice de co-occurrence** : on glisse une fenêtre `window` sur chaque walk et on incrémente `cooc[i, j]` pour tous les couples proches.
3. **TruncatedSVD** sur cette matrice → vecteur dense de **32 dimensions** par artiste.

Intuition : deux artistes qui apparaissent souvent dans les mêmes walks (donc proches dans le graphe, même sans arête directe) auront des vecteurs Euclidiens proches.

### 2.6 Concaténation finale + scaling

```python
X = [ node2vec_32d  |  lastfm_14d ]   # shape (258, 46)
X = RobustScaler().fit_transform(X)
```

`RobustScaler` (médiane / IQR) plutôt que `StandardScaler` : il est plus robuste aux outliers de degré (JUL et NINHO ont des degrés bien plus élevés que la médiane).

### 2.7 Labels & train/test split

Les "labels" sont les **communautés Louvain** calculées sur le graphe pondéré (pseudo-vérité terrain, voir Assignment 3 pour la justification) :

```python
partition = community_louvain.best_partition(G, weight="weight", random_state=42)
y = pd.Series([partition[n] for n in nodes])
```

Louvain trouve **6 communautés** sur le graphe enrichi.

Split final : `train_test_split(X, y, test_size=0.2, random_state=42)` → **206 train / 52 test**.

---

## 3. Choix d'encodage justifiés

| Type de feature | Encodage choisi | Raison |
|---|---|---|
| Embeddings Node2Vec (continues) | `RobustScaler` | Distributions à queue longue (degrés), médianes plus stables que la moyenne. |
| `log_listeners`, `log_playcount` | log + `RobustScaler` | Listeners varient de 10² à 10⁸ → log compresse. |
| `n_tags`, `has_lastfm` | brut + scaler | Ranges petits, intégrés au scaling global. |
| 10 buckets de genre | one-hot binaire | Information catégorielle multi-label (un artiste peut être à la fois rap et pop). |
| `cert_level` (non utilisé dans X final) | Ordinal | Hiérarchie naturelle Or < Platine < Diamant. |

Pas de **target encoding** (data leakage) ni de catégorielle haute cardinalité à one-hot encoder.

---

## 4. EDA — vérifications de qualité

Toutes les vérifications faites dans [`notebooks/eda_snep.ipynb`](../notebooks/eda_snep.ipynb) :

- **Données manquantes** : 72 dates de sortie (0.9 %), 531 durées d'obtention (6.3 % — colonne supprimée ensuite). Aucune feature avec > 10 % de manquants après nettoyage.
- **Outliers** : JUL (98 certifs), NINHO (65), JOHNNY HALLYDAY (61) — gardés volontairement (signal informatif, pas du bruit). `RobustScaler` gère leur influence.
- **Class imbalance** : non applicable au clustering non supervisé, mais on note que la communauté c5 (rap mainstream) contient 86 nœuds alors que c1 (rap underground) n'en a que 9 → tailles déséquilibrées, ce qui pénalise K-Means qui suppose des clusters de taille comparable.
- **Feature drift temporel** : non détecté — la plupart des features sont des agrégats sur toute la période et pas des séries temporelles.

---

## 5. Fichiers générés en sortie

| Fichier | Contenu | Usage |
|---|---|---|
| `data/graph_edges.csv` | arêtes du graphe enrichi (source, target, weight) | reproductibilité + démo |
| `data/node_features.csv` | 8 features tabulaires par artiste | EDA + démo Streamlit |
| `data/node_embeddings.csv` | matrice 32 dims Node2Vec + 14 dims Last.fm | input des modèles |
| `data/node_labels.csv` | communautés Louvain | pseudo-vérité terrain |
| `data/lastfm_artist_features.csv` | brut API Last.fm | cache pour ne pas re-appeler l'API |
| `data/lastfm_similar_edges.csv` | similarités tag Jaccard | construction du graphe enrichi |

Ces fichiers sont **versionnés sur Git** : il suffit de cloner le repo pour pouvoir lancer `scripts/main.py` sans refaire le pipeline complet.

---

## 6. Reproductibilité

Toutes les étapes aléatoires sont seedées avec `random_state=42` :
- `train_test_split` (split déterministe)
- Walks Node2Vec (`random.seed(42)` dans les walks biaisés)
- Louvain (`best_partition(..., random_state=42)`)
- TruncatedSVD (`random_state=42`)

Pour régénérer entièrement le dataset préprocessé :

```bash
python scripts/scrape_snep.py             # 1. récupération CSV brut
python scripts/fetch_lastfm_features.py   # 2. enrichissement Last.fm (API)
python scripts/build_enriched_graph.py --add-new-nodes --min-snep-anchors 2 --min-match 0.4
                                          # 3. construction graphe + embeddings
python scripts/train_enriched.py          # 4. entraînement des modèles
```
