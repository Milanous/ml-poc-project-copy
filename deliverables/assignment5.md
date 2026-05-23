# Assignment 5 — Application Streamlit (`src/app.py`)

## 1. Vue d'ensemble

Le fichier [`src/app.py`](../src/app.py) implémente une **dashboard Streamlit** qui rend le projet **interactif** et **démonstrable**. C'est l'interface utilisateur finale qui expose les 3 modèles entraînés et permet :

1. d'explorer le problème métier et les données (EDA),
2. de comparer visuellement les 3 modèles côte à côte,
3. de tester une **démo de recommandation** : sélection d'un artiste → prédiction de son cluster → top-N voisins recommandés.

## 2. Architecture en 3 onglets

L'app est organisée par `st.tabs([...])` en 3 sections :

### Onglet 1 — "Problem & EDA"

Fonction : `_render_problem_and_eda()`.

Contenu :
- Présentation textuelle du **problème métier** (recommandation d'artistes à partir des certifications SNEP) et de la motivation Last.fm.
- Tableau récapitulatif du **dataset** : nombre de nœuds, arêtes, couverture Last.fm.
- **Sélecteur déroulant** pour afficher au choix l'un des plots EDA (top artistes, distribution des certifs, évolution temporelle, etc.) → chargés dynamiquement depuis `plots/` avec `st.image()`.
- Affichage de la **distribution des niveaux de certification** et de la **distribution des tailles de communautés Louvain**.

### Onglet 2 — "Models & Results"

Fonction : `_render_models_and_results()`.

Contenu :
- **Tableau comparatif** des 3 modèles (ARI / NMI / silhouette / n_clusters) chargé depuis [`results/model_metrics.csv`](../results/model_metrics.csv) — affiché avec `st.dataframe(..., use_container_width=True)`.
- **Comparaison baseline vs enrichi** : second tableau qui montre le gain d'ARI grâce à l'enrichissement Last.fm.
- **Visualisations t-SNE** : affichage des plots `clusters_tsne_enriched.png` (4 partitions) et `louvain_tsne_annotated.png` (communautés annotées).
- Annonce du **modèle retenu** (BIRCH) avec justification synthétique.

### Onglet 3 — "Demo"

Fonction : `_render_demo()`. **Cœur de la démo interactive.**

Workflow :
1. **Sélecteur de modèle** : `st.radio(["KMeans", "GMM", "BIRCH"])` — charge le `.joblib` correspondant avec `joblib.load(MODELS[choice]["path"])`.
2. **Sélecteur d'artiste** : `st.selectbox()` listant les 258 nœuds (triés alphabétiquement).
3. À la sélection :
   - Récupération du vecteur 46d de l'artiste : `X.loc[artist]`.
   - **Prédiction du cluster** via `model.predict(X.loc[[artist]])`.
   - Affichage du **cluster prédit** + de la taille du cluster + des 5 artistes les plus représentatifs (par degré pondéré au sein du cluster).
   - **Top-N recommandations** : calcul des distances euclidiennes entre `X.loc[artist]` et tous les autres nœuds → tri ascendant → top-10. Affichage en tableau avec : nom, cluster prédit, distance, total_certifs, genre Last.fm dominant.
4. **Comparaison entre modèles** : changer le `st.radio` met à jour la prédiction et les recommandations en temps réel sans recharger le dataset (cache `@st.cache_data`).

## 3. Choix techniques

### Caching
```python
@st.cache_data
def _load_dataset():
    return load_dataset_full()  # depuis src/data.py
```
Évite de recharger les 258 nœuds × 46 features à chaque interaction. `cache_data` est invalidé automatiquement si les fichiers source changent.

### Chargement des modèles
```python
@st.cache_resource
def _load_model(path: str):
    return joblib.load(path)
```
`cache_resource` (pas `cache_data`) parce que les modèles scikit-learn ne sont pas sérialisables hash-able naturellement.

### Recommandation par distance
Le top-N voisins est calculé en **espace embeddings (Node2Vec 32d + Last.fm 14d)**, pas en espace graphe brut. Conséquence : on peut recommander des artistes **sans collaboration directe** avec la cible, dès lors qu'ils sont "proches stylistiquement" via Last.fm. C'est une recommandation **collaborative + content-based hybride**.

### Configuration centralisée
La liste des modèles, leurs chemins joblib et leurs descriptions vivent dans [`src/config.py`](../src/config.py) :
```python
MODELS = {
    "kmeans": {"name": "K-Means", "description": "...", "path": "models/kmeans.joblib"},
    "gmm":    {"name": "Gaussian Mixture", ...},
    "birch":  {"name": "BIRCH", ...},
}
```
Ajouter un 4e modèle = ajouter une entrée + un `.joblib`. L'app le détecte automatiquement.

## 4. Lancement de l'app

Deux modes :

**Mode 1 — direct** :
```bash
streamlit run src/app.py
```

**Mode 2 — orchestré** (recommandé pour la démo) :
```bash
python scripts/main.py
```
Le script `scripts/main.py` :
1. recharge les modèles depuis `models/*.joblib`,
2. ré-évalue chaque modèle sur le test set,
3. écrit `results/model_metrics.csv`,
4. lance Streamlit en sous-process.

Tout est fait pour qu'**une seule commande** suffise à une démonstration live.

## 5. Test de fonctionnement

L'app a été testée en local : démarre sur le port 8501, répond en HTTP 200, et toutes les interactions (sélecteurs, prédictions, recommandations) fonctionnent. Les 3 modèles produisent des recommandations cohérentes — par exemple, sélectionner NEKFEU avec BIRCH retourne des artistes du même sous-genre rap underground (ALPHA WANN, S.PRI NOIR, ...).

## 6. Limitations connues

- **Pas d'auth** : l'app est en local uniquement. Pour un déploiement Streamlit Cloud, prévoir `st.secrets` pour la clé Last.fm.
- **Pas de cold start** : seul un artiste **présent dans le graphe** peut être analysé. Un nouvel artiste hors SNEP demanderait un appel Last.fm à la volée + re-projection Node2Vec → reporté.
- **Recommandations dans le même cluster** : par construction, les top voisins sont en majorité du même cluster prédit (puisque la distance est calculée en espace embeddings). C'est cohérent métier mais limite la sérendipité.

## 7. Captures (référence dans le repo)

L'app est lancée en live pendant la soutenance ; les figures statiques de référence sont dans `plots/`. Pour une capture d'écran statique de la dashboard, voir `plots/clusters_tsne_enriched.png` qui correspond au visuel principal de l'onglet 2.
