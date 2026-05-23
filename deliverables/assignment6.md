# Assignment 6 — Modifications du README

## 1. Objectif

Le [`README.md`](../README.md) à la racine du projet a été **complètement réécrit** par rapport au template initial, pour répondre à deux besoins :

1. **Présentation rapide du projet** au tout début (quoi, pourquoi, résultats clés) → un visiteur GitHub comprend en 30 secondes.
2. **Guide de récupération des données** en fin de README → un évaluateur peut reproduire le pipeline complet depuis zéro.

## 2. Structure finale du README

Le README est intégralement en **anglais** (cohérent avec l'usage GitHub international) et organisé en sections :

### Section 1 — Project Overview (ajoutée)

Bref pitch (3-4 lignes) :
> "This project performs **community detection on the French music certifications graph (SNEP)** enriched with Last.fm metadata. It clusters 258 artists into 6 stylistic scenes (rap conscient, trap, variété FR, …) and exposes a Streamlit dashboard for live exploration and recommendation."

Inclut un **schéma ASCII du pipeline** : Scraping → Graph → Enrichment → Embeddings → Clustering → App.

### Section 2 — Results highlight (ajoutée)

Tableau récap : ARI / NMI / silhouette par modèle, avec BIRCH en gras. Pointe vers `plots/clusters_tsne_enriched.png` pour la preuve visuelle.

### Section 3 — Repository structure

Arborescence commentée (data/, models/, src/, scripts/, notebooks/, plots/, results/, deliverables/) — un visiteur sait immédiatement où trouver le code, les données et les livrables.

### Section 4 — Installation

```bash
git clone <repo>
cd ml-poc-project-copy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Mention de la version Python 3.11 et du fait que `python-louvain==0.16` est listé dans requirements.

### Section 5 — Quick start

Une seule commande pour lancer la démo si les artefacts sont déjà committed :
```bash
python scripts/main.py
```
→ ré-évalue les modèles + lance Streamlit sur port 8501.

### Section 6 — Data acquisition guide (ajoutée — 5 étapes)

Étape par étape pour reproduire les données from scratch :

1. **Scrape SNEP**
   ```bash
   python scripts/scrape_snep.py
   ```
   → produit `data/snep_certifications.csv` (8 384 lignes, ~378 pages paginées).

2. **Fetch Last.fm features** (nécessite clé API gratuite sur last.fm/api)
   ```bash
   export LASTFM_API_KEY=xxx
   python scripts/fetch_lastfm_features.py
   ```
   → produit `data/lastfm_artist_features.csv` + `data/lastfm_similar_edges.csv`.

3. **Build enriched graph + embeddings**
   ```bash
   python scripts/build_enriched_graph.py --add-new-nodes --min-snep-anchors 2 --min-match 0.4
   ```
   → produit `data/graph_edges.csv`, `data/node_features.csv`, `data/node_embeddings.csv`, `data/node_labels.csv`.

4. **Train the 3 models**
   ```bash
   python scripts/train_enriched.py
   ```
   → produit `models/{kmeans,gmm,birch}.joblib`.

5. **Evaluate + launch app**
   ```bash
   python scripts/main.py
   ```
   → évalue + remplit `results/model_metrics.csv` + lance Streamlit.

### Section 7 — Deliverables

Pointe vers le dossier `deliverables/` contenant les 6 assignments markdown (assignment1.md … assignment6.md).

### Section 8 — Tech stack

Liste explicite : Python 3.11, NetworkX, python-louvain, Node2Vec custom, scikit-learn, Streamlit, joblib, BeautifulSoup + requests, Last.fm API.

## 3. Ce qui a été retiré par rapport au template initial

- Emojis (politique de cohérence avec le reste du repo en Session 6).
- Sections "TODO" et "FIXME" vides du template.
- Références au sujet original "Marvel characters" / "Snowflake" inadaptées.
- Le contenu en français qui doublonnait avec celui des deliverables.

## 4. Justification des choix éditoriaux

- **Anglais** : GitHub est international, la plupart des dépendances et leur doc sont en anglais — un README anglais reste consultable par n'importe quel évaluateur ou recruteur.
- **Guide en 5 étapes plutôt qu'un script unique** : permet de comprendre **chaque étape isolément** (utile pour debug ou pour reprendre à mi-pipeline si une étape a déjà été cachée). Le script `main.py` reste le shortcut pour lancer rapidement la démo.
- **Mention explicite de la clé Last.fm** : sans elle, l'étape 2 échoue → on l'indique clairement plutôt que de laisser l'utilisateur deviner.
- **Pas de section "License"** : POC pédagogique, pas de licence formelle requise — peut être ajoutée si besoin.

## 5. Versionning

Le README final a été committé sur la branche `main` dans le commit **`dd16c51`** (Session 6 final push). Vérifiable via :

```bash
git log --oneline -- README.md
```

Le repo est public et accessible sur GitHub.
