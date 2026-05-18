# SNEP Music Community Detection

Unsupervised clustering of French music artists into musical communities, based on a
co-artist collaboration graph built from **8,384 SNEP certifications** (French recording
industry awards) and enriched with **Last.fm artist metadata** (popularity and genre tags).

The pipeline builds a weighted co-artist graph, learns 32-dimensional **Node2Vec**
embeddings on it, concatenates them with 14 Last.fm features, and compares three
clustering models (**K-Means**, **Gaussian Mixture**, **BIRCH**) against
**Louvain** communities used as a pseudo–ground truth.

A **Streamlit dashboard** presents the business context, the EDA, the model
comparison, and an interactive recommendation demo (pick an artist, get the other
artists from the same predicted community).

## Repository Structure

```
data/         raw and processed datasets (CSV) — not all files are committed
deliverables/ written assignments (assignement1.md, ...)
logs/         runtime logs
models/       trained clustering models (.joblib)
notebooks/    eda_snep.ipynb, feature_engineering.ipynb
plots/        EDA, model comparison and demo figures
results/      model_metrics.csv (test-set evaluation)
scripts/      executable entry points (scrape, fetch, build, train, evaluate)
src/          project source code (data pipeline, models, app)
```

Key files:

- `src/data.py` — `load_dataset_split()`: full data pipeline (clean → graph → features → embeddings → split).
- `src/scraping.py` — SNEP scraper (378 paginated pages, parallel, deduplicated).
- `src/lastfm.py` — thin client around the Last.fm REST API (`artist.getInfo`, `artist.getSimilar`).
- `src/enrichment.py` — merges Last.fm features and similarity edges into the SNEP graph.
- `src/metrics.py` — `compute_metrics()`: ARI, NMI, silhouette, number of clusters.
- `src/model_io.py` — generic loader for `.joblib` / `.pkl`.
- `src/config.py` — paths and `MODELS` registry consumed by `scripts/main.py`.
- `src/app.py` — Streamlit dashboard (3 tabs: problem & EDA, models & results, demo).
- `scripts/main.py` — end-to-end evaluation: loads models, computes metrics, writes
  `results/model_metrics.csv`, then launches Streamlit.

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt` (NetworkX, scikit-learn, python-louvain,
  pandas, requests, streamlit, joblib, matplotlib, etc.)

## Installation

```bash
git clone git@github.com:Milanous/ml-poc-project-copy.git
cd ml-poc-project-copy

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project also reads `.env` via `python-dotenv`. Create one at the repo root with:

```text
PYTHONPATH=./src
LASTFM_API_KEY=your_lastfm_api_key
```

The Last.fm API key is only needed to **regenerate** the enrichment data
(see "Data Acquisition" below). It is **not** required to run the dashboard if
the cached CSV files in `data/` are already present.

## Run the Project

```bash
python scripts/main.py
```

This will:

1. validate `src/app.py` and the model registry,
2. load the cached split via `src.data.load_dataset_split()`,
3. evaluate every model registered in `MODELS` (`models/{kmeans,gmm,birch}.joblib`),
4. write the metrics to `results/model_metrics.csv`,
5. print them to the terminal,
6. launch the Streamlit dashboard at <http://localhost:8501>.

## Models

| Key      | Name                    | Notes                                                                 |
|----------|-------------------------|-----------------------------------------------------------------------|
| `kmeans` | K-Means                 | Centroid baseline, `k` aligned with the number of Louvain communities.|
| `gmm`    | Gaussian Mixture Model  | Soft assignments, full covariance.                                    |
| `birch`  | BIRCH                   | Hierarchical, no sphericity assumption, best ARI on this dataset.     |

Test-set metrics are stored in `results/model_metrics.csv` after each run.

## Data Acquisition

The committed CSV files in `data/` are sufficient to run `scripts/main.py` and the
Streamlit app. If you want to **regenerate everything from scratch**, follow the
steps below in order.

### 1. Scrape SNEP certifications

Pulls all 378 paginated pages from snepmusique.com, deduplicates by
`(artist, title, category, certification)`, and exports to CSV.

```bash
python -m src.scraping
# or
python scripts/scrape_snep.py
```

Output: `data/snep_certifications.csv` (~8,400 rows).

### 2. Fetch Last.fm features (optional but recommended)

Calls the Last.fm API for every SNEP artist to retrieve popularity (`listeners`,
`playcount`), genre tags, and a small similarity neighborhood. Requires
`LASTFM_API_KEY` in `.env`.

```bash
python scripts/fetch_lastfm_features.py
```

Outputs:

- `data/lastfm_artist_features.csv` — per-artist popularity and tag features.
- `data/lastfm_similar_edges.csv` — similar-artist edges from Last.fm.

This step is rate-limited and can take 30+ minutes the first time. Re-runs are
incremental (cached responses are kept).

### 3. Build the enriched graph and feature matrices

Merges SNEP collaborations with Last.fm features and similarity edges, applies the
anchor filter, and writes the cached files consumed by `load_dataset_split()`.

```bash
python scripts/build_enriched_graph.py \
    --add-new-nodes \
    --min-snep-anchors 2 \
    --min-match 0.4
```

Outputs in `data/`:

- `graph_edges.csv`
- `node_features.csv`
- `node_embeddings.csv` (32 Node2Vec + 14 Last.fm dims)
- `node_labels.csv` (Louvain communities)

### 4. Train the three clustering models

```bash
python scripts/train_enriched.py
```

Outputs: `models/kmeans.joblib`, `models/gmm.joblib`, `models/birch.joblib`.

### 5. Evaluate and launch the app

```bash
python scripts/main.py
```

This writes `results/model_metrics.csv` and starts Streamlit at
<http://localhost:8501>.

## Reproducibility Notes

- Every random step (`train_test_split`, Node2Vec walks, Louvain, K-Means init,
  GMM, t-SNE) is seeded with `random_state=42`.
- The cached embedding CSVs are stored unscaled; `RobustScaler` is re-fit at load
  time, both during training (`scripts/train_enriched.py`) and during evaluation
  (`src/data.py::load_dataset_split`), so there is no double-scaling.
- BIRCH pickle loading requires `sys.setrecursionlimit(50000)` for the CFTree
  on >5,000 points (already handled in `scripts/train_enriched.py`).
