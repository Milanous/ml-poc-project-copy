from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# ─── Models ───────────────────────────────────────────────────────────────────
# Tâche : clustering non supervisé des communautés musicales (Node2Vec embeddings)
# Les 3 modèles sont entraînés sur les embeddings scalés (RobustScaler, dim=32).
# Chaque modèle est évalué sur X_test avec compute_metrics(y_test, model.predict(X_test)).
# y_test = labels de communauté Louvain (pseudo-vérité terrain).

MODELS = {
    "kmeans": {
        "name": "K-Means",
        "description": (
            "Baseline centroïde. k aligné sur le nombre de communautés Louvain. "
            "Suppose des clusters sphériques et de variance homogène."
        ),
        "path": MODELS_DIR / "kmeans.joblib",
    },
    "gmm": {
        "name": "Gaussian Mixture Model",
        "description": (
            "Modèle probabiliste EM. Gère les clusters elliptiques (covariance complète). "
            "Assignement souple : chaque artiste a une probabilité d'appartenance à chaque composante."
        ),
        "path": MODELS_DIR / "gmm.joblib",
    },
    "birch": {
        "name": "BIRCH",
        "description": (
            "Clustering hiérarchique par arbre de features (Balanced Iterative Reducing and Clustering). "
            "Pas d'hypothèse sphérique, complexité O(n), predict() natif via sous-clusters."
        ),
        "path": MODELS_DIR / "birch.joblib",
    },
}
